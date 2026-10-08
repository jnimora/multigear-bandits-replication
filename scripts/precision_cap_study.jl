module PrecisionCapStudy
using TOML, Dates, SHA, Statistics
using ..ManyProjectAllocator, ..ManyProjectModel, ..ManyProjectSimulation, ..PrecisionCapSimulation
using ..PrecisionCapModel, ..RayleighLargeCap
const MA=ManyProjectAllocator;const MM=ManyProjectModel;const MS=ManyProjectSimulation
const PC=PrecisionCapSimulation;const CM=PrecisionCapModel;const LC=RayleighLargeCap
const Q=Rational{BigInt};const REVISION="manyproject-precision-cap-pilot-v1"
const LOCK=ReentrantLock()
function progress(out,msg)
    lock(LOCK) do
        line=string(now(UTC))*" UTC "*msg
        open(joinpath(out,"progress.log"),"a") do io;println(io,line);flush(io);end
        println(line);flush(stdout)
    end
end
function config(path)
    c=TOML.parsefile(path)
    c["revision"]==REVISION && !c["production_enabled"] && c["master_seed"]==2026092402 || error("Only the frozen precision/cap pilot is admitted.")
    Symbol.(c["policies"])==collect(MA.POLICIES) && c["base_population"]==50 && c["maximum_population"]==1600 || error("Policies/composition changed.")
    c["allocator"]==Dict("max_states"=>200000,"max_updates"=>20000000,"seconds_per_call"=>30.0) || error("Accepted allocator limits changed.")
    p=c["precision"];p["replications"]==16 && p["populations"]==[100,400,1600] && p["burn_in"]==1000 && p["horizon"]==4000 || error("Prospective precision schedule changed.")
    t=c["cap"];t["replications"]==8 && t["caps"]==[64,128,256,512,1024,2048] && t["primary_population"]==50 && t["extra_population"]==200 && t["burn_in"]==10000 && t["horizon"]==20000 || error("Prospective cap schedule changed.")
    all(t[k]==[512,2048] for k in ("extra_population_caps","initial_check_caps","window_check_caps")) || error("Cap checks changed.")
    t["budget_per_project"]=="12862086153/13727789219840" && t["source_group"]=="persistent_weighted" || error("Persistent primitives or budget series changed.")
    [f["id"] for f in c["family"]]==["binary_asymptotic_control","anchor_H16_binding","anchor_H32_binding","anchor_H16_slack","three_source_H16_binding","three_source_H32_binding"] || error("Family inventory changed.")
    c
end
limits(c)=MA.Limits(c["allocator"]["max_states"],c["allocator"]["max_updates"],Float64(c["allocator"]["seconds_per_call"]))
window(name,b,T)=Dict{String,Any}("name"=>name,"burn"=>b,"T"=>T)
function precision_jobs(c)
    out=Dict{String,Any}[];p=c["precision"]
    for f in c["family"],L in p["populations"]
        initials=L==400 && f["precision_checks"] ? ["zero","cap","relaxed"] : ["zero"]
        for initial in initials
            windows=[window("base",p["burn_in"],p["horizon"])]
            if L==400 && f["precision_checks"] && initial=="zero"
                push!(windows,window("double_burn",2*p["burn_in"],p["horizon"]))
                push!(windows,window("double_horizon",p["burn_in"],2*p["horizon"]))
            end
            for rep in 1:p["replications"]
                id=f["id"]*"_L$(L)_$(initial)_r$(rep)"
                push!(out,Dict("id"=>id,"family"=>f["id"],"L"=>L,"rep"=>rep,"initial"=>initial,"windows"=>windows,"replications"=>p["replications"]))
            end
        end
    end
    out
end
function cap_spec(H,c)
    Dict{String,Any}("id"=>"persistent_H$(H)_binding","source_group"=>c["cap"]["source_group"],"H"=>H,
        "source_states"=>2,"wR"=>"1023/1024","success"=>["0","1/1024","3/1024"],"kappa"=>"2","epsilon"=>"1/8",
        "weights"=>["1","2"],"base_counts"=>[25,25],"budget_per_project"=>c["cap"]["budget_per_project"])
end
function cap_jobs(c,H)
    p=c["cap"];out=Dict{String,Any}[];family="persistent_H$(H)_binding"
    populations=H in p["extra_population_caps"] ? [p["primary_population"],p["extra_population"]] : [p["primary_population"]]
    for L in populations
        initials=L==p["primary_population"] && H in p["initial_check_caps"] ? ["zero","cap"] : ["zero"]
        for initial in initials
            windows=[window("base",p["burn_in"],p["horizon"])]
            if L==p["primary_population"] && initial=="zero" && H in p["window_check_caps"]
                push!(windows,window("double_burn",2*p["burn_in"],p["horizon"]))
                push!(windows,window("double_horizon",p["burn_in"],2*p["horizon"]))
            end
            for rep in 1:p["replications"]
                push!(out,Dict("id"=>family*"_L$(L)_$(initial)_r$(rep)","family"=>family,"L"=>L,"rep"=>rep,
                    "initial"=>initial,"windows"=>windows,"replications"=>p["replications"]))
            end
        end
    end
    out
end
function write_csv(path,rows,columns)
    escape(x)=begin s=string(x);occursin(r"[,\"\n]",s) ? "\""*replace(s,"\""=>"\"\"")*"\"" : s;end
    open(path,"w") do io
        println(io,join(columns,','))
        for row in rows;println(io,join((haskey(row,c) ? escape(row[c]) : "" for c in columns),','));end
    end
end
function one_job(f,job,c,out,stage)
    p=c[stage=="caps" ? "cap" : "precision"]
    onwin=function(r)
        r["planned_replications"]=job["replications"];r["job_id"]=job["id"]*"_"*r["window"]
        r["trajectory_job_id"]=job["id"];r["stage"]=stage
        LC.write_record(joinpath(out,"rep_"*r["job_id"]),r)
    end
    onfail=function(r)
        d=copy(r);delete!(d,"records");LC.write_record(joinpath(out,"partial_"*job["id"]),d)
    end
    progress(out,"Trajectory "*job["id"]*" START")
    result=PC.run_windows(f,job["L"],job["rep"],job["windows"];initial=job["initial"],master_seed=c["master_seed"],
       limits=limits(c),seconds=Float64(p["max_replication_seconds"]),on_window=onwin,on_failure=onfail,
       heartbeat=(t,n)->(t%10000==0 ? progress(out,job["id"]*" period $(t)/$(n)") : nothing))
    result["job_id"]=job["id"]
    d=copy(result);delete!(d,"records");LC.write_record(joinpath(out,"job_"*job["id"]),d)
    progress(out,"Trajectory "*job["id"]*" "*result["status"])
    result
end
# Fixed-size worker pool. Each function invocation owns one job's workspaces.
# No task captures a mutable collector binding assigned later in this scope.
function job_worker(ch,jobs,families,c,out,stage)
    local results=Dict{String,Any}[]
    for i in ch
        job=jobs[i]
        push!(results,one_job(families[job["family"]],job,c,out,stage))
    end
    results
end
function run_jobs(jobs,families,c,out,stage)
    isempty(jobs) && return Dict{String,Any}[]
    ch=Channel{Int}(length(jobs));for i in eachindex(jobs);put!(ch,i);end;close(ch)
    nw=min(Threads.nthreads(:default),length(jobs));tasks=[Threads.@spawn job_worker(ch,jobs,families,c,out,stage) for _ in 1:nw]
    results=Dict{String,Any}[];for t in tasks;append!(results,fetch(t));end
    sort!(results;by=r->r["job_id"])
    length(results)==length(jobs) && Set(r["job_id"] for r in results)==Set(j["id"] for j in jobs) || error("Trajectory result inventory mismatch.")
    results
end
function expected_windows(jobs)
    [Dict("job_id"=>j["id"]*"_"*w["name"],"family"=>j["family"],"L"=>j["L"],"replication"=>j["rep"],
      "initial_law"=>j["initial"],"burn_in"=>w["burn"],"horizon"=>w["T"],"replications"=>j["replications"]) for j in jobs for w in j["windows"]]
end
function report(c,out,stage,jobs,outcomes;extras=Dict{String,Any}())
    records=Dict{String,Any}[]
    for r in outcomes;append!(records,get(r,"records",Dict{String,Any}[]));end
    expected=expected_windows(jobs);byid=Dict(e["job_id"]=>e for e in expected)
    length(unique(r["job_id"] for r in records))==length(records) || error("Duplicated saved window.")
    for r in records
        haskey(byid,r["job_id"]) || error("Unexpected window identity.")
        e=byid[r["job_id"]]
        all(r[k]==e[k] for k in ("family","L","replication","initial_law","burn_in","horizon")) || error("Saved window not in prospective plan.")
        r["policy_names"]==collect(String.(MA.POLICIES)) || error("Lost policy in saved window.")
    end
    summaries,pairs=MS.summarize(records)
    raw=Dict{String,Any}[];usage=Dict{String,Any}[]
    for r in records,p in r["policy_results"]
        push!(raw,merge(Dict("family"=>r["family"],"L"=>r["L"],"job_id"=>r["job_id"],"window"=>r["window"]),p))
        for phase in ("burn_in","retained")
            u=p[phase*"_allocator"]
            row=Dict{String,Any}("family"=>r["family"],"L"=>r["L"],"job_id"=>r["job_id"],"policy"=>p["policy"],"phase"=>phase,
               "calls"=>u["calls"],"elapsed_seconds_diagnostic"=>u["elapsed_seconds_diagnostic"],"maximum_call_seconds_diagnostic"=>u["maximum_call_seconds_diagnostic"],
               "work"=>u["work"],"peak_states"=>u["peak_reported_states"],"selective_exact_comparisons"=>u["selective_exact_comparisons"])
            for engine in ("ordered_exact","dual_core_dp_exact","integer_dp_exact","exposed_priority_exact","cap_ordered_verified","cap_active_exact")
                row[engine]=get(u["engine_calls"],engine,0)
            end
            for engine in ("dual_core_dp_exact","integer_dp_exact")
                row["underlying_"*engine]=get(u["underlying_engine_calls"],engine,0)
            end
            push!(usage,row)
        end
    end
    cols=vcat(["family","L","initial_law","burn_in","horizon","policy","replications","bound_per_project","lambda_star"],
        vcat([[x*"_mean",x*"_se"] for x in MS.METRICS]...))
    write_csv(joinpath(out,"summary.csv"),summaries,cols)
    write_csv(joinpath(out,"paired_policy_differences.csv"),pairs,["family","L","initial_law","burn_in","horizon","comparison","replications","mean_cost_difference_per_project","paired_monte_carlo_se"])
    write_csv(joinpath(out,"replications.csv"),raw,vcat(["family","L","job_id","window","policy","replication","initial_law","burn_in","retained_periods","transition_seed","initial_seed"],collect(MS.METRICS),["realized_martingale_sd_scale_per_project","exact_pathwise_identity"]))
    write_csv(joinpath(out,"allocator_usage.csv"),usage,["family","L","job_id","policy","phase","calls","elapsed_seconds_diagnostic","maximum_call_seconds_diagnostic","work","peak_states","selective_exact_comparisons","ordered_exact","dual_core_dp_exact","integer_dp_exact","exposed_priority_exact","cap_ordered_verified","cap_active_exact","underlying_dual_core_dp_exact","underlying_integer_dp_exact"])
    # Paired cap/start/window diagnostics, preserving exactly shared shocks and
    # recording all missing partners. No pooling of correlated variants as reps.
    comparisons=paired_design(records,c,stage,jobs)
    LC.write_record(joinpath(out,"paired_design"),Dict("comparisons"=>comparisons))
    write_csv(joinpath(out,"paired_design.csv"),comparisons,["comparison","L","policy","metric","complete","replications","mean_difference","paired_monte_carlo_se","scope"])
    missing=[e["job_id"] for e in expected if !any(r->r["job_id"]==e["job_id"],records)]
    error_seen=any(r->get(r,"status","")=="error",outcomes) || get(extras,"unexpected_error",false)
    ok=length(outcomes)==length(jobs) && isempty(missing) && all(r->r["passed"],outcomes) && get(extras,"gates_passed",true)
    status=ok ? "PASS" : (error_seen ? "FAILED" : "REVIEW REQUIRED")
    review=Dict{String,Any}("revision"=>REVISION,"stage"=>stage,"passed"=>ok,"status"=>status,"julia_version"=>string(VERSION),"compute_threads"=>Threads.nthreads(:default),"architecture"=>string(Sys.ARCH),"production_authorized"=>false,
      "planned_trajectories"=>length(jobs),"completed_trajectories"=>count(r->r["passed"],outcomes),
      "planned_windows"=>length(expected),"complete_windows"=>length(records),"missing_window_ids"=>missing,
      "policy_replication_rows"=>length(raw),"summary_rows"=>length(summaries),"policies"=>collect(String.(MA.POLICIES)),
      "computational_success_is_not_stationary_precision_acceptance"=>true,"no_pilot_production_pooling"=>true,
      "numerical_enclosure_or_exact_comparisons"=>true,"replications_are_statistical_units"=>true,"diagnostic_targets_for_review_only"=>c["diagnostic_targets"],
      "cap_windows_share_shocks_not_independent_samples"=>true,"master_seed"=>c["master_seed"],"extras"=>extras)
    LC.write_record(joinpath(out,"review"),review);println("Precision/cap ",stage,": ",status)
    review
end
function paired_design(records,c,stage,jobs)
    groups=Dict{Tuple{String,Int,String,Int,Int},Dict{Int,Dict{String,Any}}}()
    for r in records
        key=(r["family"],r["L"],r["initial_law"],r["burn_in"],r["horizon"])
        d=get!(groups,key,Dict{Int,Dict{String,Any}}());haskey(d,r["replication"]) && error("Duplicate paired record.")
        d[r["replication"]]=r
    end
    planned=Set((e["family"],e["L"],e["initial_law"],e["burn_in"],e["horizon"]) for e in expected_windows(jobs))
    links=Tuple{String,Any,Any}[]
    # Only compare adjacent caps at identical load, L, start and window.
    capgroups=stage=="caps" ? [["persistent_H$(H)_binding" for H in c["cap"]["caps"]]] :
       [["anchor_H16_binding","anchor_H32_binding"],["three_source_H16_binding","three_source_H32_binding"]]
    for chain in capgroups,j in 2:length(chain)
        for key in planned
            key[1]==chain[j-1] || continue
            other=(chain[j],key[2],key[3],key[4],key[5])
            other in planned && push!(links,(chain[j]*"_minus_"*chain[j-1],key,other))
        end
    end
    for key in planned
        key[3]=="zero" || continue
        b=stage=="caps" ? c["cap"]["burn_in"] : c["precision"]["burn_in"]
        T=stage=="caps" ? c["cap"]["horizon"] : c["precision"]["horizon"]
        key[4]==b && key[5]==T || continue
        for (tag,other) in (("double_burn",(key[1],key[2],"zero",2b,T)),("double_horizon",(key[1],key[2],"zero",b,2T)),
               ("cap_start",(key[1],key[2],"cap",b,T)),("relaxed_start",(key[1],key[2],"relaxed",b,T)))
            other in planned && push!(links,(key[1]*"_"*tag*"_minus_base",key,other))
        end
    end
    R=stage=="caps" ? c["cap"]["replications"] : c["precision"]["replications"]
    out=Dict{String,Any}[]
    for (tag,k1,k2) in sort!(links;by=x->string(x))
        d1=get(groups,k1,Dict());d2=get(groups,k2,Dict());complete=Set(keys(d1))==Set(1:R)==Set(keys(d2))
        for p in MA.POLICIES,metric in ("cost_per_project","raw_bound_gap_per_project","residual_gap_per_project","cap_fraction_per_project","shadow_minus_capped_cost_per_project")
            row=Dict{String,Any}("comparison"=>tag,"L"=>k1[2],"policy"=>String(p),"metric"=>metric,"complete"=>complete,
               "replications"=>R,"scope"=>"paired same-replication same-transition-stream difference; cap/start/window partners not independent replications")
            if complete
                diffs=Float64[]
                for r in 1:R
                    d1[r]["transition_seed"]==d2[r]["transition_seed"] || error("Pair lost common transition stream.")
                    x=only(filter(q->q["policy"]==String(p),d1[r]["policy_results"]));y=only(filter(q->q["policy"]==String(p),d2[r]["policy_results"]))
                    push!(diffs,Float64(MM.parseq(y[metric*"_exact"])-MM.parseq(x[metric*"_exact"])))
                end
                s=MS.stats(diffs);row["mean_difference"]=s.mean;row["paired_monte_carlo_se"]=s.se
            end
            push!(out,row)
        end
    end
    out
end
end # module
