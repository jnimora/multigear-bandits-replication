# Include after MultiGearBandits. No third-party packages are required.
using TOML
using Random
using Statistics
using LinearAlgebra
using SHA
using Dates

const PERFORMANCE_ALGORITHMS=(:C,:R,:M,:FP)
const PERFORMANCE_PHASES=(:initialization,:index_loop,:end_to_end)

function pilot_configuration(path)
    c=TOML.parsefile(path)
    get(c,"protocol","")=="fair-performance-pilot-v1" || error("Unsupported pilot protocol.")
    for key in ("states","gears","draws","criteria","eta")
        !isempty(c[key]) && length(unique(c[key]))==length(c[key]) ||
            error("Configuration $key must be nonempty and contain no duplicates.")
    end
    all(n->n isa Integer && n>=2,c["states"]) || error("Invalid state sizes.")
    all(a->a isa Integer && a>=1,c["gears"]) || error("Invalid gears.")
    all(a->a in ("discounted","average"),c["criteria"]) || error("Invalid criterion.")
    all(e->isfinite(e) && 0<=e<=1,c["eta"]) || error("Invalid perturbations.")
    0<c["beta"]<1 || error("Invalid discount.")
    c["repetitions"]>=1 && c["warmups"]>=2 || error("Require repetitions>=1 and warmups>=2.")
    length(unique(c["states"]))==length(c["states"]) || error("Repeated state size.")
    length(unique(c["gears"]))==length(c["gears"]) || error("Repeated gear count.")
    length(unique(c["draws"]))==length(c["draws"]) && all(x->x isa Integer && x>=1,c["draws"]) || error("Invalid draw IDs.")
    isfinite(c["memory_budget_gib"]) && isfinite(c["max_case_seconds"]) &&
        c["memory_budget_gib"]>0 && c["max_case_seconds"]>0 || error("Invalid resource guard.")
    c["terminal_update"] isa Bool || error("terminal_update must be Boolean.")
    isfinite(c["audit_tolerance"]) && c["audit_tolerance"]>0 || error("Invalid audit tolerance.")
    return c
end

function pilot_options(c)
    t=c["screening"]
    return PerformanceOptions(comparison_atol=t["comparison_atol"],comparison_rtol=t["comparison_rtol"],
        pivot_atol=t["pivot_atol"],pivot_rtol=t["pivot_rtol"],exact_fallback=t["exact_fallback"],
        exact_max_states=t["exact_max_states"],terminal_update=c["terminal_update"],
        residual_fallback=get(t,"residual_fallback",false),
        residual_max_states=get(t,"residual_max_states",256))
end

function pilot_kwargs(c,criterion)
    return criterion==:discounted ? (;criterion=criterion,beta=Float64(c["beta"])) : (;criterion=criterion)
end

function pilot_git(root,args...)
    try
        return strip(read(Cmd(["git","-C",root,args...]),String))
    catch
        return "unavailable"
    end
end
file_sha(path)=isfile(path) ? bytes2hex(sha256(read(path))) : "missing"

function pilot_source_hashes(root)
    hashes=Dict{String,String}()
    for sub in ("src","scripts","configs")
        isdir(joinpath(root,sub)) || continue
        for (dir,_,files) in walkdir(joinpath(root,sub)), file in sort(files)
            (endswith(file,".jl") || endswith(file,".toml")) || continue
            full=joinpath(dir,file); islink(full) && error("Do not benchmark linked source: $full")
            hashes[relpath(full,root)]=file_sha(full)
        end
    end
    return hashes
end

function pilot_metadata(root,config_path,c)
    versionfile=joinpath(root,"julia-version")
    isfile(versionfile) && strip(read(versionfile,String))==string(VERSION) ||
        error("Running Julia does not match the repository's julia-version.")
    active=Base.active_project()
    active!==nothing && realpath(active)==realpath(joinpath(root,"Project.toml")) ||
        error("Run from the repository with --project=.")
    Threads.nthreads(:default)==1 && Threads.nthreads(:interactive)==0 || error(
        "Controlled timing requires --threads=1,0, not automatic threading.")
    BLAS.get_num_threads()==1 || error("BLAS must use one thread.")
    return Dict{String,Any}(
        "protocol"=>c["protocol"],"kind"=>"local engineering pilot; not paper results",
        "implementation_revision"=>"optimized-initialization-v1",
        "fp_initializer"=>"optimized",
        "preparation_checks"=>"all freshly solved columns; backward tolerance 1e-10",
        "julia_version"=>string(VERSION),"recorded_version"=>strip(read(versionfile,String)),
        "julia_threads"=>Threads.nthreads(:default),"interactive_threads"=>Threads.nthreads(:interactive),
        "blas_threads"=>BLAS.get_num_threads(),"blas_configuration"=>sprint(show,BLAS.get_config()),
        "os"=>string(Sys.KERNEL),"architecture"=>string(Sys.ARCH),"cpu_name"=>Sys.CPU_NAME,
        "logical_cpus"=>Sys.CPU_THREADS,"machine"=>Sys.MACHINE,
        "git_commit"=>pilot_git(root,"rev-parse","HEAD"),"git_status"=>pilot_git(root,"status","--porcelain"),
        "project_sha256"=>file_sha(joinpath(root,"Project.toml")),
        "manifest_sha256"=>file_sha(joinpath(root,"Manifest.toml")),"config_sha256"=>file_sha(config_path),
        "source_sha256"=>pilot_source_hashes(root),"started_utc"=>string(now(UTC)),
        "power_management_note"=>get(c,"power_management_note","not recorded"),
        "gc_policy"=>"enabled; full collection before each measured phase, outside its timer",
        "sample_policy"=>"one evaluation on fresh state; compilation-positive samples ineligible",
        "terminal_update"=>c["terminal_update"],"configuration"=>c)
end

# Complete and validate text representation before touching an existing report.
# No numerical kernel or measured expression calls this reporting helper.
if !isdefined(@__MODULE__, :PilotTOMLText)
    include(joinpath(@__DIR__,"toml_text_support.jl"))
end
function pilot_write_toml(path,object)
    return PilotTOMLText.write_report(path,object)
end

function pilot_archive_primitives(dir,p)
    path=joinpath(dir,"primitives_N$(p.n)_A$(p.A)_draw$(p.draw).toml")
    object=Dict{String,Any}("N"=>p.n,"A"=>p.A,"draw"=>p.draw,"master_seed"=>p.master_seed,
        "layout"=>"flattened arrays use Julia column-major order", "q"=>vec(p.q),"theta"=>vec(p.theta),
        "b"=>p.b,"positive_transition_weights"=>vec(p.weights),"Pi"=>vec(p.Pi),"delta"=>p.delta)
    pilot_write_toml(path,object)
    return file_sha(path)
end
function pilot_archive_model(dir,id,model)
    path=joinpath(dir,id*".toml")
    pilot_write_toml(path,Dict("N"=>nstates(model),"A"=>maxgear(model),"P"=>vec(model.P),
        "h"=>vec(model.h),"c"=>vec(model.c),"controllable"=>collect(model.controllable),
        "layout"=>"P[N,N,A+1],h[N,A+1],c[N,A+1],column-major; floating stored-array model"))
    return file_sha(path)
end

function pilot_snapshot_record(s)
    record=Dict{String,Any}("algorithm"=>string(s.algorithm),"status"=>string(s.status),
        "complete"=>s.complete,"states"=>s.states,"order"=>[[i,a] for (i,a) in s.order],
        "assigned_ratio"=>s.ratio,"marginal_f"=>s.f,"marginal_g"=>s.g,"dimensions"=>s.dimension,
        "comparison_evidence"=>string.(s.evidence),"counts"=>s.counts,
        "terminal_update_skipped"=>s.terminal_update_skipped)
    if hasproperty(s,:residual_comparisons) && !isempty(s.residual_comparisons)
        record["residual_comparisons"]=s.residual_comparisons
    end
    return record
end

# Measurement functions are specialized on the workspace type by the function
# barrier. Metadata creation and owned snapshots are outside index_loop timing.
function pilot_timed_loop(run)
    return @timed run_performance!(run)
end
function pilot_end_to_end(input,algorithm,kw,options)
    run=prepare_performance_run(input,algorithm;kw...,options=options)
    run_performance!(run)
    return performance_snapshot(run)
end

function pilot_phase(input,algorithm,kw,options,phase;collect_gc=true)
    if phase==:initialization
        collect_gc && GC.gc()
        t=@timed prepare_performance_run(input,algorithm;kw...,options=options)
        run=t.value
        # Neither running the loop nor calculating size is inside the init timer.
        retained=Base.summarysize(run)
        run_performance!(run)
        snapshot=performance_snapshot(run)
    elseif phase==:index_loop
        run=prepare_performance_run(input,algorithm;kw...,options=options)
        retained=Base.summarysize(run)
        collect_gc && GC.gc()
        t=pilot_timed_loop(run)
        snapshot=performance_snapshot(run)
    elseif phase==:end_to_end
        collect_gc && GC.gc()
        t=@timed pilot_end_to_end(input,algorithm,kw,options)
        snapshot=t.value; retained=-1
    else
        error("Unknown measurement phase.")
    end
    # The following documented raw counters are exposed instead of an assumed
    # universal allocation-count formula. A separate @allocations check supplies
    # total allocation counts on a fresh identical run.
    gc=Dict{String,Int}()
    for name in (:malloc,:realloc,:poolalloc,:bigalloc,:pause,:full_sweep)
        hasproperty(t.gcstats,name) && (gc[string(name)]=Int(getproperty(t.gcstats,name)))
    end
    row=Dict{String,Any}("phase"=>string(phase),"algorithm"=>string(algorithm),
        "seconds"=>t.time,"allocated_bytes"=>Int(t.bytes),"gc_seconds"=>t.gctime,
        "compile_seconds"=>t.compile_time,"recompile_seconds"=>t.recompile_time,
        "retained_bytes_including_model"=>retained,
        "steady_state"=>t.compile_time==0 && t.recompile_time==0,
        "status"=>string(snapshot.status),"complete"=>snapshot.complete,"raw_gc_counters"=>gc,
        "counts"=>snapshot.counts)
    return row,snapshot
end

function pilot_csv(path,rows,columns)
    function cell(x)
        s=string(x)
        if any(ch->ch in (',','"','\n','\r'),s)
            return string('"',replace(s,"\""=>"\"\""),'"')
        end
        return s
    end
    open(path,"w") do io
        println(io,join(columns,","))
        for row in rows; println(io,join([cell(get(row,c,"")) for c in columns],",")); end
    end
end

# Reporting only: build grouping/pairing indexes once per report.  The public
# one-argument call retains its original output and ordering.  The keyword path
# avoids row copies and unused ratios in the execution-only summary.
function pilot_summaries(rows; eligibility_key="eligible", include_ratios=true)
    summary=Dict{String,Any}[]; ratios=Dict{String,Any}[]
    groups=Dict{Tuple,Vector{Any}}()
    fp_matches=Dict{Tuple,Vector{Any}}()
    for r in rows
        key=(r["case_id"],r["algorithm"],r["phase"])
        group=get!(groups,key) do
            Any[]
        end
        push!(group,r)
        if include_ratios && r["algorithm"]=="FP"
            pairkey=(r["case_id"],r["phase"],r["repetition"])
            matches=get!(fp_matches,pairkey) do
                Any[]
            end
            # Preserve multiplicity: duplicate FP observations suppress a pair;
            # do not silently select the first/last or only the eligible one.
            push!(matches,r)
        end
    end
    # Keep the old lexicographic summary order, never Dict iteration order.
    for (id,alg,phase) in sort!(collect(keys(groups)))
        group=groups[(id,alg,phase)]
        ok=[r for r in group if get(r,eligibility_key,false)]
        row=Dict{String,Any}("case_id"=>id,"algorithm"=>alg,"phase"=>phase,
            "attempts"=>count(r->get(r,"executed",false),group),"eligible"=>length(ok))
        if !isempty(ok)
            ts=[r["seconds"] for r in ok]
            row["median_seconds"]=median(ts); row["q25_seconds"]=quantile(ts,0.25)
            row["q75_seconds"]=quantile(ts,0.75)
            row["median_allocated_bytes"]=median([r["allocated_bytes"] for r in ok])
            row["median_gc_seconds"]=median([r["gc_seconds"] for r in ok])
        end
        push!(summary,row)
    end
    include_ratios || return summary,ratios
    # A ratio is paired by INSTANCE, CRITERION, PHASE, and REPETITION.
    # Preserve original baseline-row order, including repeated baseline rows.
    # No unpaired successful-only pooling, no imputed time for a failed method.
    for r in rows
        r["algorithm"]=="FP" && continue
        matches=get(fp_matches,(r["case_id"],r["phase"],r["repetition"]),nothing)
        (matches===nothing || length(matches)!=1) && continue
        f=only(matches); eligible=get(r,eligibility_key,false) && get(f,eligibility_key,false)
        row=Dict{String,Any}("case_id"=>r["case_id"],"baseline"=>r["algorithm"],
            "phase"=>r["phase"],"repetition"=>r["repetition"],"eligible_pair"=>eligible)
        if eligible && r["seconds"]>0 && f["seconds"]>0
            row["baseline_over_FP"]=r["seconds"]/f["seconds"]
        end
        push!(ratios,row)
    end
    return summary,ratios
end

# Conservative admission GUARD, not measured peak memory. Includes model,
# primitive archive arrays, full inverse buffers, and several initialization
# temporaries. A hard operating-system memory/time limit is NOT implemented by
# this local pilot. The paper's production jobs need external enforcement.
function pilot_memory_estimate(n,A)
    return 8.0*n*n*(12.0*(A+1)+48.0)+16384.0*n*(A+1)
end

function run_performance_pilot(root,config_path)
    c=pilot_configuration(config_path); options=pilot_options(c)
    meta=pilot_metadata(root,config_path,c)
    # Unique run directory; never overwrite an earlier report or reference data.
    local_root=joinpath(root,"results","local"); mkpath(local_root)
    dir=mktempdir(local_root;prefix="fair_performance_",cleanup=false)
    inputdir=joinpath(dir,"inputs"); mkpath(inputdir)
    pilot_write_toml(joinpath(dir,"metadata.toml"),meta)
    rows=Dict{String,Any}[]; casereports=Dict{String,Any}[]
    inputs_seen=Dict{String,String}(); overall_ok=true
    println("Local fair-performance engineering pilot; NOT production evidence.")
    println("Output: ",dir)
    println("Endpoint: ",c["terminal_update"] ? "include all K numerical updates" : "Task I, omit unused terminal numerical update uniformly")
    for n in c["states"], A in c["gears"], draw in c["draws"]
        if pilot_memory_estimate(n,A)>c["memory_budget_gib"]*2.0^30
            push!(casereports,Dict("N"=>n,"A"=>A,"draw"=>draw,"status"=>"memory_guard_skip"))
            overall_ok=false; continue
        end
        local prepared,p,primhash
        try
            prepared=@timed performance_primitives(n,A,draw;master_seed=c["master_seed"])
            p=prepared.value; primhash=pilot_archive_primitives(inputdir,p)
        catch err
            err isa InterruptException && rethrow()
            push!(casereports,Dict("N"=>n,"A"=>A,"draw"=>draw,"status"=>"primitive_preparation_error",
                "error"=>sprint(showerror,err,catch_backtrace())))
            pilot_write_toml(joinpath(dir,"cases.toml"),Dict("case"=>casereports))
            overall_ok=false; continue
        end
        for eta in c["eta"], criterionstr in c["criteria"]
            criterion=Symbol(criterionstr); kw=pilot_kwargs(c,criterion)
            id="N$(n)_A$(A)_draw$(draw)_eta$(eta)_$(criterionstr)"
            started=time()
            local made,input
            try
                made=@timed PerformanceInput(performance_model(p,eta)); input=made.value
            catch err
                err isa InterruptException && rethrow()
                push!(casereports,Dict("case_id"=>id,"status"=>"model_preparation_error",
                    "primitive_sha256"=>primhash,"error"=>sprint(showerror,err,catch_backtrace())))
                pilot_write_toml(joinpath(dir,"cases.toml"),Dict("case"=>casereports))
                overall_ok=false; continue
            end
            modelid="N$(n)_A$(A)_draw$(draw)_eta$(eta)"
            modelhash=get!(inputs_seen,modelid) do
                pilot_archive_model(inputdir,modelid,input.model)
            end
            case=Dict{String,Any}("case_id"=>id,"N"=>n,"A"=>A,"draw"=>draw,"eta"=>eta,
                "criterion"=>criterionstr,"primitive_sha256"=>primhash,"model_sha256"=>modelhash,
                "primitive_preparation_seconds"=>prepared.time,"common_model_preparation_seconds"=>made.time,
                "shared_model_bytes"=>Base.summarysize(input.model),"status"=>"started")
            references=Dict{Symbol,Any}(); valid=Dict{Symbol,Bool}(); validation=Dict{String,Any}()
            # Direct-policy audits for EVERY method and EVERY selection policy.
            for alg in PERFORMANCE_ALGORITHMS
                try
                    a=audit_performance_run(input,alg;kw...,options=options,tol=c["audit_tolerance"])
                    references[alg]=a.snapshot
                    valid[alg]=a.passed
                    validation[string(alg)]=Dict("audit_passed"=>a.passed,
                        "selection_policies_checked"=>a.selection_policies_checked,
                        "max_scaled_error"=>a.max_scaled_error,"result"=>pilot_snapshot_record(a.snapshot))
                catch err
                    err isa InterruptException && rethrow()
                    valid[alg]=false
                    validation[string(alg)]=Dict("audit_passed"=>false,"error"=>sprint(showerror,err,catch_backtrace()))
                end
            end
            # Matching complete paths/MPIs is a condition for speedup eligibility,
            # not a reason to regenerate inputs. Audit failures remain archived.
            reference_exists=haskey(references,:C)
            reference_valid=reference_exists && valid[:C]
            same=Dict{Symbol,Bool}()
            for alg in PERFORMANCE_ALGORITHMS
                same[alg]=reference_valid && haskey(references,alg) &&
                    performance_agreement(references[alg],references[:C];atol=c["audit_tolerance"],rtol=c["audit_tolerance"])
                valid[alg] &= same[alg]
                validation[string(alg)]["matches_C_execution"]=same[alg]
            end
            # At eta=0 check the independently known stored-primitive one-step
            # ratios as well; this does not certify any positive-eta case.
            if eta==0 && reference_exists && references[:C].complete
                analytic=[(input.model.h[i,a]-input.model.h[i,a+1])/
                          (input.model.c[i,a+1]-input.model.c[i,a]) for i in 1:n,a in 1:A]
                controlok=all(isapprox(references[:C].mpi[i,a],analytic[i,a];atol=c["audit_tolerance"],
                    rtol=c["audit_tolerance"]) for i in 1:n,a in 1:A)
                case["action_independent_analytic_check"]=controlok
                if !controlok
                    for alg in PERFORMANCE_ALGORITHMS; valid[alg]=false; end
                end
            end
            case["validation"]=validation
            # Warm the SAME measurement functions with fresh state. No warmup
            # sample is included in the reported timing distribution.
            runnable=Dict{Symbol,Bool}()
            for alg in PERFORMANCE_ALGORITHMS
                runnable[alg]=valid[alg] && haskey(references,alg) && references[alg].complete
                if runnable[alg]
                    try
                        for _ in 1:c["warmups"], phase in PERFORMANCE_PHASES
                            _,warm=pilot_phase(input,alg,kw,options,phase)
                            warm.complete && performance_agreement(warm,references[alg];
                                atol=c["audit_tolerance"],rtol=c["audit_tolerance"]) ||
                                error("Warmup disagrees with the validated execution.")
                        end
                    catch err
                        err isa InterruptException && rethrow()
                        runnable[alg]=false; valid[alg]=false
                        validation[string(alg)]["warmup_error"]=sprint(showerror,err,catch_backtrace())
                    end
                end
            end
            # Failed/unresolved validation executions are not repeated for timing.
            for alg in PERFORMANCE_ALGORITHMS
                if !runnable[alg]
                    known_status=haskey(references,alg) ? string(references[alg].status) : "initialization_error"
                    push!(rows,Dict{String,Any}("case_id"=>id,"algorithm"=>string(alg),"phase"=>"not_timed",
                        "repetition"=>0,"ordinal"=>0,"eligible"=>false,"executed"=>false,"status"=>known_status))
                    valid[alg] || (overall_ok=false)
                end
            end
            rng=Xoshiro(performance_stream_seed(c["master_seed"],n,A,draw,Symbol("timing_$(eta)_$(criterionstr)")))
            ordinal=0; skip_remaining=false
            for repetition in 1:c["repetitions"]
                order=shuffle(rng,collect(PERFORMANCE_ALGORITHMS))
                for alg in order, phase in PERFORMANCE_PHASES
                    runnable[alg] || continue
                    ordinal+=1
                    row=Dict{String,Any}("case_id"=>id,"algorithm"=>string(alg),"phase"=>string(phase),
                        "repetition"=>repetition,"ordinal"=>ordinal,"eligible"=>false,"executed"=>false)
                    if time()-started>c["max_case_seconds"]; skip_remaining=true; end
                    if skip_remaining
                        row["status"]="case_budget_skip"; overall_ok=false
                    elseif !runnable[alg]
                        row["status"]="validation_or_warmup_failure"; overall_ok=false
                    else
                        attempt_start=time()
                        try
                            row["executed"]=true
                            measured,snap=pilot_phase(input,alg,kw,options,phase)
                            merge!(row,measured)
                            agrees=performance_agreement(snap,references[alg];atol=c["audit_tolerance"],rtol=c["audit_tolerance"])
                            row["matches_validated_execution"]=agrees
                            row["eligible"]=snap.complete && agrees && measured["steady_state"] && valid[alg] &&
                                isfinite(measured["seconds"]) && measured["seconds"]>0
                            if !agrees || !snap.complete
                                overall_ok=false; runnable[alg]=false
                            end
                        catch err
                            err isa InterruptException && rethrow()
                            row["status"]="execution_error"; row["error"]=sprint(showerror,err,catch_backtrace())
                            row["failure_wall_seconds_including_setup"]=time()-attempt_start
                            overall_ok=false; runnable[alg]=false
                        end
                    end
                    push!(rows,row)
                    # Checkpoint each observation, including unsuccessful ones.
                    open(joinpath(dir,"raw_samples.toml"),"a") do io
                        TOML.print(io,Dict("sample"=>[row]);sorted=true)
                    end
                end
            end
            missing_eligible=String[]
            for alg in PERFORMANCE_ALGORITHMS, phase in PERFORMANCE_PHASES
                valid[alg] && haskey(references,alg) && references[alg].complete || continue
                any(r->r["case_id"]==id && r["algorithm"]==string(alg) && r["phase"]==string(phase) &&
                       get(r,"eligible",false),rows) || push!(missing_eligible,"$(alg)/$(phase)")
            end
            case["phases_without_eligible_timings"]=missing_eligible
            isempty(missing_eligible) || (overall_ok=false)
            case["status"]=all(values(valid)) && !skip_remaining && isempty(missing_eligible) ?
                "processed" : "processed_with_issues"
            case["wall_seconds_including_validation_and_warmup"]=time()-started
            push!(casereports,case)
            pilot_write_toml(joinpath(dir,"cases.toml"),Dict("case"=>casereports))
            println(id,": ",case["status"],"; completion = ",
                join([string(a)*":"*(haskey(references,a) ? string(references[a].status) : "initialization_error")
                      for a in PERFORMANCE_ALGORITHMS],", "))
        end
    end
    pilot_write_toml(joinpath(dir,"raw_samples.toml"),Dict("sample"=>rows))
    summary,ratios=pilot_summaries(rows)
    pilot_csv(joinpath(dir,"samples.csv"),rows,["case_id","algorithm","phase","repetition","ordinal","status",
        "eligible","seconds","allocated_bytes","gc_seconds","compile_seconds","recompile_seconds",
        "retained_bytes_including_model","matches_validated_execution"])
    pilot_csv(joinpath(dir,"summary.csv"),summary,["case_id","algorithm","phase","attempts","eligible",
        "median_seconds","q25_seconds","q75_seconds","median_allocated_bytes","median_gc_seconds"])
    pilot_csv(joinpath(dir,"paired_ratios.csv"),ratios,["case_id","baseline","phase","repetition",
        "eligible_pair","baseline_over_FP"])
    meta["finished_utc"]=string(now(UTC))
    meta["source_sha256_after"]=pilot_source_hashes(root)
    meta["project_sha256_after"]=file_sha(joinpath(root,"Project.toml"))
    meta["manifest_sha256_after"]=file_sha(joinpath(root,"Manifest.toml"))
    meta["sources_unchanged"]=meta["source_sha256_after"]==meta["source_sha256"] &&
        meta["project_sha256_after"]==meta["project_sha256"] &&
        meta["manifest_sha256_after"]==meta["manifest_sha256"]
    meta["eligible_samples"]=count(r->get(r,"eligible",false),rows)
    meta["attempted_timing_samples"]=count(r->get(r,"executed",false),rows)
    overall_ok &= meta["sources_unchanged"]
    meta["harness_checks_passed"]=overall_ok
    pilot_write_toml(joinpath(dir,"metadata.toml"),meta)
    println("Reports: ",dir)
    println("Eligible timing samples: ",meta["eligible_samples"]," / ",meta["attempted_timing_samples"])
    println("Incomplete/unresolved instances remain in reports and do not contribute speedup ratios.")
    overall_ok || error("Fair-performance pilot found validation, runtime, resource-budget, or source-change issues; inspect reports.")
    println("Fair-performance pilot: PASS")
    return dir
end
