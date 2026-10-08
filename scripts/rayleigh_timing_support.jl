"""Certified finite-cap application TIMING PREFLIGHT, separate from generic panels.
No old kernel or harness is replaced. Only explicit adapters and certification
are added. The existing clock, balanced-round, aggregation and CSV helpers are
reused. Certification/validation are not inside algorithm timers.
"""
module RayleighTiming
using LinearAlgebra, TOML, Dates, SHA, Random, Statistics
using ..MultiGearBandits
using ..RayleighAoII
using ..RayleighBlock
const Q = Rational{BigInt}
const APP_VARIANTS=(:C,:R,:M,:FP_legacy,:FP,:BLOCK)
const APP_PHASES=(:initialization,:index_loop,:end_to_end)
const APP_REVISION="rayleigh-certified-timing-preflight-v1"
include("controlled_timing_support.jl")
include("rayleigh_timing_certificate.jl")

function configuration(path)
    c=TOML.parsefile(path)
    c["revision"]==APP_REVISION && c["criterion"]=="average" || error("Unknown application preflight.")
    c["expected_cases"]==4 && c["expected_aggregates"]==216 && length(c["case"])==4 ||
        error("This launcher accepts only the four-case preflight, not a production grid.")
    Tuple(Symbol.(c["variants"]))==APP_VARIANTS || error("All six variants must be retained.")
    [(Int(s["H"]),length(s["success"])-1) for s in c["case"]]==[(4,3),(16,2),(32,2),(16,3)] ||
        error("The prespecified finite-cap grid changed.")
    ids=[String(s["id"]) for s in c["case"]]
    length(unique(ids))==4 && all(occursin(r"^[A-Za-z0-9_-]+$",s) for s in ids) || error("Invalid case ids.")
    for s in c["case"]
        s["curvature_rule"]=="certified_dyadic" && s["source_states"]==2 && s["wR"]=="3/4" &&
        s["kappa"]=="2" && s["chi"]=="1" || error("Preflight source/menu rule changed.")
        a=length(s["success"])-1
        s["success"]==(a==2 ? ["0","1/4","3/4"] : ["0","1/4","1/2","3/4"]) || error("Success menu changed.")
    end
    c["resource_mode"]=="constructed_sync_gain_not_battery_calibration" &&
        c["require_positive_curvature"] && !c["terminal_update"] || error("Resource/endpoint conventions changed.")
    0<c["output_atol"]<=1e-9 && 0<c["output_rtol"]<=1e-9 || error("Do not loosen output agreement requirements.")
    c["threshold_policy_limit"]==10000 || error("Preflight family limit changed.")
    t=c["timing"]
    for key in ("repetitions","warmups","calibration_repetitions","minimum_rounds","maximum_rounds","max_batch_size","reference_block_columns")
        t[key] isa Integer && !(t[key] isa Bool) && t[key]>0 || error("Invalid timing integer: $key")
    end
    t["repetitions"]==3 && t["warmups"]>=2 && t["calibration_repetitions"]>=2 &&
        2<=t["minimum_rounds"]<=t["maximum_rounds"]<=256 && t["max_batch_size"]<=128 || error("Preflight measurement bounds changed.")
    for key in ("target_batch_seconds","target_accumulated_seconds","batch_memory_mib","max_case_seconds",
                "audit_tolerance","diagnostic_minimum_batch_seconds","cpu_wall_ratio_low","cpu_wall_ratio_high",
                "clock_gap_absolute_seconds","clock_gap_relative")
        t[key] isa Real && !(t[key] isa Bool) && isfinite(t[key]) && t[key]>0 || error("Invalid timing scalar: $key")
    end
    t["target_batch_seconds"]<=t["target_accumulated_seconds"]==0.01 && t["max_case_seconds"]<=600 &&
        t["cpu_wall_ratio_low"]<1<t["cpu_wall_ratio_high"] && t["audit_tolerance"]<=1e-7 || error("Preflight timing/diagnostic limits changed.")
    opts=block_options(c);RayleighBlock.check_options(opts)
    screen=c["screening"]
    expected=Dict("comparison_atol"=>1e-11,"comparison_rtol"=>1e-9,"pivot_atol"=>0.0,"pivot_rtol"=>1e-12,
        "exact_fallback"=>true,"exact_max_states"=>8,"residual_fallback"=>true,"residual_max_states"=>256)
    screen==expected || error("Existing generic safeguard settings must be preserved.")
    c
end
function block_options(c)
    d=c["numerical"]
    Set(keys(d))==Set(string.(fieldnames(BlockOptions))) || error("Block option schema differs.")
    o=BlockOptions(value_atol=Float64(d["value_atol"]),value_rtol=Float64(d["value_rtol"]),
        refresh_every=Int(d["refresh_every"]),exact_fallback=Bool(d["exact_fallback"]),
        exact_max_H=Int(d["exact_max_H"]),exact_max_A=Int(d["exact_max_A"]),
        exact_max_steps=Int(d["exact_max_steps"]),exact_max_input_bits=Int(d["exact_max_input_bits"]),
        seconds_limit=Float64(d["seconds_limit"]))
    all(getfield(o,f)==getfield(BlockOptions(),f) for f in fieldnames(BlockOptions)) ||
        error("Accepted block safeguards must remain unchanged for this first timing preflight.")
    o
end
function generic_options(c)
    pilot_options(Dict("screening"=>c["screening"],"terminal_update"=>false))
end

struct AppContext{I,F}
    model::AoIIModel{Q}
    input::I
    family::F
    generic_options::PerformanceOptions
    block_options::BlockOptions
    block_columns::Int
end
function AppContext(m,c)
    input=topology_performance_input(m)
    AppContext(m,input,OrderedThresholdFamily(collect(2:m.H+1)),generic_options(c),block_options(c),
        Int(c["timing"]["reference_block_columns"]))
end

"""Preinitialized compact block adapter. The accepted numerical selector and
statistic update routines are used UNCHANGED. Output buffers are allocated in
initialization. Just as for C/R/M/FP, the unused LAST numerical update is omitted.
The accepted public numerical_downshift function and its archives are untouched.
"""
mutable struct CompactBlockRun
    workspace::NumericBlockWorkspace
    log::MultiGearBandits.PerformanceLog
    bounds::Matrix{Float64}
    status::Symbol
    message::String
    terminal_update_skipped::Bool
end
function app_prepare(ctx::AppContext,::Val{:BLOCK})
    w=NumericBlockWorkspace(ctx.model;options=ctx.block_options)
    log=MultiGearBandits.PerformanceLog(ctx.input.model)
    CompactBlockRun(w,log,Matrix{Float64}(undef,length(log.state),6),:ready,"",false)
end
function app_prepare(ctx::AppContext,v::Val{V}) where {V}
    V in REPLAY_VARIANTS || error("Unknown application timing method $V")
    replay_prepare(ctx.input,v,(;criterion=:average,family=ctx.family),ctx.generic_options,ctx.block_columns)
end
function app_run!(r::CompactBlockRun)
    r.status==:ready && r.log.k==0 || error("Block timing runs require a fresh workspace, not a reused mutated run.")
    w=r.workspace;l=r.log;K=length(l.state);started=time_ns()
    while l.k<K
        if (time_ns()-started)/1e9>w.options.seconds_limit
            r.status=:time_limit;r.message="Cooperative block-loop time limit";return r.status
        end
        try
            selected=choose_frontier!(w);x=selected.candidate;k=l.k+1
            if k<K
                numerical_shift!(w,x.h,x.a)
            else
                # The compact endpoint agrees with terminal_update=false in all
                # generic methods: record final assignment; no unused update.
                r.terminal_update_skipped=true
            end
            l.state[k]=x.h+1;l.gear[k]=x.a;l.f[k]=x.f.value;l.g[k]=x.g.value;l.ratio[k]=x.index.value
            l.dimension[k]=length(w.lengths);l.evidence[k]=selected.mode;l.mpi[x.h,x.a]=x.index.value
            r.bounds[k,:].=(x.index.lo,x.index.hi,x.f.lo,x.f.hi,x.g.lo,x.g.hi)
            l.k=k;r.status=k==K ? :complete : :running
        catch err
            err isa RayleighBlock.NumericalIssue || rethrow()
            r.status=err.status;r.message=err.message;return r.status
        end
    end
    r.status
end
app_run!(r::MultiGearBandits.PerformanceRun)=run_performance!(r)
function app_snapshot(r::CompactBlockRun)
    l=r.log;k=l.k;counts=copy(r.workspace.counts)
    counts["assignments"]=k;counts["terminal_numerical_updates_skipped"]=Int(r.terminal_update_skipped)
    (algorithm=:BLOCK,status=r.status,complete=r.status==:complete,states=copy(l.states),mpi=copy(l.mpi),
     order=[(l.state[j],l.gear[j]) for j in 1:k],f=l.f[1:k],g=l.g[1:k],ratio=l.ratio[1:k],
     dimension=l.dimension[1:k],evidence=l.evidence[1:k],counts=counts,
     terminal_update_skipped=r.terminal_update_skipped,residual_comparisons=Dict{String,Any}[])
end
app_snapshot(r::MultiGearBandits.PerformanceRun)=performance_snapshot(r)
function app_fill_initialization!(runs,ctx,v)
    for j in eachindex(runs);runs[j]=app_prepare(ctx,v);end
    nothing
end
function app_fill_loops!(runs)
    for r in runs;app_run!(r);end
    nothing
end
function app_fill_complete!(outputs,ctx,v)
    for j in eachindex(outputs)
        r=app_prepare(ctx,v);app_run!(r);outputs[j]=app_snapshot(r)
    end
    nothing
end

function matches_exact(s,ref,c)
    s.complete && s.status==:complete && s.terminal_update_skipped &&
        s.states==collect(2:size(ref.mpi,1)+1) && s.order==ref.order &&
        length(s.ratio)==length(ref.ratio) && size(s.mpi)==size(ref.mpi) || return false
    all(isapprox(x,Float64(y);atol=c["output_atol"],rtol=c["output_rtol"]) for (x,y) in zip(s.ratio,ref.ratio)) || return false
    all(!ismissing(s.mpi[j]) && isapprox(s.mpi[j],Float64(ref.mpi[j]);atol=c["output_atol"],rtol=c["output_rtol"])
        for j in eachindex(ref.mpi))
end
function snapshot_matches(a,b,c)
    performance_agreement(a,b;atol=c["output_atol"],rtol=c["output_rtol"]) || return false
    # Compare full logged metrics against the method's independently validated
    # reference; cross-method metrics can use different normalizations.
    all(length(getproperty(a,k))==length(getproperty(b,k)) &&
        all(isapprox(x,y;atol=c["output_atol"],rtol=c["output_rtol"]) for (x,y) in zip(getproperty(a,k),getproperty(b,k)))
        for k in (:f,:g,:ratio))
end

function validate_methods(ctx,record,c;save=(name,r)->nothing)
    ref=exact_reference_record(record);validated=Dict{Symbol,Any}();checks=Dict{String,Any}[]
    # No exact reference is ever passed to app_prepare or app_run!.
    live=app_prepare(ctx,Val(:BLOCK));app_run!(live);s=app_snapshot(live)
    block_record=pilot_snapshot_record(s);block_record["returned_enclosures"]=[collect(live.bounds[k,:]) for k in 1:live.log.k]
    save("BLOCK",block_record)
    matches_exact(s,ref,c) || error("Compact block output failed exact-path comparison: $(s.status)")
    for k in eachindex(s.order)
        for (j,q) in ((1,ref.ratio[k]),(3,ref.f[k]),(5,ref.g[k]))
            Q(live.bounds[k,j])<=q<=Q(live.bounds[k,j+1]) || error("Compact block enclosure misses exact reference.")
        end
    end
    # Compare mechanical timing adaptation to the accepted public live routine.
    accepted=numerical_downshift(ctx.model;options=ctx.block_options)
    accepted.complete && [(h+1,a) for (h,a) in accepted.order]==s.order && accepted.assignments==s.ratio &&
        accepted.f==s.f && accepted.g==s.g && accepted.modes==s.evidence || error("Compact/live numerical adaptation mismatch.")
    all(accepted.bounds[k]==collect(live.bounds[k,:]) for k in eachindex(s.order)) || error("Adapted enclosure log mismatch.")
    all(accepted.counts[k]==v for (k,v) in live.workspace.counts) || error("Adaptation changed selector/rebuild/fallback work.")
    validated[:BLOCK]=s
    push!(checks,Dict("algorithm"=>"BLOCK","passed"=>true,"exact_path"=>true,"enclosures_verified"=>true,
        "accepted_live_routine_equal"=>true,"counts"=>s.counts))
    for v in REPLAY_VARIANTS
        alg=v==:FP_legacy ? :FP : v;init=v==:FP_legacy ? :legacy : :optimized
        audit=audit_performance_run(ctx.input,alg;criterion=:average,family=ctx.family,options=ctx.generic_options,
            fp_initializer=init,reference_block_columns=ctx.block_columns,tol=c["timing"]["audit_tolerance"])
        s=audit.snapshot;save(string(v),pilot_snapshot_record(s))
        audit.passed && matches_exact(s,ref,c) || error("Generic $v failed independent audit/exact path: $(s.status)")
        validated[v]=s
        err=maximum(abs(Float64(x)-Float64(y))/(1+max(abs(Float64(x)),abs(Float64(y)))) for (x,y) in zip(s.ratio,ref.ratio))
        push!(checks,Dict("algorithm"=>string(v),"passed"=>true,"exact_path"=>true,
            "selection_policy_checks"=>audit.selection_policies_checked,"maximum_policy_audit_error"=>audit.max_scaled_error,
            "maximum_scaled_index_error"=>err,"counts"=>s.counts))
    end
    validated,checks
end

# Uses the SAME clock/diagnostic boundary and count checks as the accepted
# generic harness. Every initialization is postvalidated; no validation audits
# are inside numerical timers. Automatic GC remains enabled, never forced here.
function app_measure_batch(ctx,v,phase,batch,reference,exactref,clock,c)
    started=time_ns();batch>=1 || error("Empty application batch")
    prototype=app_prepare(ctx,v);retained=Base.summarysize(prototype)
    if phase==:initialization
        runs=Vector{typeof(prototype)}(undef,batch)
        t=timing_instrument(clock,app_fill_initialization!,runs,ctx,v)
        app_fill_loops!(runs);outputs=[app_snapshot(r) for r in runs]
    elseif phase==:index_loop
        runs=Vector{typeof(prototype)}(undef,batch);app_fill_initialization!(runs,ctx,v)
        t=timing_instrument(clock,app_fill_loops!,runs)
        outputs=[app_snapshot(r) for r in runs]
    elseif phase==:end_to_end
        protoout=app_snapshot(prototype);outputs=Vector{typeof(protoout)}(undef,batch)
        t=timing_instrument(clock,app_fill_complete!,outputs,ctx,v)
    else
        error("Unknown application phase.")
    end
    measured=t.measured
    matches=all(snapshot_matches(s,reference,c) && matches_exact(s,exactref,c) for s in outputs)
    countcheck=timing_work_count_check(outputs,reference)
    complete=all(s.complete for s in outputs)
    diagnostics=timing_diagnostics(measured.time,t.outer_wall_seconds,t.cpu_seconds,t.continuous_seconds,c["timing"])
    rawgc=Dict{String,Int}()
    for name in (:malloc,:realloc,:poolalloc,:bigalloc,:pause,:full_sweep)
        hasproperty(measured.gcstats,name) && (rawgc[string(name)]=Int(getproperty(measured.gcstats,name)))
    end
    row=Dict{String,Any}("executed"=>true,"phase"=>string(phase),"batch_size"=>batch,
        "status"=>string(outputs[1].status),"complete"=>complete,"window_seconds"=>measured.time,
        "cpu_seconds"=>t.cpu_seconds,"outer_wall_seconds"=>t.outer_wall_seconds,"continuous_seconds"=>t.continuous_seconds,
        "window_allocated_bytes"=>Int(measured.bytes),"window_gc_seconds"=>measured.gctime,
        "compile_seconds"=>measured.compile_time,"recompile_seconds"=>measured.recompile_time,
        "raw_gc_counters"=>rawgc,"retained_bytes_including_model"=>retained,
        "matches_validated_execution"=>matches,"equal_work_counts_in_batch"=>countcheck.passed,
        "counts_per_evaluation"=>outputs[1].counts,"validated_outputs"=>length(outputs),
        "reference_validation_audit_solves"=>countcheck.reference_audit_solves,
        "maximum_timed_validation_audit_solves"=>countcheck.maximum_timed_audit_solves,
        "work_count_mismatches"=>countcheck.differences,
        "steady_state"=>measured.compile_time==0 && measured.recompile_time==0,
        "seconds"=>measured.time/batch,"allocated_bytes"=>measured.bytes/batch,"gc_seconds"=>measured.gctime/batch)
    for (key,value) in pairs(diagnostics);row[string(key)]=value;end
    row["execution_eligible"]=complete && matches && countcheck.passed && row["steady_state"] && measured.time>0 && t.outer_wall_seconds>0
    row["duration_credit_seconds"]=row["execution_eligible"] ? diagnostics.duration_credit_seconds : 0.0
    timing_record_batch_cost!(row,started,(calls=0,seconds=0.0))
    row
end

# Output ratios use BLOCK as denominator, with whole SIX-way blocks excluded
# only from the optional diagnostic sensitivity view. All raw rows are retained.
function block_ratios(rows)
    indexed=Dict{Tuple,Vector{Any}}();flagged=Set{Tuple}()
    for r in rows
        key=(r["case_id"],r["phase"],r["repetition"])
        get(r,"diagnostic_flag",false) && push!(flagged,key)
        r["algorithm"]=="BLOCK" && push!(get!(indexed,key,Any[]),r)
    end
    out=Dict{String,Any}[]
    for r in rows
        r["algorithm"]=="BLOCK" && continue
        key=(r["case_id"],r["phase"],r["repetition"]);found=get(indexed,key,Any[])
        length(found)==1 || continue
        b=only(found);ok=get(r,"eligible",false) && get(b,"eligible",false)
        valid=all(haskey(s,"seconds") && isfinite(s["seconds"]) && s["seconds"]>0 for s in (r,b))
        q=valid ? r["seconds"]/b["seconds"] : NaN
        push!(out,Dict("case_id"=>key[1],"phase"=>key[2],"repetition"=>key[3],"baseline"=>r["algorithm"],
            "baseline_over_BLOCK"=>q,"eligible_pair"=>ok && valid,
            "any_variant_flagged_in_block"=>key in flagged,"eligible_diagnostic_sensitivity"=>ok && valid && !(key in flagged)))
    end
    out
end
function write_summaries(out,rows)
    primary,_=pilot_summaries(rows;include_ratios=false)
    execution,_=pilot_summaries(rows;eligibility_key="execution_eligible",include_ratios=false)
    cols=["case_id","algorithm","phase","attempts","eligible","median_seconds","q25_seconds","q75_seconds","median_allocated_bytes","median_gc_seconds"]
    pilot_csv(joinpath(out,"summary.csv"),primary,cols)
    pilot_csv(joinpath(out,"execution_summary.csv"),execution,cols)
    pilot_csv(joinpath(out,"paired_ratios.csv"),block_ratios(rows),["case_id","baseline","phase","repetition",
        "eligible_pair","baseline_over_BLOCK","any_variant_flagged_in_block","eligible_diagnostic_sensitivity"])
    pilot_csv(joinpath(out,"samples.csv"),rows,["case_id","algorithm","phase","repetition","eligible","execution_eligible",
        "seconds","allocated_bytes","gc_seconds","evaluations","rounds","duration_credit_seconds","diagnostic_flag",
        "batch_total_wall_seconds","batch_outside_timer_seconds","forced_gc_calls","dai_verified"])
    pilot_write_toml(joinpath(out,"raw_samples.toml"),Dict("sample"=>rows))
end

function calibrate(ctx,refs,exactref,clock,c,id,out,progress,deadline)
    plans=Dict{Symbol,Any}();t=c["timing"]
    for phase in APP_PHASES
        times=Float64[];sizes=Int[]
        for v in APP_VARIANTS
            val=Val(v);dt=Float64[]
            for j in 1:(t["warmups"]+t["calibration_repetitions"])
                time_ns()<deadline || error("Case ceiling reached during calibration.")
                row=app_measure_batch(ctx,val,phase,1,refs[v],exactref,clock,c)
                stage=j<=t["warmups"] ? "warmup" : "calibration"
                merge!(row,Dict("case_id"=>id,"algorithm"=>string(v),"stage"=>stage,"iteration"=>j))
                timing_append_record(joinpath(out,"calibration.toml"),"batch",row)
                timing_require_matching_batch(row,v,phase,stage;require_steady=stage=="calibration")
                if stage=="calibration";push!(dt,row["seconds"]);end
                push!(sizes,row["retained_bytes_including_model"])
            end
            push!(times,median(dt))
        end
        plan=timing_batch_plan(times,maximum(sizes),t)
        plan["case_id"]=id;plan["phase"]=string(phase);plans[phase]=plan
        timing_append_record(joinpath(out,"batch_plans.toml"),"plan",plan)
        timing_progress!(progress,"$id $phase calibrated: batch " * string(plan["batch_size"]),force=true)
    end
    plans
end

function run_timing_case(record,c,out,rows,clock,progress,session)
    id=String(record["id"]);started=time_ns();deadline=started+UInt64(round(Int,c["timing"]["max_case_seconds"]*1e9))
    case=Dict{String,Any}("case_id"=>id,"passed"=>false,"dai_verified"=>record["dai_verified"],
        "H"=>record["model"]["H"],"A"=>record["model"]["A"])
    timing_progress!(progress,"$id START",force=true)
    try
        st=time_ns();m=certificate_model(record);ctx=AppContext(m,c);exactref=exact_reference_record(record)
        modeldir=joinpath(out,"inputs");mkpath(modeldir)
        case["model_sha256"]=pilot_archive_model(modeldir,id,ctx.input.model)
        case["common_input_seconds"]=(time_ns()-st)/1e9
        st=time_ns()
        save=(name,s)->pilot_write_toml(joinpath(out,"validation_"*id*"_"*name*".toml"),s)
        refs,checks=validate_methods(ctx,record,c;save=save)
        case["independent_validation_seconds"]=(time_ns()-st)/1e9;case["validation"]=checks
        time_ns()<deadline || error("Case ceiling reached during independent validation.")
        st=time_ns();plans=calibrate(ctx,refs,exactref,clock,c,id,out,progress,deadline)
        case["warmup_calibration_seconds"]=(time_ns()-st)/1e9
        seed=performance_stream_seed(c["master_seed"],m.H,length(m.p)-1,1,Symbol("rayleigh-preflight-$session-$id"))
        case["schedule_seed"]=string(seed);rng=Xoshiro(seed)
        for repetition in 1:c["timing"]["repetitions"],phase in shuffle(rng,collect(APP_PHASES))
            batch=Int(plans[phase]["batch_size"])
            timing_progress!(progress,"$id repetition $repetition/3 $phase START",force=true)
            measure=(v,p,b)->app_measure_batch(ctx,Val(v),p,b,refs[v],exactref,clock,c)
            group=timing_balanced_rounds(measure,APP_VARIANTS,phase,batch,c["timing"],rng;
                stop_requested=()->time_ns()>=deadline,
                on_round=(round,credit)->timing_progress!(progress,"$id $phase round $round; minimum credit $(minimum(values(credit)))"))
            for b in group.rows
                merge!(b,Dict("case_id"=>id,"repetition"=>repetition))
                timing_append_record(joinpath(out,"subbatches.toml"),"batch",b)
            end
            for v in APP_VARIANTS
                r=timing_sample_row(group,v,id,phase,repetition,session,"exact_verified_nondecreasing",c["timing"])
                r["dai_verified"]=true;r["certificate_file"]="certificate_"*id*".toml"
                push!(rows,r)
            end
            write_summaries(out,rows)
            timing_progress!(progress,"$id repetition $repetition/3 $phase DONE; $(group.rounds) rounds; $(group.stop_reason)",force=true)
            group.group_target_reached && all(get(r,"execution_eligible",false) for r in group.rows) ||
                error("Balanced group not duration-qualified: $(group.stop_reason). Preserve all rows; no automatic retry.")
        end
        # Reconstruct the frozen model again to check the used arrays directly.
        comparison=dense_model(m;exact=false)
        isequal(ctx.input.model.P,comparison.P) && isequal(ctx.input.model.h,comparison.h) &&
            isequal(ctx.input.model.c,comparison.c) || error("A timing implementation changed common input arrays.")
        case["common_arrays_unchanged"]=true;case["passed"]=true
    catch err
        case["error"]=sprint(showerror,err,catch_backtrace())
    end
    case["elapsed_seconds"]=(time_ns()-started)/1e9
    pilot_write_toml(joinpath(out,"timing_case_"*id*".toml"),case)
    timing_progress!(progress,"$id "*(case["passed"] ? "PASS" : "FAILED"),force=true)
    case
end
end # module
