# New harness; existing initialization and numerical kernels are untouched.
if !isdefined(@__MODULE__, :replay_prepare)
    include(joinpath(@__DIR__,"performance_replay_support.jl"))
end
if !isdefined(@__MODULE__, :TimingClocks)
    include(joinpath(@__DIR__,"controlled_timing_clocks.jl"))
end
if !isdefined(@__MODULE__, :timing_work_count_check)
    include(joinpath(@__DIR__,"timing_work_counts.jl"))
end
if !isdefined(@__MODULE__, :TimingCampaignProgress)
    include(joinpath(@__DIR__,"timing_campaign_support.jl"))
end

function timing_configuration(path)
    c=TOML.parsefile(path)
    get(c,"protocol","")=="controlled-timing-v1" || error("Unsupported controlled-timing protocol.")
    for key in ("states","gears","draws","eta","criteria","variants")
        haskey(c,key) && c[key] isa AbstractVector && !isempty(c[key]) &&
            length(unique(c[key]))==length(c[key]) || error("Invalid or duplicate filter: $key.")
    end
    all(x->x isa Integer && !(x isa Bool) && x>=2,c["states"]) || error("Invalid states.")
    all(x->x isa Integer && !(x isa Bool) && x>=1,c["gears"]) || error("Invalid gears.")
    all(x->x isa Integer && !(x isa Bool) && x>=1,c["draws"]) || error("Invalid draws.")
    all(x->x isa Real && !(x isa Bool) && isfinite(x) && 0<=x<=1,c["eta"]) || error("Invalid eta.")
    all(x->x in ("discounted","average"),c["criteria"]) || error("Invalid criterion.")
    # Same five-way comparison for all cases; no case-dependent removal.
    Set(c["variants"])==Set(string.(REPLAY_VARIANTS)) || error("Require C, R, M, FP_legacy and FP.")
    for key in ("repetitions","warmups","calibration_repetitions","minimum_rounds",
                "maximum_rounds","max_batch_size","reference_block_columns")
        haskey(c,key) && c[key] isa Integer && !(c[key] isa Bool) && c[key]>=1 || error("Invalid $key.")
    end
    c["warmups"]>=2 && c["calibration_repetitions"]>=2 || error("Require at least two warmups/calibrations.")
    2<=c["minimum_rounds"]<=c["maximum_rounds"] || error("Invalid round limits.")
    for key in ("target_batch_seconds","target_accumulated_seconds","batch_memory_mib",
                "memory_budget_gib","max_case_seconds","audit_tolerance",
                "diagnostic_minimum_batch_seconds","cpu_wall_ratio_low","cpu_wall_ratio_high",
                "clock_gap_absolute_seconds","clock_gap_relative")
        haskey(c,key) && c[key] isa Real && !(c[key] isa Bool) && isfinite(c[key]) && c[key]>0 ||
            error("Invalid $key.")
    end
    c["cpu_wall_ratio_low"]<1<c["cpu_wall_ratio_high"] || error("Invalid CPU/wall diagnostic thresholds.")
    c["target_accumulated_seconds"]>=c["target_batch_seconds"] || error("Accumulated target below batch target.")
    c["require_clean_git"] isa Bool || error("require_clean_git must be Boolean.")
    c["require_clean_git"] || error("Measured sessions require a clean tree. Use the diagnostic test script before committing.")
    return c
end

function timing_validate_session(s)
    s isa AbstractString && occursin(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$",s) ||
        error("Session label must be 1-64 letters/digits/dots/underscores/hyphens, beginning with a letter or digit.")
    return String(s)
end
function timing_conditions(note;input=stdin,output=stdout)
    if note===nothing
        println(output,"Enter the actual power/workload/launch conditions. Finish with a line containing END.")
        lines=String[]
        while !eof(input)
            line=readline(input)
            strip(line)=="END" && break
            push!(lines,line)
        end
        note=join(lines,"\n")
    end
    note isa AbstractString || error("Conditions must be text.")
    s=strip(String(note))
    length(s)>=10 || error("Provide actual run conditions (at least 10 characters).")
    any(x->occursin(x,uppercase(s)),("UNRECORDED","INSERT HERE","TODO")) && error("Replace conditions placeholders.")
    return s
end

# No-op wrapper diagnostics are NOT subtracted from algorithm measurements.
timing_noop!(x::Base.RefValue{Int}) = (x[] += 1; nothing)
function timing_clock_probe(clock; samples=7)
    x=Ref(0)
    for _ in 1:3; timing_instrument(clock,timing_noop!,x); end
    rows=Dict{String,Any}[]
    for j in 1:samples
        m=timing_instrument(clock,timing_noop!,x)
        push!(rows,Dict("sample"=>j,"kernel_seconds"=>m.measured.time,
            "cpu_seconds"=>m.cpu_seconds,"outer_wall_seconds"=>m.outer_wall_seconds,
            "continuous_seconds"=>m.continuous_seconds,"allocated_bytes"=>m.measured.bytes))
    end
    return rows
end

# Each invocation makes new workspaces. Only numerical work, vector assignment,
# and the common batch loop are timed. Validation and all snapshot creation except
# the requested end-to-end output are outside the timer.
function timing_measure_batch(input,v,kw,options,block,phase,batch,reference,tol,clock,c;collect_gc=false)
    # Automatic GC stays enabled. The optional explicit collection is retained
    # only for diagnostics and is never requested by the controlled runner.
    batch_started=time_ns()
    batch>=1 || error("Empty timing batch.")
    prototype=replay_prepare(input,v,kw,options,block)
    retained=Base.summarysize(prototype)
    if phase==:initialization
        runs=Vector{typeof(prototype)}(undef,batch)
        forced_gc=timing_optional_collection(collect_gc)
        t=timing_instrument(clock,replay_fill_initialization!,runs,input,v,kw,options,block)
        replay_fill_loops!(runs)
        outputs=[performance_snapshot(run) for run in runs]
    elseif phase==:index_loop
        runs=Vector{typeof(prototype)}(undef,batch)
        replay_fill_initialization!(runs,input,v,kw,options,block)
        forced_gc=timing_optional_collection(collect_gc)
        t=timing_instrument(clock,replay_fill_loops!,runs)
        outputs=[performance_snapshot(run) for run in runs]
    elseif phase==:end_to_end
        sample=performance_snapshot(prototype)
        outputs=Vector{typeof(sample)}(undef,batch)
        forced_gc=timing_optional_collection(collect_gc)
        t=timing_instrument(clock,replay_fill_complete!,outputs,input,v,kw,options,block)
    else
        error("Unknown measured phase: $phase.")
    end
    measured=t.measured
    matches=all(s->performance_agreement(s,reference;atol=tol,rtol=tol),outputs)
    complete=all(s->s.complete,outputs)
    # Audited references carry validation-only solves that are absent by design
    # from timed outputs. Keep those separate; compare every algorithm counter.
    countcheck=timing_work_count_check(outputs,reference)
    equalcounts=countcheck.passed
    d=timing_diagnostics(measured.time,t.outer_wall_seconds,t.cpu_seconds,t.continuous_seconds,c)
    rawgc=Dict{String,Int}()
    for name in (:malloc,:realloc,:poolalloc,:bigalloc,:pause,:full_sweep)
        hasproperty(measured.gcstats,name) && (rawgc[string(name)]=Int(getproperty(measured.gcstats,name)))
    end
    row=Dict{String,Any}("executed"=>true,"batch_size"=>batch,"phase"=>string(phase),
        "status"=>string(outputs[1].status),"complete"=>complete,
        "window_seconds"=>measured.time,"cpu_seconds"=>t.cpu_seconds,
        "outer_wall_seconds"=>t.outer_wall_seconds,"continuous_seconds"=>t.continuous_seconds,
        "window_allocated_bytes"=>Int(measured.bytes),"window_gc_seconds"=>measured.gctime,
        "compile_seconds"=>measured.compile_time,"recompile_seconds"=>measured.recompile_time,
        "raw_gc_counters"=>rawgc,"retained_bytes_including_model"=>retained,
        "matches_validated_execution"=>matches,"equal_work_counts_in_batch"=>equalcounts,
        "counts_per_evaluation"=>outputs[1].counts,"validated_outputs"=>length(outputs),
        "reference_validation_audit_solves"=>countcheck.reference_audit_solves,
        "maximum_timed_validation_audit_solves"=>countcheck.maximum_timed_audit_solves,
        "work_count_mismatches"=>countcheck.differences,
        "steady_state"=>measured.compile_time==0 && measured.recompile_time==0)
    for (key,value) in pairs(d); row[string(key)]=value; end
    row["execution_eligible"]=matches && complete && equalcounts && row["steady_state"] &&
        measured.time>0 && t.outer_wall_seconds>0
    row["seconds"]=measured.time/batch
    row["allocated_bytes"]=measured.bytes/batch
    row["gc_seconds"]=measured.gctime/batch
    row["duration_credit_seconds"]=row["execution_eligible"] ? d.duration_credit_seconds : 0.0
    timing_record_batch_cost!(row,batch_started,forced_gc)
    return row
end

# The memory cap is an admission heuristic, not a hard RSS limit. A single
# instance larger than the cap is rejected explicitly, not shrunk behind the user.
function timing_batch_plan(variant_times,retained_bytes,c)
    !isempty(variant_times) && all(x->isfinite(x) && x>0,variant_times) || error("Invalid calibration times.")
    retained_bytes>0 || error("Invalid retained workspace size.")
    memorycap=floor(Int,c["batch_memory_mib"]*2.0^20/(2*retained_bytes))-1
    memorycap>=1 || error("The batch storage guard does not admit even one fresh workspace.")
    cap=min(Int(c["max_batch_size"]),memorycap)
    fastest=minimum(variant_times)
    wanted=c["target_batch_seconds"]/fastest
    batch=ceil(Int,clamp(wanted,1.0,Float64(cap)))
    return Dict{String,Any}("batch_size"=>batch,"effective_batch_cap"=>cap,
        "memory_cap_batch_size"=>memorycap,"fastest_calibrated_seconds"=>fastest,
        "estimated_fastest_batch_seconds"=>fastest*batch,
        "batch_duration_capped"=>wanted>cap,
        "maximum_retained_workspace_bytes_including_model"=>retained_bytes,
        "memory_guard_is_not_peak_rss"=>true)
end

function timing_append_record(path,key,row)
    open(path,"a") do io
        TOML.print(io,Dict(key=>[row]);sorted=true)
    end
end

# A round runs one batch of EVERY variant. Stopping decisions are made only
# between complete rounds, never by method, ratio, confidence interval or winner.
# The caller gives a deterministic seeded RNG. Independent samples never reuse
# a mutated workspace: measure() builds a new pool at every invocation.
function timing_balanced_rounds(measure,variants,phase,batch,c,rng;
                                stop_requested=()->false,on_round=(round,credit)->nothing)
    progress=Dict(v=>0.0 for v in variants)
    rows=Dict{String,Any}[]
    reason="maximum_rounds"
    completed_rounds=0
    for round in 1:c["maximum_rounds"]
        if stop_requested()
            reason="case_budget"; break
        end
        roundfailed=false
        for (position,v) in enumerate(shuffle(rng,collect(variants)))
            row=Dict{String,Any}("algorithm"=>string(v),"phase"=>string(phase),
                "round"=>round,"round_position"=>position,"batch_size"=>batch,
                "executed"=>false,"execution_eligible"=>false,"duration_credit_seconds"=>0.0)
            try
                merge!(row,measure(v,phase,batch))
                # Reject unexpected callback identities; useful for tests too.
                row["algorithm"]=string(v); row["phase"]=string(phase)
                row["round"]=round; row["round_position"]=position
                row["batch_size"]==batch || error("A method changed the common batch size.")
                credit=Float64(get(row,"duration_credit_seconds",0.0))
                isfinite(credit) && credit>=0 || error("Invalid duration credit.")
                progress[v]+=get(row,"execution_eligible",false) ? credit : 0.0
                roundfailed |= !get(row,"complete",false) || !get(row,"matches_validated_execution",false) ||
                    !get(row,"equal_work_counts_in_batch",false)
            catch err
                err isa InterruptException && rethrow()
                row["status"]="execution_error"; row["error"]=sprint(showerror,err,catch_backtrace())
                row["execution_eligible"]=false; row["duration_credit_seconds"]=0.0
                roundfailed=true
            end
            push!(rows,row)
        end
        completed_rounds=round
        # Read-only diagnostic copy: callbacks cannot alter duration credits.
        # Progress I/O runs after all five timers, never inside a measured batch.
        on_round(round,copy(progress))
        # Retain the whole failed round, but never time another changed-state run.
        if roundfailed
            reason="execution_failure"; break
        end
        if round>=c["minimum_rounds"] && all(progress[v]>=c["target_accumulated_seconds"] for v in variants)
            reason="target_reached"; break
        end
    end
    return (rows=rows,rounds=completed_rounds,stop_reason=reason,
        progress=progress,group_target_reached=reason=="target_reached")
end

function timing_sample_row(group,v,id,phase,repetition,session,monotonicity,c)
    rs=[r for r in group.rows if r["algorithm"]==string(v)]
    executed=[r for r in rs if get(r,"executed",false) && haskey(r,"window_seconds")]
    nevals=sum((Int(r["batch_size"]) for r in executed);init=0)
    row=Dict{String,Any}("case_id"=>id,"algorithm"=>string(v),"phase"=>string(phase),
        "repetition"=>repetition,"session"=>session,"executed"=>!isempty(executed),
        "batch_count"=>length(rs),"evaluations"=>nevals,"rounds"=>group.rounds,
        "stop_reason"=>group.stop_reason,"duration_target_seconds"=>c["target_accumulated_seconds"],
        "duration_credit_seconds"=>group.progress[v],
        "duration_target_reached"=>group.progress[v]>=c["target_accumulated_seconds"],
        "balanced_group_target_reached"=>group.group_target_reached,
        "mpi_monotonicity"=>monotonicity,"dai_verified"=>false)
    row["execution_eligible"]=!isempty(rs) && length(rs)==group.rounds &&
        all(get(r,"execution_eligible",false) for r in rs)
    # Duration-limited observations remain in execution_summary.csv and raw files;
    # they cannot silently qualify for the duration-qualified primary comparison.
    row["eligible"]=row["execution_eligible"] && group.group_target_reached
    row["diagnostic_flag"]=any(get(r,"diagnostic_flag",false) for r in rs)
    row["flagged_batches"]=count(r->get(r,"diagnostic_flag",false),rs)
    row["cpu_low_batches"]=count(r->get(r,"low_cpu_fraction",false),rs)
    row["cpu_high_batches"]=count(r->get(r,"high_cpu_fraction",false),rs)
    row["clock_gap_batches"]=count(r->get(r,"continuous_clock_gap",false),rs)
    if nevals>0
        wall=sum(r["window_seconds"] for r in executed)
        cpu=sum(r["cpu_seconds"] for r in executed)
        row["accumulated_wall_seconds"]=wall
        row["accumulated_cpu_seconds"]=cpu
        row["accumulated_continuous_seconds"]=sum(r["continuous_seconds"] for r in executed)
        row["accumulated_outer_wall_seconds"]=sum(r["outer_wall_seconds"] for r in executed)
        row["seconds"]=wall/nevals
        row["cpu_seconds_per_evaluation"]=cpu/nevals
        row["allocated_bytes"]=sum(r["window_allocated_bytes"] for r in executed)/nevals
        row["gc_seconds"]=sum(r["window_gc_seconds"] for r in executed)/nevals
        row["compile_seconds"]=sum(r["compile_seconds"] for r in executed)
        row["recompile_seconds"]=sum(r["recompile_seconds"] for r in executed)
        row["minimum_batch_wall_seconds"]=minimum(r["window_seconds"] for r in executed)
        row["maximum_batch_wall_seconds"]=maximum(r["window_seconds"] for r in executed)
        row["short_batches"]=count(r->r["window_seconds"]<c["target_batch_seconds"],executed)
    end
    timing_record_sample_cost!(row,executed)
    return row
end

function timing_reporting(rows)
    primary,ratios=pilot_summaries(rows)
    # Same eligibility substitution as the previous copy/merge path, without
    # copying every sample or calculating a second set of discarded ratios.
    execution_summary,_=pilot_summaries(rows;
        eligibility_key="execution_eligible",include_ratios=false)
    # Index ALL rows, not only eligible rows or members of a particular pair.
    # A flagged third variant (including an ineligible one) excludes the whole
    # case/phase/repetition block from the secondary sensitivity view only.
    flagged_blocks=Set{Tuple}()
    for r in rows
        if get(r,"diagnostic_flag",false)
            push!(flagged_blocks,(r["case_id"],r["phase"],r["repetition"]))
        end
    end
    sensitivity=Dict{String,Any}[]
    for r in ratios
        flagged=(r["case_id"],r["phase"],r["repetition"]) in flagged_blocks
        s=copy(r);s["any_variant_flagged_in_block"]=flagged
        s["eligible_diagnostic_sensitivity"]=get(r,"eligible_pair",false) && !flagged
        push!(sensitivity,s)
    end
    return (summary=primary,ratios=ratios,execution_summary=execution_summary,sensitivity=sensitivity)
end

function timing_save_summaries(dir,rows,reports)
    s=timing_reporting(rows)
    cols=["case_id","algorithm","phase","attempts","eligible","median_seconds",
        "q25_seconds","q75_seconds","median_allocated_bytes","median_gc_seconds"]
    pilot_csv(joinpath(dir,"summary.csv"),s.summary,cols)
    pilot_csv(joinpath(dir,"execution_summary.csv"),s.execution_summary,cols)
    pilot_csv(joinpath(dir,"paired_ratios.csv"),s.ratios,
        ["case_id","baseline","phase","repetition","eligible_pair","baseline_over_FP"])
    pilot_csv(joinpath(dir,"paired_ratios_diagnostic.csv"),s.sensitivity,
        ["case_id","baseline","phase","repetition","eligible_pair","baseline_over_FP",
         "any_variant_flagged_in_block","eligible_diagnostic_sensitivity"])
    pilot_csv(joinpath(dir,"samples.csv"),rows,["session","case_id","algorithm","phase","repetition",
        "stop_reason","execution_eligible","eligible","evaluations","batch_count",
        "accumulated_wall_seconds","accumulated_cpu_seconds","seconds","cpu_seconds_per_evaluation",
        "allocated_bytes","gc_seconds","duration_credit_seconds","duration_target_reached",
        "balanced_group_target_reached","minimum_batch_wall_seconds","maximum_batch_wall_seconds",
        "short_batches","flagged_batches","diagnostic_flag","mpi_monotonicity","dai_verified"])
    pilot_csv(joinpath(dir,"monotonicity.csv"),reports,["case_id","status","mpi_monotonicity","dai_verified"])
    pilot_csv(joinpath(dir,"overhead.csv"),rows,["session","case_id","algorithm","phase","repetition",
        "batch_cost_accounting_complete","batch_count","evaluations","accumulated_wall_seconds",
        "batch_total_wall_seconds","batch_outside_timer_seconds","forced_gc_calls","forced_gc_seconds","gc_regime"])
    pilot_csv(joinpath(dir,"campaign_cases.csv"),reports,["case_id","status",
        "wall_seconds_including_validation","common_input_load_and_validation_seconds",
        "independent_validation_seconds","calibration_seconds","balanced_groups_wall_seconds",
        "calibration_forced_gc_calls","mpi_monotonicity","dai_verified"])
end

function timing_source_hashes(root)
    d=replay_source_hashes(root)
    for rel in (".gitignore",".gitattributes","README.md","CITATION.cff")
        p=joinpath(root,rel)
        islink(p) && error("Linked provenance file is not supported: $p")
        isfile(p) && (d[rel]=file_sha(p))
    end
    return d
end

function timing_metadata(root,archive,configpath,c,original,note,session,clock)
    m=replay_metadata(root,archive,configpath,c,original,note)
    m["kind"]="bounded-batch timing-harness validation and saved-input engineering controls; not production evidence"
    m["implementation_revision"]="controlled-timing-v2-natural-gc-progress; numerical kernels unchanged"
    m["protocol"]="controlled-timing-v2"
    m["reporting_revision"]="indexed-reporting-v1"
    m["configuration_schema"]=c["protocol"]
    m["gc_policy"]="automatic GC enabled; no forced collection per controlled batch; timed GC included"
    m["forced_gc_per_controlled_batch"]=false
    m["progress_scope"]="flushed phase messages and between-round heartbeats; not a hard timeout"
    m["batch_cost_scope"]="preparation, timed expression and postvalidation inside timing_measure_batch; excludes outer reporting"
    m["not_poolable_with"]="v1 forced-full-GC timing sessions: same numerical algorithms, different memory-reclamation regime"
    m["work_count_contract"]="exact algorithm-count agreement; reference audit_solves separate; timed audit_solves must be zero"
    m["session"]=session
    m["source_sha256"]=timing_source_hashes(root)
    m["operator_conditions"]=note
    m["launch_arguments"]=copy(ARGS)
    m["julia_executable_directory"]=Sys.BINDIR
    m["pid"]=getpid()
    m["parent_pid"]=Int(ccall(:getppid,Cint,()))
    m["clock"]=timing_clock_metadata(clock)
    m["clock_probe"]=timing_clock_probe(clock)
    m["system_before"]=timing_system_snapshot()
    m["environment_allowlist"]=Dict(k=>get(ENV,k,"unset") for k in
        ("JULIA_NUM_THREADS","JULIA_NUM_GC_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"))
    m["gc_threads"]=isdefined(Threads,:ngcthreads) ? Threads.ngcthreads() : -1
    m["total_system_memory_bytes"]=Sys.total_memory()
    m["free_memory_bytes_at_start"]=Sys.free_memory()
    m["algorithm_order"]="session-seeded case order, phase order and complete round order; all variants participate in every round"
    m["sample_unit"]="one case/phase/repetition aggregate of bounded batches; subbatches are not independent observations"
    m["duration_rule"]="at least minimum_rounds, then stop between complete rounds when EVERY variant accumulates target min(kernel wall, process CPU) seconds, or a saved cap is reached"
    m["primary_estimator"]="sum of timed batch wall seconds divided by sum of evaluations, then within-case medians across repetitions"
    m["exclusion_rule"]="no trimming on elapsed times or diagnostic flags; duration-qualified and execution-only views both retained"
    m["diagnostic_sensitivity"]="secondary only: drop complete case/phase/repetition blocks when any variant is flagged"
    m["cpu_timing_is_not_core_affinity"]=true
    m["peak_process_memory_measured"]=false
    m["power_snapshot_scope"]="before/after snapshots, not continuous monitoring or proof of all-run sleep prevention"
    return m
end

function timing_case_validation(input,saved,kw,options,block,tol)
    eq=audit_fp_initialization(input;kw...,block_columns=block,tol=tol)
    eq.passed || error("Old/new FP initialization mismatch.")
    refs=Dict{Symbol,Any}(); validation=Dict{String,Any}()
    for variant in REPLAY_VARIANTS
        alg=variant==:FP_legacy ? :FP : variant
        init=variant==:FP_legacy ? :legacy : :optimized
        a=audit_performance_run(input,alg;kw...,options=options,tol=tol,
            fp_initializer=init,reference_block_columns=block)
        refs[variant]=a.snapshot
        matchold=replay_matches_archive(a.snapshot,saved;tol=tol)
        screen=performance_mpi_screen(a.snapshot;atol=tol/100,rtol=tol)
        validation[string(variant)]=Dict{String,Any}("audit_passed"=>a.passed,
            "complete"=>a.snapshot.complete,"matches_archived_C_execution"=>matchold,
            "selection_policies_checked"=>a.selection_policies_checked,
            "max_scaled_error"=>a.max_scaled_error,"result"=>pilot_snapshot_record(a.snapshot),
            "mpi_monotonicity"=>string(screen.status),"decreasing_positions"=>screen.decreasing_positions,
            "dai_verified"=>false)
    end
    good=true
    for variant in REPLAY_VARIANTS
        d=validation[string(variant)]
        d["matches_current_C_execution"]=performance_agreement(refs[variant],refs[:C];atol=tol,rtol=tol)
        good &= d["audit_passed"] && d["complete"] && d["matches_archived_C_execution"] && d["matches_current_C_execution"]
    end
    screen=performance_mpi_screen(refs[:C];atol=tol/100,rtol=tol)
    return (passed=good,references=refs,validation=validation,
        equivalence=Dict{String,Any}(string(k)=>v for (k,v) in pairs(eq)),screen=screen)
end

function timing_calibrate(input,kw,options,block,references,tol,clock,c,rng;
                          on_batch=(stage,pass,v,p)->nothing,stop_requested=()->false)
    rows=Dict{String,Any}[]
    # Warm every actual measurement path before the independent calibration set.
    jobs=[(v,p) for v in REPLAY_VARIANTS for p in PERFORMANCE_PHASES]
    for pass in 1:c["warmups"], (v,p) in shuffle(rng,jobs)
        stop_requested() && error("Case budget reached before a warmup batch.")
        on_batch("single_warmup",pass,v,p)
        row=timing_measure_batch(input,Val(v),kw,options,block,p,1,references[v],tol,clock,c;collect_gc=false)
        timing_require_matching_batch(row,v,p,"single_warmup")
        row["algorithm"]=string(v);row["stage"]="single_warmup";row["pass"]=pass
        push!(rows,row)
    end
    for pass in 1:c["calibration_repetitions"], (v,p) in shuffle(rng,jobs)
        stop_requested() && error("Case budget reached before a calibration batch.")
        on_batch("single_calibration",pass,v,p)
        row=timing_measure_batch(input,Val(v),kw,options,block,p,1,references[v],tol,clock,c;collect_gc=false)
        timing_require_matching_batch(row,v,p,"single_calibration";require_steady=true)
        row["algorithm"]=string(v);row["stage"]="single_calibration";row["pass"]=pass
        push!(rows,row)
    end
    plans=Dict{Symbol,Dict{String,Any}}()
    maxbytes=maximum(r["retained_bytes_including_model"] for r in rows)
    for p in PERFORMANCE_PHASES
        times=Float64[]
        for v in REPLAY_VARIANTS
            rs=[r for r in rows if r["algorithm"]==string(v) && r["phase"]==string(p) && r["stage"]=="single_calibration"]
            # CPU bounds keep a descheduled calibration from selecting batch=1.
            ts=[r["duration_credit_seconds"] for r in rs if r["duration_credit_seconds"]>0]
            isempty(ts) && error("No positive CPU/wall calibration for $v/$p.")
            push!(times,median(ts))
        end
        plans[p]=timing_batch_plan(times,maxbytes,c)
        for pass in 1:c["warmups"], v in shuffle(rng,collect(REPLAY_VARIANTS))
            stop_requested() && error("Case budget reached before a configured-batch warmup.")
            on_batch("configured_batch_warmup",pass,v,p)
            row=timing_measure_batch(input,Val(v),kw,options,block,p,plans[p]["batch_size"],references[v],tol,clock,c;collect_gc=false)
            timing_require_matching_batch(row,v,p,"configured_batch_warmup";require_steady=true)
            row["algorithm"]=string(v);row["stage"]="configured_batch_warmup";row["pass"]=pass
            push!(rows,row)
        end
    end
    return (plans=plans,rows=rows)
end

function run_controlled_timing(root,archive,configpath;conditions=nothing,session="")
    campaign_started=time_ns()
    realpath(root)==realpath(pwd()) || error("Run from the repository root.")
    c=timing_configuration(configpath);session=timing_validate_session(session)
    archive=realpath(archive)
    original,selected=replay_cases(archive,c)
    oldconfig=original["configuration"]
    options=pilot_options(oldconfig);tol=c["audit_tolerance"];block=c["reference_block_columns"]
    clock=TimingClocks()
    note=timing_conditions(conditions)
    meta=timing_metadata(root,archive,configpath,c,original,note,session,clock)
    inputhashes=Dict{String,String}()
    for saved in selected
        path=replay_model_path(archive,saved)
        isfile(path) && !islink(path) || error("Saved input missing or linked: $path")
        file_sha(path)==saved["model_sha256"] || error("Saved input hash mismatch: $path")
        inputhashes[relpath(path,archive)]=saved["model_sha256"]
    end
    localroot=joinpath(root,"results","local");mkpath(localroot)
    dir=mktempdir(localroot;prefix="controlled_timing_$(session)_",cleanup=false)
    for (rel,hash) in meta["source_sha256"]
        dest=joinpath(dir,"source_snapshot",rel);mkpath(dirname(dest));cp(joinpath(root,rel),dest)
        file_sha(dest)==hash || error("Source changed during snapshot.")
    end
    cp(configpath,joinpath(dir,"configuration.toml"))
    file_sha(joinpath(dir,"configuration.toml"))==meta["config_sha256"] || error("Configuration changed during snapshot.")
    for (rel,hash) in inputhashes
        dest=joinpath(dir,rel);mkpath(dirname(dest));cp(joinpath(archive,rel),dest)
        file_sha(dest)==hash || error("Input changed during snapshot.")
    end
    meta["archived_input_sha256"]=inputhashes
    orderseed=performance_stream_seed(oldconfig["master_seed"],0,0,0,Symbol("controlled-case-order-$session"))
    ordered=shuffle(Xoshiro(orderseed),selected)
    meta["case_order"]=String[s["case_id"] for s in ordered]
    meta["case_order_seed"]=string(orderseed)
    meta["harness_status"]="started"
    pilot_write_toml(joinpath(dir,"metadata.toml"),meta)
    rows=Dict{String,Any}[];reports=Dict{String,Any}[]
    progress=TimingCampaignProgress(joinpath(dir,"progress.log");start_ns=campaign_started)
    allok=true;interrupted=false;batch_count=0;executed_batch_count=0
    println("Saved-input controlled timing; engineering evidence, not production results.")
    println("Output: ",dir)
    println("Session: ",session,"; cases: ",length(ordered),"; variants: C, R, M, FP_legacy, FP")
    println("Primary observations aggregate fresh-state batches; target ",c["target_accumulated_seconds"]," CPU/wall-bounded seconds per variant and phase.")
    println("Diagnostic flags are retained in the primary comparisons.")
    println("GC policy: automatic; no forced collection before controlled batches.")
    flush(stdout)
    for (case_number,saved) in enumerate(ordered)
        id=saved["case_id"];started=time_ns();caseok=true
        budget=()->Float64(time_ns()-started)/1e9>c["max_case_seconds"]
        timing_progress!(progress,"Case $(case_number)/$(length(ordered)) START $(id)";force=true)
        case=Dict{String,Any}("case_id"=>id,"N"=>saved["N"],"A"=>saved["A"],
            "draw"=>saved["draw"],"eta"=>saved["eta"],"criterion"=>saved["criterion"],
            "model_sha256"=>saved["model_sha256"],"dai_verified"=>false,"status"=>"started")
        try
            case["system_before"]=timing_system_snapshot()
            pilot_memory_estimate(saved["N"],saved["A"])<=c["memory_budget_gib"]*2.0^30 ||
                error("Conservative case memory-admission guard exceeded; this is not a hard OS limit.")
            path=replay_model_path(archive,saved)
            loadtime=@timed PerformanceInput(replay_load_model(path,saved["model_sha256"]))
            input=loadtime.value;case["common_input_load_and_validation_seconds"]=loadtime.time
            case["saved_array_bits_preserved"]=true
            kw=pilot_kwargs(oldconfig,Symbol(saved["criterion"]))
            timing_progress!(progress,"$(id): independent validation START";force=true)
            validation_started=time_ns()
            validation=timing_case_validation(input,saved,kw,options,block,tol)
            case["independent_validation_seconds"]=Float64(time_ns()-validation_started)/1e9
            timing_progress!(progress,"$(id): independent validation DONE";force=true)
            case["validation"]=validation.validation
            case["initialization_equivalence"]=validation.equivalence
            case["mpi_monotonicity"]=string(validation.screen.status)
            case["decreasing_positions"]=validation.screen.decreasing_positions
            validation.passed || error("Five-way numerical or archived-input validation failed; case retained without speedups.")
            refs=validation.references
            seed=performance_stream_seed(oldconfig["master_seed"],saved["N"],saved["A"],saved["draw"],
                Symbol("controlled-$session-$(saved["eta"])-$(saved["criterion"])"))
            case["schedule_seed"]=string(seed)
            rng=Xoshiro(seed)
            timing_progress!(progress,"$(id): warmup/calibration START";force=true)
            calibration_started=time_ns()
            calibration_progress=(stage,pass,v,p)->begin
                if timing_progress_due(progress)
                    timing_progress!(progress,"$(id): $(stage), pass $(pass), $(v)/$(p)")
                end
                nothing
            end
            cal=timing_calibrate(input,kw,options,block,refs,tol,clock,c,rng;
                on_batch=calibration_progress,stop_requested=budget)
            case["calibration_seconds"]=Float64(time_ns()-calibration_started)/1e9
            case["calibration_forced_gc_calls"]=sum(r["forced_gc_calls"] for r in cal.rows)
            case["calibration_forced_gc_calls"]==0 || error("Unexpected forced collection in calibration.")
            timing_progress!(progress,"$(id): warmup/calibration DONE";force=true)
            case["batch_plans"]=Dict(string(p)=>plan for (p,plan) in cal.plans)
            for r in cal.rows
                r["case_id"]=id;r["session"]=session
                timing_append_record(joinpath(dir,"calibration.toml"),"calibration",r)
            end
            case["balanced_groups_wall_seconds"]=0.0
            for repetition in 1:c["repetitions"]
                for phase in shuffle(rng,collect(PERFORMANCE_PHASES))
                    batch=cal.plans[phase]["batch_size"]
                    timing_progress!(progress,"$(id): repetition $(repetition)/$(c["repetitions"]), $(phase) START";force=true)
                    group_started=time_ns()
                    measure=(v,p,n)->timing_measure_batch(input,Val(v),kw,options,block,p,n,refs[v],tol,clock,c;collect_gc=false)
                    round_progress=(round,credit)->begin
                        if timing_progress_due(progress)
                            smallest=timing_progress_seconds(minimum(values(credit)))
                            timing_progress!(progress,"$(id): repetition $(repetition), $(phase), round $(round), minimum credit $(smallest) / $(c["target_accumulated_seconds"]) s")
                        end
                        nothing
                    end
                    group=timing_balanced_rounds(measure,REPLAY_VARIANTS,phase,batch,c,rng;
                        stop_requested=budget,on_round=round_progress)
                    group_seconds=Float64(time_ns()-group_started)/1e9
                    case["balanced_groups_wall_seconds"]+=group_seconds
                    timing_progress!(progress,"$(id): repetition $(repetition), $(phase) DONE; $(group.rounds) rounds; $(group.stop_reason); group wall $(timing_progress_seconds(group_seconds)) s";force=true)
                    # Disk I/O is outside both the timers AND the complete balanced
                    # observation. Only this bounded observation is kept in memory.
                    for r in group.rows
                        r["case_id"]=id;r["repetition"]=repetition;r["session"]=session
                        batch_count+=1;r["batch_id"]=batch_count
                        executed_batch_count+=get(r,"executed",false) ? 1 : 0
                        timing_append_record(joinpath(dir,"subbatches.toml"),"batch",r)
                    end
                    for v in REPLAY_VARIANTS
                        row=timing_sample_row(group,v,id,phase,repetition,session,case["mpi_monotonicity"],c)
                        push!(rows,row);timing_append_record(joinpath(dir,"raw_samples.toml"),"sample",row)
                        caseok &= row["eligible"] && get(row,"forced_gc_calls",-1)==0
                    end
                    group.stop_reason=="execution_failure" && error("Numerical/runtime error in a saved measured round.")
                end
            end
            case["status"]=caseok ? "processed" : "processed_with_duration_or_eligibility_issues"
        catch err
            caseok=false
            case["status"]=err isa InterruptException ? "interrupted" : "case_error"
            case["error"]=sprint(showerror,err,catch_backtrace())
            println(stderr,"Case ",id,": ",sprint(showerror,err))
            interrupted=err isa InterruptException
        end
        allok &= caseok
        case["wall_seconds_including_validation"]=Float64(time_ns()-started)/1e9
        case["system_after"]=timing_system_snapshot()
        push!(reports,case)
        pilot_write_toml(joinpath(dir,"cases.toml"),Dict("case"=>reports))
        timing_save_summaries(dir,rows,reports)
        timing_progress!(progress,"Case $(case_number)/$(length(ordered)) DONE $(id): $(case["status"]); MP screen = $(get(case,"mpi_monotonicity","unavailable"))";force=true)
        interrupted && break
    end
    meta["finished_utc"]=string(now(UTC));meta["system_after"]=timing_system_snapshot()
    meta["campaign_elapsed_seconds"]=Float64(time_ns()-campaign_started)/1e9
    meta["recorded_batch_total_wall_seconds"]=sum((get(r,"batch_total_wall_seconds",0.0) for r in rows);init=0.0)
    meta["recorded_batch_outside_timer_seconds"]=sum((get(r,"batch_outside_timer_seconds",0.0) for r in rows);init=0.0)
    meta["recorded_timed_wall_seconds"]=sum((get(r,"accumulated_wall_seconds",0.0) for r in rows);init=0.0)
    meta["recorded_forced_gc_calls"]=sum((get(r,"forced_gc_calls",0) for r in rows);init=0)+
        sum((get(r,"calibration_forced_gc_calls",0) for r in reports);init=0)
    meta["forced_gc_count_scope"]="recorded calibration and aggregate rows; does not cover interrupted unsaved work"
    meta["source_sha256_after"]=timing_source_hashes(root)
    meta["sources_unchanged"]=meta["source_sha256_after"]==meta["source_sha256"] &&
        pilot_git(root,"rev-parse","HEAD")==meta["git_commit"] && pilot_git(root,"status","--porcelain")==meta["git_status"] &&
        file_sha(configpath)==meta["config_sha256"]
    meta["original_archive_unchanged"]=file_sha(joinpath(archive,"metadata.toml"))==meta["source_metadata_sha256"] &&
        file_sha(joinpath(archive,"cases.toml"))==meta["source_cases_sha256"] &&
        all(file_sha(joinpath(archive,rel))==hash for (rel,hash) in inputhashes)
    meta["expected_aggregate_samples"]=length(selected)*length(REPLAY_VARIANTS)*length(PERFORMANCE_PHASES)*c["repetitions"]
    meta["recorded_aggregate_samples"]=length(rows)
    meta["execution_eligible_samples"]=count(r->get(r,"execution_eligible",false),rows)
    meta["duration_qualified_samples"]=count(r->get(r,"eligible",false),rows)
    meta["samples_with_diagnostic_flags"]=count(r->get(r,"diagnostic_flag",false),rows)
    meta["recorded_subbatch_attempts"]=batch_count
    meta["executed_subbatches"]=executed_batch_count
    meta["recorded_evaluations"]=sum((get(r,"evaluations",0) for r in rows);init=0)
    meta["interrupted"]=interrupted
    allok &= meta["sources_unchanged"] && meta["original_archive_unchanged"] &&
        length(rows)==meta["expected_aggregate_samples"] && !interrupted && meta["recorded_forced_gc_calls"]==0
    meta["harness_checks_passed"]=allok
    meta["harness_status"]=allok ? "passed" : "completed_with_issues"
    pilot_write_toml(joinpath(dir,"metadata.toml"),meta)
    timing_save_summaries(dir,rows,reports)
    println("Reports: ",dir)
    println("Duration-qualified aggregate samples: ",meta["duration_qualified_samples"]," / ",meta["expected_aggregate_samples"])
    println("Samples with diagnostic flags (retained): ",meta["samples_with_diagnostic_flags"])
    println("Balanced subbatches recorded: ",batch_count)
    println("Accumulated duration is not one uninterrupted window, an independent instance count, or a confidence-interval guarantee.")
    allok || error("Controlled timing found duration/runtime/validation/integrity issues; preserve and inspect the saved reports.")
    println("Recorded forced full collections: ",meta["recorded_forced_gc_calls"])
    println("Recorded batch envelope seconds: ",meta["recorded_batch_total_wall_seconds"])
    println("Recorded timed-expression seconds: ",meta["recorded_timed_wall_seconds"])
    println("Controlled timing replay: PASS")
    flush(stdout)
    return dir
end
