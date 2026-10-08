# Include after MultiGearBandits; reuse the previous reporting utilities.
# No input generators are called by the replay path.
if !isdefined(@__MODULE__,:pilot_summaries)
    include(joinpath(@__DIR__,"performance_support.jl"))
end
const REPLAY_VARIANTS=(:C,:R,:M,:FP_legacy,:FP)
if !isdefined(@__MODULE__,:instance_archive_manifest_check)
    include(joinpath(@__DIR__,"instance_archive_contract.jl"))
end

function replay_configuration(path)
    c=TOML.parsefile(path)
    get(c,"protocol","")=="initialization-replay-v1" || error("Unsupported replay protocol.")
    for key in ("states","gears","draws","criteria","eta")
        haskey(c,key) && !isempty(c[key]) && length(unique(c[key]))==length(c[key]) ||
            error("Replay filter $key must be nonempty without duplicates.")
    end
    all(n->n isa Integer && n>=2,c["states"]) || error("Invalid replay state size.")
    all(a->a isa Integer && a>=1,c["gears"]) || error("Invalid replay gear count.")
    all(d->d isa Integer && d>=1,c["draws"]) || error("Invalid replay draw.")
    all(k->k in ("discounted","average"),c["criteria"]) || error("Invalid replay criterion.")
    all(e->e isa Real && isfinite(e) && 0<=e<=1,c["eta"]) || error("Invalid replay perturbation filter.")
    for key in ("repetitions","warmups","max_batch_size","reference_block_columns")
        c[key] isa Integer && c[key]>=1 || error("Invalid $key.")
    end
    c["warmups"]>=2 || error("Use at least two full measurement warmups.")
    for key in ("target_window_seconds","batch_memory_mib","memory_budget_gib","max_case_seconds","audit_tolerance")
        c[key] isa Real && isfinite(c[key]) && c[key]>0 || error("Invalid $key.")
    end
    c["require_clean_git"] isa Bool || error("require_clean_git must be Boolean.")
    return c
end

# Exact saved-file integrity + bitwise stored-array preservation. A file hash is
# an integrity check against this archive, not a cryptographic authenticity claim.
function replay_load_model(path,expected_hash)
    isfile(path) && !islink(path) || error("Missing or linked saved model: $path")
    file_sha(path)==expected_hash || error("Saved model checksum mismatch: $path")
    a=TOML.parsefile(path); n=Int(a["N"]); A=Int(a["A"])
    n>=1 && A>=1 || error("Invalid saved model dimensions.")
    get(a,"layout","")=="P[N,N,A+1],h[N,A+1],c[N,A+1],column-major; floating stored-array model" ||
        error("Unknown saved model array layout.")
    length(a["P"])==n*n*(A+1) && length(a["h"])==n*(A+1) &&
        length(a["c"])==n*(A+1) && length(a["controllable"])==n || error("Saved array lengths disagree.")
    all(x->x isa Bool,a["controllable"]) || error("Saved controllability flags must be Boolean.")
    P=reshape(Float64.(a["P"]),n,n,A+1); h=reshape(Float64.(a["h"]),n,A+1)
    c=reshape(Float64.(a["c"]),n,A+1); mask=Bool.(a["controllable"])
    model=FiniteMultiGearModel(P,h,c;controllable=mask)
    for (loaded,stored) in ((model.P,P),(model.h,h),(model.c,c))
        isequal(reinterpret(UInt64,vec(loaded)),reinterpret(UInt64,vec(stored))) ||
            error("Model construction changed a saved floating-point array.")
    end
    return model
end

function replay_cases(archive,c)
    metapath=joinpath(archive,"metadata.toml"); casepath=joinpath(archive,"cases.toml")
    isfile(metapath) && isfile(casepath) || error("Select an extracted complete pilot directory, not a ZIP.")
    meta=TOML.parsefile(metapath)
    if get(meta,"protocol","")==INSTANCE_ARCHIVE_PROTOCOL
        instance_archive_manifest_check(archive,meta)
    else
        get(meta,"protocol","")=="fair-performance-pilot-v1" ||
            error("Expected a saved v1 pilot or a finalized multigear input-only archive.")
    end
    original=meta["configuration"]
    meta["terminal_update"]==original["terminal_update"] || error("Conflicting archived endpoint records.")
    allcases=TOML.parsefile(casepath)["case"]
    selected=[s for s in allcases if all(haskey(s,k) for k in ("N","A","draw","eta","criterion","case_id")) &&
        s["N"] in c["states"] && s["A"] in c["gears"] && s["draw"] in c["draws"] &&
        s["eta"] in c["eta"] && s["criterion"] in c["criteria"]]
    ids=[s["case_id"] for s in selected]
    length(unique(ids))==length(ids) || error("Duplicate case IDs in source archive.")
    expected=prod(length(c[k]) for k in ("states","gears","draws","eta","criteria"))
    length(selected)==expected || error("Archive contains $(length(selected)) matching cases; filters request $expected. No new instances are generated.")
    for s in selected
        id=s["case_id"]; criterion=s["criterion"]
        occursin(r"^[A-Za-z0-9_.-]+$",id) && endswith(id,"_"*criterion) || error("Unsafe or invalid archived case ID.")
        haskey(s,"model_sha256") && haskey(s,"validation") && haskey(s["validation"],"C") ||
            error("Missing archived model/reference for $id.")
    end
    return meta,selected
end

function replay_model_path(archive,s)
    suffix="_"*s["criterion"]
    id=chop(s["case_id"];tail=length(suffix))
    path=joinpath(archive,"inputs",id*".toml")
    isdir(joinpath(archive,"inputs")) && !islink(joinpath(archive,"inputs")) || error("Invalid saved inputs directory.")
    return path
end

function replay_matches_archive(snapshot,s;tol=1e-7)
    old=s["validation"]["C"]["result"]
    string(snapshot.status)==old["status"] && snapshot.states==old["states"] || return false
    order=[(Int(pair[1]),Int(pair[2])) for pair in old["order"]]
    snapshot.order==order && length(snapshot.ratio)==length(old["assigned_ratio"]) || return false
    return all(isapprox(x,y;atol=tol,rtol=tol) for (x,y) in zip(snapshot.ratio,old["assigned_ratio"]))
end

function replay_prepare(input,::Val{V},kw,options,block;trace=nothing) where {V}
    V in REPLAY_VARIANTS || error("Invalid replay variant.")
    alg=V==:FP_legacy ? :FP : V
    initializer=V==:FP_legacy ? :legacy : :optimized
    return prepare_performance_run(input,alg;kw...,options=options,fp_initializer=initializer,
        reference_block_columns=block,preparation_trace=trace)
end
function replay_fill_initialization!(runs,input,v,kw,options,block)
    for j in eachindex(runs); runs[j]=replay_prepare(input,v,kw,options,block); end
    return nothing
end
function replay_fill_loops!(runs)
    for run in runs; run_performance!(run); end
    return nothing
end
function replay_fill_complete!(outputs,input,v,kw,options,block)
    for j in eachindex(outputs)
        run=replay_prepare(input,v,kw,options,block)
        run_performance!(run); outputs[j]=performance_snapshot(run)
    end
    return nothing
end

# Every mutating loop receives its OWN freshly initialized workspace. Pool and
# output-vector allocation are outside the timers. For end-to-end, every owned
# compact result IS created inside timing and retained for validation afterward.
# The small common batch-loop/vector-assignment overhead is included for all.
function replay_measure(input,v,kw,options,block,phase,batch,reference,tol;collect_gc=true)
    batch>=1 || error("Empty timing batch.")
    prototype=replay_prepare(input,v,kw,options,block)
    retained=Base.summarysize(prototype)
    if phase==:initialization
        runs=Vector{typeof(prototype)}(undef,batch)
        collect_gc && GC.gc()
        measured=@timed replay_fill_initialization!(runs,input,v,kw,options,block)
        replay_fill_loops!(runs) # not part of initialization timing
        outputs=[performance_snapshot(run) for run in runs]
    elseif phase==:index_loop
        runs=Vector{typeof(prototype)}(undef,batch)
        replay_fill_initialization!(runs,input,v,kw,options,block)
        collect_gc && GC.gc()
        measured=@timed replay_fill_loops!(runs)
        outputs=[performance_snapshot(run) for run in runs]
    elseif phase==:end_to_end
        sample=performance_snapshot(prototype)
        outputs=Vector{typeof(sample)}(undef,batch)
        collect_gc && GC.gc()
        measured=@timed replay_fill_complete!(outputs,input,v,kw,options,block)
    else
        error("Unknown replay phase.")
    end
    matches=all(s->performance_agreement(s,reference;atol=tol,rtol=tol),outputs)
    complete=all(s->s.complete,outputs)
    equalcounts=all(s->s.counts==outputs[1].counts,outputs)
    rawgc=Dict{String,Int}()
    for name in (:malloc,:realloc,:poolalloc,:bigalloc,:pause,:full_sweep)
        hasproperty(measured.gcstats,name) && (rawgc[string(name)]=Int(getproperty(measured.gcstats,name)))
    end
    row=Dict{String,Any}("phase"=>string(phase),"batch_size"=>batch,
        "seconds"=>measured.time/batch,"window_seconds"=>measured.time,
        "allocated_bytes"=>measured.bytes/batch,"window_allocated_bytes"=>Int(measured.bytes),
        "gc_seconds"=>measured.gctime/batch,"window_gc_seconds"=>measured.gctime,
        "compile_seconds"=>measured.compile_time,"recompile_seconds"=>measured.recompile_time,
        "raw_gc_counters"=>rawgc,"retained_bytes_including_model"=>retained,
        "validated_outputs"=>length(outputs),"matches_validated_execution"=>matches,
        "equal_work_counts_in_batch"=>equalcounts,"counts_per_evaluation"=>outputs[1].counts,
        "status"=>string(outputs[1].status),"complete"=>complete,
        "steady_state"=>measured.compile_time==0 && measured.recompile_time==0)
    row["eligible"]=matches && complete && equalcounts && row["steady_state"] && isfinite(measured.time) && measured.time>0
    return row
end

function replay_power_snapshot()
    if Sys.isapple()
        records=Dict{String,String}()
        for (label,args) in (("battery",["pmset","-g","batt"]),("settings",["pmset","-g","custom"]))
            records[label]=try
                strip(read(pipeline(Cmd(args),stderr=devnull),String))
            catch err
                "unavailable: "*sprint(showerror,err)
            end
        end
        return records
    end
    return Dict("battery"=>"not collected on this platform","settings"=>"not collected on this platform")
end

function replay_source_hashes(root)
    out=pilot_source_hashes(root)
    for sub in ("test","docs")
        isdir(joinpath(root,sub)) || continue
        for (dir,_,files) in walkdir(joinpath(root,sub)), file in sort(files)
            any(ext->endswith(file,ext),(".jl",".toml",".md")) || continue
            path=joinpath(dir,file); islink(path) && error("Linked source not supported: $path")
            out[relpath(path,root)]=file_sha(path)
        end
    end
    for name in ("Project.toml","Manifest.toml","julia-version","LICENSE","LICENSE-DATA","LICENSES.md")
        isfile(joinpath(root,name)) && (out[name]=file_sha(joinpath(root,name)))
    end
    return out
end

function replay_metadata(root,archive,config_path,c,original,conditions)
    Threads.nthreads(:default)==1 && Threads.nthreads(:interactive)==0 && BLAS.get_num_threads()==1 ||
        error("Replay timing requires --threads=1,0 and one BLAS thread.")
    isfile(joinpath(root,"julia-version")) && strip(read(joinpath(root,"julia-version"),String))==string(VERSION) ||
        error("Julia does not match julia-version.")
    active=Base.active_project()
    active!==nothing && realpath(active)==realpath(joinpath(root,"Project.toml")) || error("Activate with --project=.")
    status=pilot_git(root,"status","--porcelain")
    commit=pilot_git(root,"rev-parse","HEAD")
    c["require_clean_git"] && (status!="" || commit=="unavailable") && error(
        "Commit the tested source changes before this replay. No timing run started. Git status: $status")
    note=conditions
    if note===nothing
        print("Describe actual power/low-power settings and competing workload for this run: ")
        flush(stdout); note=readline()
    end
    note=strip(note)
    length(note)>=10 && !occursin("UNRECORDED",uppercase(note)) || error("Provide a factual conditions note (at least 10 characters).")
    return Dict{String,Any}("protocol"=>c["protocol"],"kind"=>"archived-input engineering replay; not production evidence",
        "julia_version"=>string(VERSION),"recorded_version"=>strip(read(joinpath(root,"julia-version"),String)),
        "source_archive"=>abspath(archive),"source_metadata_sha256"=>file_sha(joinpath(archive,"metadata.toml")),
        "source_cases_sha256"=>file_sha(joinpath(archive,"cases.toml")),"source_git_commit"=>original["git_commit"],
        "source_archive_was_dirty"=>get(original,"git_status","")!="",
        "git_commit"=>commit,"git_status"=>status,"configuration"=>c,"config_sha256"=>file_sha(config_path),
        "source_sha256"=>replay_source_hashes(root),"operator_conditions"=>note,
        "power_snapshot_before"=>replay_power_snapshot(),"started_utc"=>string(now(UTC)),
        "julia_threads"=>Threads.nthreads(:default),"interactive_threads"=>Threads.nthreads(:interactive),
        "blas_threads"=>BLAS.get_num_threads(),"blas_configuration"=>sprint(show,BLAS.get_config()),
        "os"=>string(Sys.KERNEL),"architecture"=>string(Sys.ARCH),"cpu_name"=>Sys.CPU_NAME,"machine"=>Sys.MACHINE,
        "logical_cpus"=>Sys.CPU_THREADS,"terminal_update"=>original["terminal_update"],
        "preparation_checks"=>"all freshly solved columns; common normwise backward tolerance 1e-10",
        "algorithm_order"=>"seeded randomized variant AND phase order within each repetition",
        "timing_units"=>"seconds/allocated bytes/GC seconds PER EVALUATION; total window fields retained",
        "instrumentation"=>"separate runs; never included in principal timing samples",
        "gc_policy"=>"enabled; full GC before window, outside timer",
        "source_environment_matches"=>get(original,"julia_version","")==string(VERSION) &&
            get(original,"project_sha256","")==file_sha(joinpath(root,"Project.toml")) &&
            get(original,"manifest_sha256","")==file_sha(joinpath(root,"Manifest.toml")))
end

function replay_instrument(input,variant,kw,options,block)
    trace=PreparationTrace()
    run=replay_prepare(input,Val(variant),kw,options,block;trace=trace)
    stages=[Dict{String,Any}(string(k)=> (v isa Symbol ? string(v) : v) for (k,v) in pairs(row)) for row in trace.stages]
    residuals=[Dict{String,Any}(string(k)=> (v isa Symbol ? string(v) : v) for (k,v) in pairs(row)) for row in trace.residuals]
    return (stages=stages,residuals=residuals,counts=performance_counts(run))
end

function run_initialization_replay(root,archive,config_path;conditions=nothing)
    realpath(root)==realpath(pwd()) || error("Run from the repository root.")
    archive=realpath(archive); c=replay_configuration(config_path)
    original,selected=replay_cases(archive,c); oldconfig=original["configuration"]
    options=pilot_options(oldconfig); tol=c["audit_tolerance"]; block=c["reference_block_columns"]
    meta=replay_metadata(root,archive,config_path,c,original,conditions)
    # Validate all source hashes BEFORE creating output or running algorithms.
    inputhashes=Dict{String,String}()
    for s in selected
        path=replay_model_path(archive,s)
        file_sha(path)==s["model_sha256"] || error("Saved model checksum mismatch: $path")
        inputhashes[relpath(path,archive)]=s["model_sha256"]
    end
    localroot=joinpath(root,"results","local"); mkpath(localroot)
    dir=mktempdir(localroot;prefix="initialization_replay_",cleanup=false)
    # Preserve exact source files, not regenerated inputs or a mutable URL.
    for (rel,hash) in meta["source_sha256"]
        path=joinpath(dir,"source_snapshot",rel); mkpath(dirname(path)); cp(joinpath(root,rel),path)
        file_sha(path)==hash || error("Source changed while capturing snapshot.")
    end
    for (rel,hash) in inputhashes
        path=joinpath(dir,rel); mkpath(dirname(path)); cp(joinpath(archive,rel),path)
        file_sha(path)==hash || error("Input changed while capturing snapshot.")
    end
    meta["archived_input_sha256"]=inputhashes
    pilot_write_toml(joinpath(dir,"metadata.toml"),meta)
    rows=Dict{String,Any}[]; reports=Dict{String,Any}[]; stages=Dict{String,Any}[]
    all_ok=true
    println("Archived-input initialization replay; NOT production evidence.")
    println("Output: ",dir)
    println("Cases: ",length(selected),"; variants: C, R, M, FP_legacy, FP")
    println("Matched all-solved-column preparation checks; pivot kernels unchanged.")
    for saved in selected
        id=saved["case_id"]; started=time()
        case=Dict{String,Any}("case_id"=>id,"N"=>saved["N"],"A"=>saved["A"],"draw"=>saved["draw"],
            "eta"=>saved["eta"],"criterion"=>saved["criterion"],"model_sha256"=>saved["model_sha256"],
            "dai_verified"=>false,"status"=>"started")
        references=Dict{Symbol,Any}(); valid=Dict(v=>false for v in REPLAY_VARIANTS)
        validation=Dict{String,Any}(); runnable=Dict(v=>false for v in REPLAY_VARIANTS)
        try
            pilot_memory_estimate(saved["N"],saved["A"])<=c["memory_budget_gib"]*2.0^30 ||
                error("Conservative admission memory guard exceeded; no hard OS limit is claimed.")
            path=replay_model_path(archive,saved)
            tinput=@timed PerformanceInput(replay_load_model(path,saved["model_sha256"]))
            input=tinput.value; case["saved_array_bits_preserved"]=true
            case["common_input_load_and_validation_seconds"]=tinput.time
            criterion=Symbol(saved["criterion"]); kw=pilot_kwargs(oldconfig,criterion)
            eq=audit_fp_initialization(input;kw...,block_columns=block,tol=tol)
            case["initialization_equivalence"]=Dict{String,Any}(string(k)=>v for (k,v) in pairs(eq))
            eq.passed || error("Legacy/optimized initialized objects disagree.")
            for variant in REPLAY_VARIANTS
                alg=variant==:FP_legacy ? :FP : variant
                init=variant==:FP_legacy ? :legacy : :optimized
                try
                    a=audit_performance_run(input,alg;kw...,options=options,tol=tol,
                        fp_initializer=init,reference_block_columns=block)
                    references[variant]=a.snapshot
                    matchold=replay_matches_archive(a.snapshot,saved;tol=tol)
                    valid[variant]=a.passed && a.snapshot.complete && matchold
                    screen=performance_mpi_screen(a.snapshot;atol=tol/100,rtol=tol)
                    validation[string(variant)]=Dict{String,Any}("audit_passed"=>a.passed,
                        "selection_policies_checked"=>a.selection_policies_checked,"max_scaled_error"=>a.max_scaled_error,
                        "matches_archived_C_execution"=>matchold,"result"=>pilot_snapshot_record(a.snapshot),
                        "mpi_monotonicity"=>string(screen.status),"decreasing_positions"=>screen.decreasing_positions,
                        "minimum_assignment_gap"=>screen.minimum_gap,"dai_verified"=>false)
                catch err
                    err isa InterruptException && rethrow()
                    validation[string(variant)]=Dict("error"=>sprint(showerror,err,catch_backtrace()))
                end
            end
            for variant in REPLAY_VARIANTS
                agrees=valid[:C] && haskey(references,variant) &&
                    performance_agreement(references[variant],references[:C];atol=tol,rtol=tol)
                valid[variant] &= agrees
                validation[string(variant)]["matches_current_C_execution"]=agrees
            end
            case["validation"]=validation
            if haskey(references,:C)
                screen=performance_mpi_screen(references[:C];atol=tol/100,rtol=tol)
                case["mpi_monotonicity"]=string(screen.status)
                case["decreasing_positions"]=screen.decreasing_positions
            end
            all(values(valid)) || (all_ok=false)
            # Warm the actual batch measurement machinery, then calibrate one
            # common batch size per case/phase across ALL valid variants.
            calibration=Dict{Tuple{Symbol,Symbol},Float64}(); retained=Int[]
            for variant in REPLAY_VARIANTS
                valid[variant] || continue
                try
                    for phase in PERFORMANCE_PHASES
                        for _ in 1:c["warmups"]
                            r=replay_measure(input,Val(variant),kw,options,block,phase,1,references[variant],tol)
                            r["matches_validated_execution"] && r["complete"] || error("Warmup disagrees with validation.")
                            calibration[(variant,phase)]=r["seconds"]
                            push!(retained,r["retained_bytes_including_model"])
                        end
                    end
                    runnable[variant]=true
                    # Instrumentation-specific compilation is warmed separately.
                    replay_instrument(input,variant,kw,options,block)
                    replay_instrument(input,variant,kw,options,block)
                    instrument=replay_instrument(input,variant,kw,options,block)
                    validation[string(variant)]["preparation_residuals"]=instrument.residuals
                    validation[string(variant)]["preparation_counts"]=instrument.counts
                    for stage in instrument.stages
                        row=copy(stage); row["case_id"]=id; row["algorithm"]=string(variant); push!(stages,row)
                    end
                catch err
                    err isa InterruptException && rethrow()
                    runnable[variant]=false; all_ok=false
                    validation[string(variant)]["warmup_or_instrumentation_error"]=sprint(showerror,err,catch_backtrace())
                end
            end
            any(values(runnable)) || error("No validated warmed variant is runnable.")
            # Retained workspace is only a conservative batch-admission estimate,
            # NOT peak RSS. Account for an extra prototype and setup temporaries.
            maxbytes=maximum(retained)
            memorycap=floor(Int,c["batch_memory_mib"]*2.0^20/max(1,2*maxbytes))-1
            memorycap>=1 || error("Batch storage guard cannot admit even one fresh workspace.")
            cap=min(c["max_batch_size"],memorycap)
            batches=Dict{Symbol,Int}()
            for phase in PERFORMANCE_PHASES
                fastest=minimum(calibration[(v,phase)] for v in REPLAY_VARIANTS if runnable[v])
                desired=fastest>0 ? c["target_window_seconds"]/fastest : Float64(cap)
                batches[phase]=Int(ceil(clamp(desired,1.0,Float64(cap))))
                # The batch size can change specialization; warm that exact size.
                for variant in REPLAY_VARIANTS
                    runnable[variant] || continue
                    for _ in 1:c["warmups"]
                        r=replay_measure(input,Val(variant),kw,options,block,phase,batches[phase],references[variant],tol)
                        r["matches_validated_execution"] && r["complete"] || error("Full batch warmup disagrees.")
                    end
                end
            end
            case["batch_sizes"]=Dict(string(k)=>v for (k,v) in batches)
            case["batch_memory_guard_bytes"]=c["batch_memory_mib"]*2.0^20
            case["preparation_instrumentation_is_separate"]=true
            rng=Xoshiro(performance_stream_seed(oldconfig["master_seed"],saved["N"],saved["A"],saved["draw"],
                Symbol("replay_$(saved["eta"])_$(saved["criterion"])")))
            ordinal=0
            for repetition in 1:c["repetitions"]
                jobs=shuffle(rng,[(v,p) for v in REPLAY_VARIANTS for p in PERFORMANCE_PHASES])
                for (variant,phase) in jobs
                    ordinal+=1
                    row=Dict{String,Any}("case_id"=>id,"algorithm"=>string(variant),"phase"=>string(phase),
                        "repetition"=>repetition,"ordinal"=>ordinal,"executed"=>false,"eligible"=>false,
                        "mpi_monotonicity"=>get(case,"mpi_monotonicity","unavailable"),"dai_verified"=>false)
                    if !runnable[variant]
                        row["status"]="validation_or_warmup_failure"
                    elseif time()-started>c["max_case_seconds"]
                        row["status"]="case_budget_skip"; all_ok=false
                    else
                        try
                            row["executed"]=true
                            measured=replay_measure(input,Val(variant),kw,options,block,phase,batches[phase],references[variant],tol)
                            merge!(row,measured)
                            row["window_target_reached"]=row["window_seconds"]>=c["target_window_seconds"]
                            if !row["matches_validated_execution"] || !row["complete"] || !row["equal_work_counts_in_batch"]
                                runnable[variant]=false; all_ok=false
                            end
                        catch err
                            err isa InterruptException && rethrow()
                            row["status"]="execution_error"; row["error"]=sprint(showerror,err,catch_backtrace())
                            runnable[variant]=false; all_ok=false
                        end
                    end
                    push!(rows,row)
                    open(joinpath(dir,"raw_samples.toml"),"a") do io; TOML.print(io,Dict("sample"=>[row]);sorted=true); end
                end
            end
            missing=String[]
            for v in REPLAY_VARIANTS, phase in PERFORMANCE_PHASES
                any(r->r["case_id"]==id && r["algorithm"]==string(v) && r["phase"]==string(phase) &&
                    get(r,"eligible",false),rows) || push!(missing,"$v/$phase")
            end
            case["phases_without_eligible_timings"]=missing
            isempty(missing) || (all_ok=false)
            case["status"]=all(values(valid)) && isempty(missing) ? "processed" : "processed_with_issues"
        catch err
            err isa InterruptException && rethrow()
            all_ok=false; case["status"]="case_error"; case["error"]=sprint(showerror,err,catch_backtrace())
            case["validation"]=validation
        end
        case["wall_seconds_including_validation"]=time()-started
        push!(reports,case)
        pilot_write_toml(joinpath(dir,"cases.toml"),Dict("case"=>reports))
        pilot_write_toml(joinpath(dir,"preparation_stages.toml"),Dict("stage"=>stages))
        println(id,": ",case["status"],"; MP screen = ",get(case,"mpi_monotonicity","unavailable"))
    end
    pilot_write_toml(joinpath(dir,"raw_samples.toml"),Dict("sample"=>rows))
    summary,ratios=pilot_summaries(rows)
    pilot_csv(joinpath(dir,"summary.csv"),summary,["case_id","algorithm","phase","attempts","eligible",
        "median_seconds","q25_seconds","q75_seconds","median_allocated_bytes","median_gc_seconds"])
    pilot_csv(joinpath(dir,"paired_ratios.csv"),ratios,["case_id","baseline","phase","repetition","eligible_pair","baseline_over_FP"])
    pilot_csv(joinpath(dir,"samples.csv"),rows,["case_id","algorithm","phase","repetition","ordinal","batch_size",
        "status","eligible","seconds","window_seconds","allocated_bytes","window_allocated_bytes","gc_seconds",
        "compile_seconds","recompile_seconds","window_target_reached","mpi_monotonicity","dai_verified"])
    pilot_csv(joinpath(dir,"preparation_stages.csv"),stages,["case_id","algorithm","stage","seconds","allocated_bytes",
        "gc_seconds","compile_seconds","recompile_seconds"])
    pilot_csv(joinpath(dir,"monotonicity.csv"),reports,["case_id","status","mpi_monotonicity","dai_verified"])
    meta["finished_utc"]=string(now(UTC)); meta["power_snapshot_after"]=replay_power_snapshot()
    meta["source_sha256_after"]=replay_source_hashes(root)
    meta["sources_unchanged"]=meta["source_sha256_after"]==meta["source_sha256"] &&
        pilot_git(root,"rev-parse","HEAD")==meta["git_commit"] && pilot_git(root,"status","--porcelain")==meta["git_status"]
    meta["original_archive_unchanged"]=file_sha(joinpath(archive,"metadata.toml"))==meta["source_metadata_sha256"] &&
        file_sha(joinpath(archive,"cases.toml"))==meta["source_cases_sha256"] &&
        all(file_sha(joinpath(archive,rel))==hash for (rel,hash) in inputhashes)
    meta["eligible_windows"]=count(r->get(r,"eligible",false),rows)
    meta["executed_windows"]=count(r->get(r,"executed",false),rows)
    meta["eligible_evaluations"]=sum((get(r,"batch_size",0) for r in rows if get(r,"eligible",false));init=0)
    meta["eligible_windows_below_target"]=count(r->get(r,"eligible",false) && !get(r,"window_target_reached",false),rows)
    all_ok &= meta["sources_unchanged"] && meta["original_archive_unchanged"]
    meta["harness_checks_passed"]=all_ok
    pilot_write_toml(joinpath(dir,"metadata.toml"),meta)
    println("Reports: ",dir)
    println("Eligible timing windows: ",meta["eligible_windows"]," / ",meta["executed_windows"])
    println("Eligible evaluations within those windows: ",meta["eligible_evaluations"])
    println("Eligible windows below target duration (batch cap retained): ",meta["eligible_windows_below_target"])
    println("MP screens are not DAI certificates; nonmonotone completed cases remain in timing reports.")
    all_ok || error("Initialization replay found validation/runtime/integrity issues; inspect saved reports.")
    println("Initialization replay: PASS")
    return dir
end
