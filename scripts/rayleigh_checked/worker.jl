include("common.jl")
function rc_worker(configpath,jobpath,out)
    VERSION==VersionNumber(strip(read(joinpath(RC_ROOT,"julia-version"),String))) || error("Wrong Julia version")
    Threads.nthreads()==1 || error("Use one Julia compute thread")
    BLAS.set_num_threads(1);c=rc_config(configpath);job=TOML.parsefile(jobpath)
    mkpath(out);method=job["method"];criterion=job["criterion"]
    method=="BLOCK" && criterion!="average" && error("BLOCK is average-only")
    t=@timed begin
        m=rc_model(c,job["N"],job["A"],job["profile"])
        input=method=="BLOCK" ? nothing : rc_input(m)
    end
    record=Dict{String,Any}("job"=>job,"model"=>rc_record(m),"status"=>"validating",
        "scope"=>"Constructive small-positive-curvature family; empirical path checks, not a family-wide PCL certificate",
        "selection_contract"=>MultiGearBandits._SELECTION_CONTRACT,
        "implementation_contract"=>"reference-optimized-20261003",
        "policy_family"=>"ordered_threshold",
        "comparison_safeguard"=>Dict("bound_method"=>get(c,"residual_bound","inverse"),
            "state_limit"=>get(c,"residual_max_states",256),
            "cost_included_in_timings"=>true),
        "generation_and_input_seconds"=>t.time,"generation_allocated_bytes"=>t.bytes,
        "environment"=>Dict("julia"=>string(VERSION),"blas"=>string(BLAS.get_config()),
            "blas_threads"=>BLAS.get_num_threads(),"compute_threads"=>Threads.nthreads(),
            "cpu_target"=>Sys.CPU_NAME,"system_memory_bytes"=>Int(Sys.total_memory())),
        "measurements"=>Any[],"cross_method_validated"=>false)
    if input!==nothing
        record["dense_array_sha256"]=Dict(string(name)=>bytes2hex(sha256(reinterpret(UInt8,vec(getfield(input.model,name))))) for name in (:P,:h,:c))
    end
    rc_save(joinpath(out,"result.toml"),record)
    clock=TimingClocks();record["clock"]=timing_clock_metadata(clock)
    smallm=rc_model(c,9,job["A"],job["profile"]);small=method=="BLOCK" ? nothing : rc_input(smallm)
    for _ in 1:c["warmups"];timing_instrument(clock,rc_execution,small,smallm,method,criterion,c);end
    small=nothing;GC.gc()
    ref,checks,valid=rc_validate(input,m,method,criterion,c,joinpath(out,"progress.toml"))
    record["audit"]=checks;record["algorithm_status"]=string(ref.status)
    record["accepted_assignments"]=length(ref.order);record["counts"]=ref.counts
    rc_bin(joinpath(out,"snapshot.bin"),ref)
    if !valid
        record["status"]="validation_failed";rc_save(joinpath(out,"result.toml"),record);return
    end
    work=rc_work(ref,job["N"],job["A"],criterion)
    rc_check_work(ref,work) || error("Reconstructed work disagrees with counters")
    record["work"]=work
    for rep in 1:c["repetitions"]
        GC.gc();rc_save(joinpath(out,"progress.toml"),Dict("stage"=>"measurement","repetition"=>rep))
        t=timing_instrument(clock,rc_execution,input,m,method,criterion,c);v=t.measured.value;s=v.snapshot
        okay=rc_matching(s,ref) && timing_work_count_check([s],ref).passed &&
            t.measured.compile_time+t.measured.recompile_time==0
        if !okay
            empty!(record["measurements"]);record["status"]="measurement_failed"
            record["rejected_repetition"]=rep;rc_save(joinpath(out,"result.toml"),record);return
        end
        ratio=t.outer_wall_seconds>0 ? t.cpu_seconds/t.outer_wall_seconds : 0.0
        flag=(t.outer_wall_seconds>=.1 && !(0.8<=ratio<=1.2)) ||
            abs(t.continuous_seconds-t.outer_wall_seconds)>max(.05,.02*t.outer_wall_seconds)
        push!(record["measurements"],Dict("repetition"=>rep,"seconds"=>t.measured.time,
            "initialization_seconds"=>v.initialization_seconds,"index_loop_seconds"=>v.index_loop_seconds,
            "output_seconds"=>v.output_seconds,"allocated_bytes"=>t.measured.bytes,"gc_seconds"=>t.measured.gctime,
            "cpu_seconds"=>t.cpu_seconds,"cpu_wall_ratio"=>ratio,"continuous_seconds"=>t.continuous_seconds,
            "diagnostic_flag"=>flag,"compile_seconds"=>t.measured.compile_time+t.measured.recompile_time))
        record["status"]="measuring";rc_save(joinpath(out,"result.toml"),record)
    end
    record["status"]="complete_provisional";record["peak_rss_bytes"]=Int(Sys.maxrss())
    rc_save(joinpath(out,"result.toml"),record)
end
if abspath(PROGRAM_FILE)==@__FILE__
    try
        rc_worker(ARGS...)
    catch err
        rc_save(joinpath(ARGS[3],"error.toml"),Dict("error"=>sprint(showerror,err,catch_backtrace())))
        rethrow()
    end
end
