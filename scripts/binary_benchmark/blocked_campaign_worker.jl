include(joinpath(@__DIR__,"core.jl"))
include(joinpath(@__DIR__,"blocked_fp.jl"))
const BF=BlockedBinaryFP

using .BinaryBenchmark, LinearAlgebra, TOML, Dates, SHA
const BB=BinaryBenchmark
BB.result(w::BF.Workspace)=BF.result(w)
BLAS.set_num_threads(1)
Threads.nthreads()==1 || error("Use one Julia thread")
function emit(path,d)
    mkpath(dirname(path)); open(io->TOML.print(io,d;sorted=true),path*".tmp","w")
    mv(path*".tmp",path;force=true)
end
function envinfo()
    Dict("julia"=>string(VERSION),"blas"=>string(BLAS.get_config()),"blas_threads"=>BLAS.get_num_threads(),
        "julia_threads"=>Threads.nthreads(),"cpu"=>Sys.CPU_NAME,"machine"=>Sys.MACHINE)
end
function prepare(x,m,beta,method)
    method=="FP_blocked_down" ? BF.prepare(x,beta;blocksize=32,tail=64) :
        method=="FP_multigear" ? BB.prepare_mg(m,beta) : BB.prepare_fp(x,beta,method=="FP2020_up")
end
function execute!(w,method)
    method=="FP_blocked_down" ? BF.run!(w) :
        method=="FP_multigear" ? BB.run_mg!(w) : BB.run_fp!(w)
end
function whole(x,m,beta,method)
    BB.result(execute!(prepare(x,m,beta,method),method))
end
function phased(x,m,beta,method)
    t0=time_ns(); w=prepare(x,m,beta,method); t1=time_ns()
    w=execute!(w,method); t2=time_ns(); out=BB.result(w); t3=time_ns()
    (out,Float64(t1-t0)/1e9,Float64(t2-t1)/1e9,Float64(t3-t2)/1e9)
end
function agrees(out,ref,order=nothing)
    out["status"]=="complete" && all(isfinite,out["indices"]) &&
    maximum(abs.(out["indices"]-ref)./(1 .+abs.(ref)))<=1e-7 &&
    (order===nothing || out["order"]==order)
end
function work(jobpath,outdir)
    job=TOML.parsefile(jobpath); mkpath(outdir); n=job["N"]
    started=time(); progress(d)=emit(joinpath(outdir,"progress.toml"),merge(Dict("utc"=>string(now(UTC))),d))
    if job["kind"]=="generate"
        progress(Dict("stage"=>"generate"))
        x=BB.generate(n,job["seed"],job["family"]); BB.saveinput(outdir,x)
        emit(joinpath(outdir,"result.toml"),Dict("status"=>"complete","seconds"=>time()-started,"job"=>job))
        return
    end
    beta=job["beta"]; x=BB.loadinput(job["input"])
    if job["kind"]=="validate"
        # Compile on tiny inputs before the full validation. No timings are used
        # as benchmark observations in this phase.
        tiny=BB.generate(8,123,job["family"])
        for up in (false,true)
            w=BB.run_fp!(BB.prepare_fp(tiny,beta,up)); BB.validate_path(tiny,beta,w.order,w.indices;up=up,fresh_steps=[0,4,8])
        end
        checks=Dict{String,Any}(); ref=Float64[]; valid=true
        for up in (false,true)
            label=up ? "up" : "down"; progress(Dict("stage"=>"pcl_validation","orientation"=>label))
            GC.gc(); w=BB.run_fp!(BB.prepare_fp(x,beta,up)); out=BB.result(w)
            if w.status==:complete
                ref=isempty(ref) ? copy(w.indices) : ref
                audit=BB.validate_path(x,beta,w.order,w.indices;up=up,fresh_steps=unique([0,n÷2,n]))
                ok=audit["status"]=="numerical_pcl_path_pass" && agrees(out,ref)
            else
                audit=Dict("status"=>"fp_incomplete"); ok=false
            end
            checks[label]=Dict("output"=>out,"audit"=>audit,"passed"=>ok); valid &=ok
        end
        emit(joinpath(outdir,"result.toml"),Dict("status"=>valid ? "eligible" : "rejected_or_unresolved",
            "job"=>job,"checks"=>checks,"reference_indices"=>ref,"environment"=>envinfo(),
            "elapsed_seconds"=>time()-started,"peak_rss_bytes"=>Int(Sys.maxrss())))
        return
    end
    validation=TOML.parsefile(job["validation"])
    ref=validation["reference_indices"]
    pathorder(method)=validation["checks"][method=="FP2020_up" ? "up" : "down"]["output"]["order"]
    m=BB.mg_input(x) # shared admission/layout preparation, outside algorithm timing
    tiny=BB.generate(8,125,job["family"]); tm=BB.mg_input(tiny)
    rows=Dict{String,Any}[]; outputs=Dict{String,Any}()
    methods=job["methods"]
    for method in methods
        for _ in 1:2; whole(tiny,tm,beta,method); phased(tiny,tm,beta,method); end
    end
    # First compile the enclosing measurement paths too. Any residual compilation
    # in a full-size run is recorded and disqualifies that timing observation.
    for method in methods; @timed whole(tiny,tm,beta,method); @timed phased(tiny,tm,beta,method); end
    data=Dict{String,Any}("status"=>"running","job"=>job,"environment"=>envinfo(),"measurements"=>rows,"outputs"=>outputs)
    for rep in 1:job["repetitions"]
        ordering=isodd(rep) ? methods : reverse(methods)
        for method in ordering
            progress(Dict("stage"=>"timing","method"=>method,"repetition"=>rep))
            GC.gc(); cpu0=ccall(:clock,Clong,())/1e6
            t=@timed whole(x,m,beta,method)
            cpu=ccall(:clock,Clong,())/1e6-cpu0; out=t.value
            valid=agrees(out,ref,pathorder(method)) && t.compile_time+t.recompile_time==0
            push!(rows,Dict("method"=>method,"phase"=>"end_to_end","repetition"=>rep,"seconds"=>t.time,
                "allocated_bytes"=>Int(t.bytes),"gc_seconds"=>t.gctime,"compile_seconds"=>t.compile_time+t.recompile_time,
                "cpu_seconds"=>cpu,"valid"=>valid,"output_status"=>out["status"],
                "order_matches_audited_path"=>out["order"]==pathorder(method),
                "maximum_relative_index_error"=>maximum(abs.(out["indices"]-ref)./(1 .+abs.(ref)))))
            outputs[method]=out; emit(joinpath(outdir,"result.toml"),data)
        end
    end
    for rep in 1:get(job,"phase_repetitions",1), method in (isodd(rep) ? methods : reverse(methods))
        progress(Dict("stage"=>"phase_diagnostic","method"=>method,"repetition"=>rep)); GC.gc()
        t=@timed phased(x,m,beta,method); out,ti,tl,to=t.value
        for (phase,seconds) in (("initialization",ti),("index_loop",tl),("output",to))
            push!(rows,Dict("method"=>method,"phase"=>phase,"repetition"=>rep,"seconds"=>seconds,
                "valid"=>agrees(out,ref,pathorder(method)) && t.compile_time+t.recompile_time==0,
                "scope"=>"separate diagnostic execution; not pooled with primary timings"))
        end
        emit(joinpath(outdir,"result.toml"),data)
    end
    data["status"]=all(r["valid"] for r in rows) ? "complete" : "measurement_failed"
    data["elapsed_seconds"]=time()-started; data["peak_rss_bytes"]=Int(Sys.maxrss())
    emit(joinpath(outdir,"result.toml"),data)
end
try
    work(ARGS...)
catch err
    length(ARGS)>=2 && emit(joinpath(ARGS[2],"error.toml"),Dict("error"=>sprint(showerror,err,catch_backtrace())))
    rethrow()
end
