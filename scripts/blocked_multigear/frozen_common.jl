# Reconstruct stored primitives; no random draws, menu recomputation, or tuning.
using LinearAlgebra,TOML,SHA,Serialization,Dates,Statistics
include("blocked_fp.jl")
using .MultiGearBandits
const BFP=BlockedMultiGearFP
include("../controlled_timing_clocks.jl")
BLAS.set_num_threads(1)
save_record(path,data)=begin
    mkpath(dirname(path));open(io->TOML.print(io,data;sorted=true),path*".tmp","w")
    mv(path*".tmp",path;force=true)
end
file_hash(path)=open(io->bytes2hex(sha256(io)),path)
function array_hashes(model)
    Dict(string(k)=>bytes2hex(sha256(reinterpret(UInt8,vec(getfield(model,k))))) for k in (:P,:h,:c))
end
function combined_hash(model)
    context=SHA.SHA2_256_CTX()
    for k in (:P,:h,:c);SHA.update!(context,reinterpret(UInt8,vec(getfield(model,k))));end
    bytes2hex(SHA.digest!(context))
end
function frozen_fraction(s)
    q=split(replace(s,"//"=>"/"),'/')
    length(q) in (1,2) || error("Invalid stored fraction")
    Float64(parse(BigInt,q[1])//(length(q)==1 ? BigInt(1) : parse(BigInt,q[2])))
end
function reconstruct(old,kind;warmup=false)
    meta=old["model"];H=warmup ? 3 : Int(meta["H"]);A=Int(meta["A"])
    B=kind=="rayleigh" ? 1 : Int(meta["B"]);N=B*(H+1)
    P=zeros(N,N,A+1);h=zeros(N,A+1);c=zeros(N,A+1);mask=falses(N)
    if kind=="rayleigh"
        wR=frozen_fraction(meta["wR"]);reset=frozen_fraction.(meta["p"]);resource=frozen_fraction.(meta["c"])
        for a in 0:A
            P[1,1,a+1]=wR;P[1,2,a+1]=1.0-wR
            for age in 1:H
                P[age+1,1,a+1]=reset[a+1];P[age+1,min(age+1,H)+1,a+1]=1.0-reset[a+1]
                h[age+1,a+1]=Float64(age);c[age+1,a+1]=resource[a+1]
            end
        end
        mask[2:end].=true;family=OrderedThresholdFamily(collect(2:N))
        topology=:reset_or_increment
    elseif kind=="channel"
        wR=Float64(meta["wR"]);Q=reduce(vcat,transpose.(meta["Q"]))
        success=reduce(vcat,transpose.(meta["success"]));resource=Float64.(meta["resource"])
        idx(age,b)=(b-1)*(H+1)+age+1
        for b in 1:B,age in 0:H
            i=idx(age,b);mask[i]=age>0
            for a in 0:A
                reset=age==0 ? wR : 1-wR+(2wR-1)*success[b,a+1]
                for bp in 1:B
                    P[i,idx(0,bp),a+1]+=reset*Q[b,bp]
                    P[i,idx(min(age+1,H),bp),a+1]+=(1-reset)*Q[b,bp]
                end
                h[i,a+1]=age;c[i,a+1]=age==0 ? 0.0 : resource[a+1]
            end
        end
        family=RowThresholdFamily([[idx(age,b) for age in 1:H] for b in 1:B])
        topology=:common_reachable_state
    else
        error("Unknown frozen cohort")
    end
    model=FiniteMultiGearModel(P,h,c;controllable=mask)
    (;input=PerformanceInput(model;average_topology=topology),family)
end
function input_identity(old,kind,input)
    model=input.model
    if kind=="rayleigh"
        hashes=array_hashes(model)
        return Dict("passed"=>hashes==old["dense_array_sha256"],"dense_array_sha256"=>hashes,
                    "original_primitive_sha256"=>old["model"]["primitive_sha256"])
    end
    digest=combined_hash(model)
    Dict("passed"=>digest==old["model"]["primitive_sha256"],"primitive_sha256"=>digest)
end
function frozen_options(kind)
    kind=="rayleigh" ? PerformanceOptions(exact_max_states=8,residual_fallback=true,
        residual_max_states=10000,residual_bound=:contraction) :
        PerformanceOptions(exact_max_states=12,residual_fallback=true,
        residual_max_states=128,residual_bound=:inverse)
end
function prepare_run(input,family,job,variant)
    criterion=Symbol(job["criterion"])
    kw=criterion==:discounted ? (;criterion,beta=job["beta"]) : (;criterion)
    options=frozen_options(job["kind"])
    if variant in ("FP","M")
        prepare_performance_run(input,Symbol(variant);kw...,family,options)
    else
        variant in ("FP_BLOCKED","FP_CACHED") || error("Unknown implementation")
        BFP.prepare(input;kw...,family,options,blocksize=32,
                    reference_width=variant=="FP_CACHED" ? 8 : 0,reference_panels=2,reference_order=:physical)
    end
end
execute!(r::PerformanceRun)=run_performance!(r)
execute!(r::BFP.Workspace)=BFP.run!(r)
snapshot(r::PerformanceRun)=performance_snapshot(r)
snapshot(r::BFP.Workspace)=BFP.snapshot(r)
base(r::PerformanceRun)=r
base(r::BFP.Workspace)=r.run
block_counts(r::PerformanceRun)=Dict{String,Any}()
function block_counts(r::BFP.Workspace)
    Dict{String,Any}("flushes"=>r.flushes,"flushed_rank"=>r.flushed_rank,"flush_work"=>r.flush_work,
        "cached_rows"=>r.cached_rows,"direct_rows"=>r.direct_rows,"panel_fills"=>r.panel_fills,
        "blocksize"=>r.blocksize,"reference_width"=>r.reference_width,"reference_panels"=>length(r.ages))
end
function total_execution(input,family,job,variant)
    r=prepare_run(input,family,job,variant);execute!(r);snapshot(r)
end
function phase_execution(input,family,job,variant)
    t0=time_ns();r=prepare_run(input,family,job,variant);t1=time_ns()
    execute!(r);t2=time_ns();s=snapshot(r);t3=time_ns()
    (;snapshot=s,initialization=(t1-t0)/1e9,index_loop=(t2-t1)/1e9,output=(t3-t2)/1e9,blocking=block_counts(r))
end
function agreement(s,reference)
    s.complete && reference.complete && performance_agreement(s,reference;atol=1e-8,rtol=1e-7) &&
        all(isapprox.(s.f,reference.f;atol=1e-8,rtol=1e-7)) && all(isapprox.(s.g,reference.g;atol=1e-8,rtol=1e-7))
end
function errors(s,reference)
    length(s.ratio)==length(reference.ratio) || return Dict("same_length"=>false)
    d=Dict{String,Any}("same_length"=>true)
    for k in (:f,:g,:ratio)
        x=getfield(s,k);y=getfield(reference,k)
        d[string(k)*"_scaled_error"]=maximum((abs(a-b)/max(1.0,abs(a),abs(b)) for (a,b) in zip(x,y));init=0.0)
    end
    d
end
stable_counts(s)=Dict(k=>v for (k,v) in s.counts if k!="audit_solves")
function checked_blocked(input,family,job,variant,progress)
    r=prepare_run(input,family,job,variant);run=base(r);K=length(run.log.state)
    targets=unique([0,(K-1)÷2,K-1]);audits=Any[]
    while run.status in (:ready,:running)
        if run.log.k in targets
            a=BFP.audit(r;atol=1e-8,rtol=1e-7)
            push!(audits,Dict("step"=>run.log.k,"passed"=>a.passed,"tableau_error"=>a.tableau_error,
                            "f_error"=>a.cached_f_error,"g_error"=>a.cached_g_error))
            save_record(progress,Dict("stage"=>"independent_audit","variant"=>variant,"step"=>run.log.k))
            a.passed || break
        end
        BFP.step!(r)
    end
    s=snapshot(r)
    (;snapshot=s,audits,passed=s.complete && length(audits)==length(targets) && all(a->a["passed"],audits),blocking=block_counts(r))
end
function clocks_record(t)
    ratio=t.outer_wall_seconds>0 ? t.cpu_seconds/t.outer_wall_seconds : 0.0
    gap=abs(t.continuous_seconds-t.outer_wall_seconds)
    flag=(t.outer_wall_seconds>=.1 && !(0.8<=ratio<=1.2)) || gap>max(.05,.02*t.outer_wall_seconds)
    Dict{String,Any}("seconds"=>t.measured.time,"cpu_seconds"=>t.cpu_seconds,"outer_seconds"=>t.outer_wall_seconds,
        "continuous_seconds"=>t.continuous_seconds,"cpu_wall_ratio"=>ratio,"diagnostic_flag"=>flag,
        "compiled_seconds"=>t.measured.compile_time+t.measured.recompile_time,
        "allocated_bytes"=>t.measured.bytes,"gc_seconds"=>t.measured.gctime)
end
