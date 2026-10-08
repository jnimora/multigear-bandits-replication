module BinaryBenchmark
using LinearAlgebra, Random, TOML, SHA, Dates
const ROOT = get(ENV,"BINARY_BENCH_REPO",normpath(joinpath(@__DIR__,"../..")))
include(joinpath(ROOT,"src/MultiGearBandits.jl"))
using .MultiGearBandits
const MGB = MultiGearBandits

"""Numerical screens, not interval-arithmetic certificates. Near-zero work is unresolved."""
Base.@kwdef struct Checks
    work_atol::Float64 = 1e-12
    work_rtol::Float64 = 1e-10
    monotone_atol::Float64 = 1e-10
    monotone_rtol::Float64 = 1e-9
    pivot_tol::Float64 = 1e-12
end
band(x,c::Checks)=c.work_atol+c.work_rtol*max(1.0,abs(x))
mono_ok(old,new,up,c)=(!isfinite(old) || (up ? old-new : new-old)>=
    -(c.monotone_atol+c.monotone_rtol*max(1.0,abs(old),abs(new))))

struct Input
    P0::Matrix{Float64}
    P1::Matrix{Float64}
    R0::Vector{Float64}
    R1::Vector{Float64}
end
function generate(n,seed,family)
    rng=Xoshiro(seed)
    p0=family=="uniform" ? rand(rng,n,n) : randexp(rng,n,n)
    p1=family=="uniform" ? rand(rng,n,n) : randexp(rng,n,n)
    p0 ./= sum(p0;dims=2); p1 ./= sum(p1;dims=2)
    r0=family=="uniform" ? zeros(n) : rand(rng,n)
    Input(p0,p1,r0,rand(rng,n))
end
sha(path)=open(io->bytes2hex(sha256(io)),path)
function saveinput(dir,x)
    mkpath(dir); n=length(x.R0); hashes=Dict{String,String}()
    for key in (:P0,:P1,:R0,:R1)
        path=joinpath(dir,string(key)*".f64")
        open(io->write(io,getproperty(x,key)),path,"w"); hashes[basename(path)]=sha(path)
    end
    open(joinpath(dir,"input.toml"),"w") do io
        TOML.print(io,Dict("N"=>n,"format"=>"little-endian Float64, column-major, no header",
            "sha256"=>hashes,"resource"=>"c0=0,c1=1"))
    end
end
function loadinput(dir)
    d=TOML.parsefile(joinpath(dir,"input.toml")); n=d["N"]
    data=Any[]
    for (key,dims) in (("P0",(n,n)),("P1",(n,n)),("R0",(n,)),("R1",(n,)))
        path=joinpath(dir,key*".f64"); sha(path)==d["sha256"][key*".f64"] || error("Input hash mismatch")
        a=Array{Float64}(undef,dims); open(io->read!(io,a),path); push!(data,a)
    end
    Input(data...)
end

# Replacing column 1 of I-beta*P by 1 is a common, nonsingular change
# of unknowns for beta<1 and the bias/gain normalization for beta=1.
# The identical change is applied to BOTH action matrices and Delta.
function system(P,beta)
    n=size(P,1); M=-beta*P
    @inbounds for i in 1:n; M[i,i]+=1; M[i,1]=1; end
    M
end
function delta(x,beta)
    D=beta*(x.P1-x.P0); D[:,1].=0; D
end
function initial(x,beta,up;rectangular=false)
    pc=up ? x.P0 : x.P1; pr=up ? x.P1 : x.P0
    F=lu!(system(pc,beta)); n=length(x.R0)
    if !rectangular
        # The published minimal-tableau initializer specialized to unit work.
        # At an all-passive/all-active endpoint the work marginal is exactly 1.
        # This avoids a second NxN transition-difference matrix and value solves.
        Xt=copy(transpose(system(pr,beta))); ldiv!(transpose(F),Xt); X=copy(transpose(Xt))
        f=up ? x.R1-X*x.R0 : X*x.R1-x.R0; g=ones(n)
        all(isfinite,X) && all(isfinite,f) || error("Nonfinite initialization")
        return X,f,g
    end
    B=hcat(up ? x.R0 : x.R1,fill(up ? 0.0 : 1.0,n))
    ldiv!(F,B)
    D=delta(x,beta)
    f=x.R1-x.R0+D*@view(B[:,1]); g=ones(n)+D*@view(B[:,2])
    rhs=rectangular ? D : system(pr,beta)
    Xt=copy(transpose(rhs)); ldiv!(transpose(F),Xt); X=copy(transpose(Xt))
    all(isfinite,X) && all(isfinite,f) && all(isfinite,g) || error("Nonfinite initialization")
    X,f,g
end

mutable struct FP
    X::Matrix{Float64}
    f::Vector{Float64}
    g::Vector{Float64}
    ids::Vector{Int}
    indices::Vector{Float64}
    order::Vector{Int}
    sequence::Vector{Float64}
    column::Vector{Float64}
    m::Int
    k::Int
    up::Bool
    status::Symbol
    rejected::Int
    minimum_work::Float64
    minimum_gap::Float64
end
function prepare_fp(x,beta,up=false)
    X,f,g=initial(x,beta,up); n=length(f)
    FP(X,f,g,collect(1:n),fill(NaN,n),zeros(Int,n),zeros(n),zeros(n),n,0,up,:ready,0,Inf,Inf)
end

function select_positive(f,g,ids,m,up,c)
    best=0; value=up ? -Inf : Inf; rejected=0; minwork=Inf
    @inbounds for s in 1:m
        ff=f[s]; gg=g[s]
        isfinite(ff) && isfinite(gg) || return (0,NaN,:nonfinite,rejected,minwork)
        minwork=min(minwork,gg); b=band(gg,c)
        if gg < -b; rejected+=1; continue; end
        gg<=b && return (0,NaN,:unresolved_work,rejected,minwork)
        v=ff/gg; isfinite(v) || return (0,NaN,:nonfinite,rejected,minwork)
        if best==0 || (up ? v>value : v<value) || (v==value && ids[s]<ids[best])
            best=s; value=v
        end
    end
    (best,value,best==0 ? :no_positive_candidate : :ok,rejected,minwork)
end

# Packed single-buffer Schur update. Swap the selected coordinate to the
# boundary, then update a contiguous leading block; no matrix slicing copies,
# outer-product temporaries, or per-pivot allocations. No @fastmath.
function pivot!(w::FP,p,c)
    m=w.m; eta=w.X[p,p]
    isfinite(eta) && eta>c.pivot_tol || return :unsafe_pivot
    if p!=m
        @inbounds for j in 1:m; w.X[p,j],w.X[m,j]=w.X[m,j],w.X[p,j]; end
        @inbounds for i in 1:m; w.X[i,p],w.X[i,m]=w.X[i,m],w.X[i,p]; end
        w.f[p],w.f[m]=w.f[m],w.f[p]; w.g[p],w.g[m]=w.g[m],w.g[p]
        w.ids[p],w.ids[m]=w.ids[m],w.ids[p]
    end
    r=m-1; fd=w.f[m]/eta; gd=w.g[m]/eta
    @inbounds for i in 1:r
        v=w.X[i,m]; w.column[i]=v
        w.f[i]-=fd*v; w.g[i]-=gd*v
    end
    @inbounds for j in 1:r
        z=w.X[m,j]/eta
        @simd for i in 1:r; w.X[i,j]-=w.column[i]*z; end
    end
    w.m=r
    :ok
end
function run_fp!(w::FP,c=Checks();checks=true)
    n=length(w.indices)
    while w.k<n
        p,v,st,rej,mw=select_positive(w.f,w.g,w.ids,w.m,w.up,c)
        w.rejected+=rej; w.minimum_work=min(w.minimum_work,mw)
        st==:ok || (w.status=st;return w)
        if w.k>0
            old=w.sequence[w.k]; gap=w.up ? old-v : v-old; w.minimum_gap=min(w.minimum_gap,gap)
            checks && !mono_ok(old,v,w.up,c) && (w.status=:nonmonotone;return w)
        end
        k=w.k+1; id=w.ids[p]; w.indices[id]=v; w.order[k]=id; w.sequence[k]=v; w.k=k
        if k<n
            st=pivot!(w,p,c); st==:ok || (w.status=st;return w)
        end
    end
    w.status=:complete; w
end

function mg_input(x)
    n=length(x.R0); P=cat(x.P0,x.P1;dims=3)
    PerformanceInput(FiniteMultiGearModel(P,-hcat(x.R0,x.R1),hcat(zeros(n),ones(n))))
end
function prepare_mg(x,beta)
    prepare_performance_run(x,:FP;criterion=beta==1 ? :average : :discounted,
        beta=beta==1 ? nothing : beta)
end

# Experimental adapter only: archived production driver and kernels unchanged.
# Positive candidates are selected in original state labels. The original
# reduced pivot kernel and initialization are used without replacement.
function run_mg!(run,c=Checks())
    w=run.core; log=run.log; n=length(log.state); minwork=Inf; mingap=Inf; rejected=0
    while log.k<n
        p,v,st,rej,mw=select_positive(w.f,w.g,w.state_at,w.m,false,c)
        rejected+=rej; minwork=min(minwork,mw)
        st==:ok || (run.status=st;return (run,minwork,mingap,rejected))
        if log.k>0
            old=log.ratio[log.k]; mingap=min(mingap,v-old)
            mono_ok(old,v,false,c) || (run.status=:nonmonotone;return (run,minwork,mingap,rejected))
        end
        id=w.state_at[p]; ff=w.f[p]; gg=w.g[p]; k=log.k+1; dim=w.m
        if k<n
            report=pivot_reduced_fp!(w,id;pivot_atol=0.0,pivot_rtol=c.pivot_tol,check_unichain=false)
            report.accepted || (run.status=report.reason;return (run,minwork,mingap,rejected))
        else
            run.terminal_update_skipped=true
        end
        log.state[k]=id; log.gear[k]=1; log.f[k]=ff; log.g[k]=gg; log.ratio[k]=v
        log.dimension[k]=dim; log.evidence[k]=:positive_work_screen
        log.mpi[id,1]=v; log.k=k; w.k=k
    end
    run.status=:complete; (run,minwork,mingap,rejected)
end

function result(w::FP)
    Dict{String,Any}("status"=>string(w.status),"indices"=>copy(w.indices),"order"=>w.order[1:w.k],
        "sequence"=>w.sequence[1:w.k],"minimum_candidate_work"=>w.minimum_work,
        "minimum_monotone_gap"=>w.minimum_gap,"negative_candidates_skipped"=>w.rejected)
end
function result(t::Tuple)
    run,mw,gap,rej=t; l=run.log
    Dict{String,Any}("status"=>string(run.status),"indices"=>[ismissing(x) ? NaN : x for x in l.mpi[:,1]],
        "order"=>l.state[1:l.k],"sequence"=>l.ratio[1:l.k],"minimum_candidate_work"=>mw,
        "minimum_monotone_gap"=>gap,"negative_candidates_skipped"=>rej)
end

"""Independent rectangular derivative-tableau audit of ALL state marginals
on the complete visited path, including eliminated states and the terminal
policy. Floating-point PCL-path validation, not a rigorous rounding bound.
Retains only O(N) output; its O(N^3) computation is outside timed kernels."""
function validate_path(x,beta,order,indices;up=false,c=Checks(),fresh_steps=Int[])
    n=length(x.R0); length(order)==n && sort(order)==collect(1:n) || error("Incomplete permutation")
    Y,f,g=initial(x,beta,up;rectangular=true)
    ids=collect(1:n); column=zeros(n); actions=fill(up ? 0 : 1,n)
    minimum_work=Inf; max_index_error=0.0; fresh_error=0.0; rows=Dict{String,Any}[]
    for k in 0:n
        mg=minimum(g); minimum_work=min(minimum_work,mg)
        mg>band(mg,c) || return Dict("status"=>"nonpositive_or_unresolved_path_work","step"=>k,"minimum_work"=>mg)
        if k in fresh_steps
            pc=similar(x.P0); rc=zeros(n); gc=Float64.(actions)
            @inbounds for j in 1:n,i in 1:n; pc[i,j]=actions[i]==1 ? x.P1[i,j] : x.P0[i,j]; end
            @inbounds for i in 1:n; rc[i]=actions[i]==1 ? x.R1[i] : x.R0[i]; end
            B=system(pc,beta)\hcat(rc,gc); D=delta(x,beta)
            ff=x.R1-x.R0+D*@view(B[:,1]); gg=ones(n)+D*@view(B[:,2])
            er=max(maximum(abs.(ff-f)./(1 .+abs.(ff))),maximum(abs.(gg-g)./(1 .+abs.(gg))))
            fresh_error=max(fresh_error,er); er<=1e-7 || return Dict("status"=>"fresh_solve_disagreement","step"=>k,"error"=>er)
        end
        push!(rows,Dict("step"=>k,"minimum_work"=>mg))
        k==n && break
        m=n-k; id=order[k+1]; p=findfirst(==(id),@view(ids[1:m])); p===nothing && error("Repeated state")
        v=f[id]/g[id]; er=abs(v-indices[id])/(1+abs(indices[id])); max_index_error=max(max_index_error,er)
        er<=1e-7 || return Dict("status"=>"index_disagreement","step"=>k,"error"=>er)
        # Check greedy choice against ALL remaining states, not just selected edge.
        best=up ? maximum(f[i]/g[i] for i in @view(ids[1:m])) : minimum(f[i]/g[i] for i in @view(ids[1:m]))
        abs(best-v)<=1e-8*(1+abs(v)) || return Dict("status"=>"selection_disagreement","step"=>k)
        sign=up ? -1.0 : 1.0; eta=1+sign*Y[id,p]
        eta>c.pivot_tol || return Dict("status"=>"unsafe_audit_pivot","step"=>k)
        @inbounds for i in 1:n; column[i]=Y[i,p]; end
        if p!=m
            @inbounds for i in 1:n; Y[i,p]=Y[i,m]; end
            ids[p]=ids[m]
        end
        fd=sign*f[id]/eta; gd=sign*g[id]/eta
        @inbounds @simd for i in 1:n; f[i]-=fd*column[i]; g[i]-=gd*column[i]; end
        @inbounds for j in 1:m-1
            z=sign*Y[id,j]/eta
            @simd for i in 1:n; Y[i,j]-=column[i]*z; end
        end
        actions[id]=up ? 1 : 0
    end
    seq=indices[order]
    all(mono_ok(seq[k],seq[k+1],up,c) for k in 1:n-1) || return Dict("status"=>"nonmonotone")
    Dict("status"=>"numerical_pcl_path_pass","minimum_work"=>minimum_work,
        "maximum_relative_index_disagreement"=>max_index_error,"maximum_fresh_solve_error"=>fresh_error,
        "all_states_all_visited_policies"=>true,"terminal_policy_checked"=>true,
        "rigorous_interval_certificate"=>false,"path_rows"=>rows)
end
end
