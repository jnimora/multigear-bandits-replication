using LinearAlgebra, TOML, Dates, SHA, Statistics, Serialization
const RC_ROOT=normpath(joinpath(@__DIR__,"../.."))
include(joinpath(RC_ROOT,"src/MultiGearBandits.jl"))
using .MultiGearBandits
include(joinpath(RC_ROOT,"scripts/rayleigh_aoii_support.jl"))
include(joinpath(RC_ROOT,"scripts/rayleigh_block_support.jl"))
include(joinpath(RC_ROOT,"scripts/controlled_timing_clocks.jl"))
include(joinpath(RC_ROOT,"scripts/timing_work_counts.jl"))
const RQ=Rational{BigInt}
function rc_save(path,data)
    mkpath(dirname(path));open(io->TOML.print(io,data;sorted=true),path*".tmp","w")
    mv(path*".tmp",path;force=true)
end
rc_bin(path,data)=open(io->serialize(io,data),path,"w")
rc_load(path)=open(deserialize,path)
function rc_config(path)
    c=TOML.parsefile(path)
    c["protocol"] in ("rayleigh-checked-timing-v1","rayleigh-checked-timing-v2") || error("Unknown protocol")
    c["repetitions"]==2 && c["warmups"]==2 || error("Use two repetitions and two small warmups")
    0<c["campaign_seconds"]<=72000 || error("Campaign must fit within twenty hours")
    0<c["beta"]<1 && 0<c["worker_seconds"]<=7200 || error("Invalid limits")
    c["curvature_ratio_denominator"] in (256,1024,4096) || error("Unplanned curvature")
    c["methods"]==["R","M","FP","BLOCK"] || error("Keep the four planned methods")
    c["criteria"]==["discounted","average"] || error("Keep both criteria")
    length(c["profiles"])==2 || error("Use two fixed parameter profiles")
    for p in c["panels"]
        all(n->2<=n<=10001,p["states"]) && all(a->a in (1,2,4,8,16),p["gears"]) || error("Invalid grid")
    end
    c
end
function rc_model(c,n,A,profile)
    spec=c["profiles"][profile];q=RayleighAoII.parse_q
    k=q(spec["kappa"]);smax=q(spec["success_max"])
    m=RayleighAoII.from_success(n-1;nsrc=2,wR=spec["wR"],chi=1,
        success=[smax*a/A for a in 0:A],kappa=k,epsilon=k/c["curvature_ratio_denominator"])
    # Exact dyadic menus ensure BLOCK's rational source and dense Float64
    # implementations operate on identical mathematical primitives.
    all(x->RQ(Float64(x))==x,vcat(m.p,m.c,[m.wR])) || error("Primitive is not exactly representable")
    m
end
function rc_record(m)
    d=Dict("H"=>m.H,"A"=>length(m.p)-1,"wR"=>string(m.wR),"p"=>string.(m.p),
        "c"=>string.(m.c),"kappa"=>string(m.kappa),"epsilon"=>string(m.epsilon))
    d["primitive_sha256"]=bytes2hex(sha256(join(vcat(string(m.H),string(m.wR),string.(m.p),string.(m.c)),"|")))
    d
end
rc_input(m)=PerformanceInput(RayleighAoII.dense_model(m;exact=false);average_topology=:reset_or_increment)
rc_options(c=Dict())=PerformanceOptions(residual_fallback=true,
    residual_max_states=get(c,"residual_max_states",256),
    residual_bound=Symbol(get(c,"residual_bound","inverse")),exact_max_states=8,terminal_update=false)
function rc_prepare(input,m,method,criterion,c)
    method=="BLOCK" && return rc_block_prepare(m,c)
    prepare_performance_run(input,Symbol(method);criterion=Symbol(criterion),
        beta=criterion=="discounted" ? c["beta"] : nothing,
        family=OrderedThresholdFamily(collect(2:m.H+1)),options=rc_options(c))
end

# A new timing adapter: the accepted interval selector and block recursions
# remain unchanged. Enclosure checks and the added in-loop monotonicity guard
# are inside every timed run. It requires no dense representation of the input.
mutable struct RCBlockRun
    workspace::RayleighBlock.NumericBlockWorkspace
    log::MultiGearBandits.PerformanceLog
    status::Symbol
    terminal_update_skipped::Bool
    monotonicity_checks::Int
end
function rc_block_prepare(m,c)
    H=m.H;A=length(m.p)-1;K=H*A
    w=RayleighBlock.NumericBlockWorkspace(m;options=RayleighBlock.BlockOptions(seconds_limit=c["worker_seconds"]))
    mpi=Matrix{Union{Missing,Float64}}(undef,H,A);fill!(mpi,missing)
    log=MultiGearBandits.PerformanceLog(collect(2:H+1),vcat(0,collect(1:H)),mpi,
        zeros(Int,K),zeros(Int,K),zeros(K),zeros(K),zeros(K),zeros(Int,K),fill(:unassigned,K),0)
    RCBlockRun(w,log,:ready,false,0)
end
function rc_block_accept!(r,x,mode)
    l=r.log;w=r.workspace;k=l.k+1;K=length(l.state)
    x.g.lo>0 || (r.status=:unresolved_resource;return r.status)
    if k>1
        r.monotonicity_checks+=1;o=rc_options()
        if !MultiGearBandits._mpi_nondecreasing(l.ratio[k-1],x.index.value,o.monotonicity_atol,o.monotonicity_rtol)
            r.status=:nonmonotone;return r.status
        end
    end
    if k<K
        RayleighBlock.numerical_shift!(w,x.h,x.a)
    else
        r.terminal_update_skipped=true
    end
    l.state[k]=x.h+1;l.gear[k]=x.a;l.f[k]=x.f.value;l.g[k]=x.g.value;l.ratio[k]=x.index.value
    l.dimension[k]=length(w.lengths);l.evidence[k]=mode;l.mpi[x.h,x.a]=x.index.value
    l.k=k;r.status=k==K ? :complete : :running
end
function rc_step!(r::RCBlockRun)
    try
        s=RayleighBlock.choose_frontier!(r.workspace)
        rc_block_accept!(r,s.candidate,s.mode)
    catch err
        err isa RayleighBlock.NumericalIssue || rethrow()
        r.status=err.status
    end
    r.status
end
rc_step!(r::PerformanceRun)=step_performance!(r)
function rc_run!(r)
    while r.status in (:ready,:running);rc_step!(r);end
    r.status
end
rc_snapshot(r::PerformanceRun)=performance_snapshot(r)
function rc_snapshot(r::RCBlockRun)
    l=r.log;k=l.k;counts=copy(r.workspace.counts)
    counts["assignments"]=k;counts["audit_solves"]=0
    counts["monotonicity_checks"]=r.monotonicity_checks
    counts["terminal_numerical_updates_skipped"]=Int(r.terminal_update_skipped)
    (algorithm=:BLOCK,status=r.status,complete=r.status==:complete,
     selection_contract=MultiGearBandits._SELECTION_CONTRACT,
     states=copy(l.states),mpi=copy(l.mpi),order=[(l.state[j],l.gear[j]) for j in 1:k],
     f=l.f[1:k],g=l.g[1:k],ratio=l.ratio[1:k],dimension=l.dimension[1:k],evidence=l.evidence[1:k],
     counts=counts,terminal_update_skipped=r.terminal_update_skipped)
end
function rc_matching(a,b)
    a.complete && b.complete && performance_agreement(a,b;atol=1e-8,rtol=1e-7) &&
        all(isapprox.(a.f,b.f;atol=1e-8,rtol=1e-7)) && all(isapprox.(a.g,b.g;atol=1e-8,rtol=1e-7))
end
function rc_execution(input,m,method,criterion,c)
    t0=time_ns();r=rc_prepare(input,m,method,criterion,c);t1=time_ns()
    rc_run!(r);t2=time_ns();s=rc_snapshot(r);t3=time_ns()
    (;snapshot=s,initialization_seconds=(t1-t0)/1e9,index_loop_seconds=(t2-t1)/1e9,output_seconds=(t3-t2)/1e9)
end
function rc_audit(r::PerformanceRun,m,criterion,c)
    w=r.core
    e=criterion=="average" ? evaluate_average(w.model,w.actions;omega=w.omega) :
        evaluate_discounted(w.model,w.actions;beta=c["beta"])
    V=e isa DiscountedEvaluation ? hcat(e.F,e.G) : vcat(hcat(e.phi,e.gamma),reshape([e.hbar,e.cbar],1,2))
    err=0.0
    for j in 1:w.nfrontier
        i=MultiGearBandits._performance_state(w,j);a=w.actions[i]
        f,g=MultiGearBandits._performance_pair_metrics(w,j)
        ff,gg=MultiGearBandits._performance_metrics(w.model,i,a,criterion=="average" ? 1.0 : c["beta"],V)
        err=max(err,abs(f-ff)/max(1.0,abs(f),abs(ff)),abs(g-gg)/max(1.0,abs(g),abs(gg)))
    end
    err
end
function rc_audit(r::RCBlockRun,m,criterion,c)
    # Fresh scalar excursions, independent of the block statistics, in 128-bit
    # BigFloat. This does not change the precision of a timed execution.
    setprecision(BigFloat,128) do
        acts=RayleighBlock.frozen_actions(r.workspace)
        mm=RayleighAoII.AoIIModel(m.H,m.nsrc,BigFloat(m.wR),BigFloat(m.chi),BigFloat.(m.success),BigFloat(m.kappa),BigFloat(m.epsilon))
        e=RayleighAoII.excursion(mm,acts);rows=RayleighAoII.scalar_frontier(mm,acts,e)
        front=RayleighBlock.numerical_frontier!(deepcopy(r.workspace)).frontier
        length(rows)==length(front) || return Inf
        err=0.0
        for x in front
            j=findfirst(row->row[1]==x.h && row[2]==x.a,rows);j===nothing && return Inf
            y=rows[j]
            for (a,b) in ((x.f.value,y[3]),(x.g.value,y[4]),(x.index.value,y[5]))
                err=max(err,Float64(abs(a-b)/max(1,abs(a),abs(b))))
            end
        end
        err
    end
end
function rc_validate(input,m,method,criterion,c,progress)
    r=rc_prepare(input,m,method,criterion,c);K=length(r.log.state)
    points=Set((0,(K-1)÷2,K-1));checks=Any[]
    while r.status in (:ready,:running)
        if r.log.k in points
            err=rc_audit(r,m,criterion,c)
            push!(checks,Dict("step"=>r.log.k,"maximum_scaled_error"=>err,"passed"=>isfinite(err)&&err<=1e-7))
            rc_save(progress,Dict("stage"=>"validation","checks"=>checks))
            checks[end]["passed"] || return rc_snapshot(r),checks,false
        end
        rc_step!(r)
    end
    s=rc_snapshot(r);s,checks,s.complete && length(checks)==length(points)
end

# Leading loop arithmetic, reconstructed from the actual threshold path.
# FP retains all active tableau coordinates; all methods cache/evaluate only
# feasible comparisons. Packed reference dot products read H entries when
# 4*active >= H, otherwise the direct active-coordinate reduction reads active.
function rc_work(s,n,A,criterion)
    H=n-1;K=H*A;length(s.order)==K || error("Incomplete path")
    acts=vcat(0,fill(A,H));active=H;high=A>=2 ? H : 0
    prior=Set([(2,A)]);pairs=0;newpairs=0;retained=0;fp=0.0;fprefresh=0;fpexposed=0;fpdots=0;fpmaterialized=0
    for (k,(i,a)) in enumerate(s.order)
        (i,a) in prior || error("Non-threshold path")
        dim=active-Int(a==1)
        acts[i]-=1
        after=copy(prior);delete!(after,(i,a))
        for j in i:min(i+1,n)
            delete!(after,(j,acts[j]))
            acts[j]>0 && (j==2 || acts[j-1]<acts[j]) && push!(after,(j,acts[j]))
        end
        if k<K
            pairs+=length(after);retained+=length(intersect(prior,after));newpairs+=length(setdiff(after,prior))
            q=count(pair->pair[2]>=2 && pair in prior,after)
            ex=count(pair->pair[2]>=2 && !(pair in prior),after)
            dotdim=4active>=H ? H : active
            fp+=(a==1 ? 2.0*dim^2 : 4.0*active^2)+2.0*dotdim*q+4.0*dim*ex
            fprefresh+=q;fpexposed+=ex;fpdots+=dotdim*q;fpmaterialized+=dim*ex
        end
        active=dim
        prior=after
    end
    d=n+Int(criterion=="average");u=K-1
    Dict("updates"=>u,"post_update_frontier_pairs"=>pairs,"retained_pairs"=>retained,
        "new_pairs"=>newpairs,"FP_retained_comparisons"=>fprefresh,"FP_new_comparisons"=>fpexposed,
        "FP_reference_dot_elements"=>fpdots,"FP_materialization_elements"=>fpmaterialized,
        "C_leading_loop_work"=>(2/3)*u*d^3+4.0*n*pairs,
        "R_leading_loop_work"=>4.0*u*d^2+4.0*n*pairs,
        "M_leading_loop_work"=>4.0*u*d^2+2.0*d*retained+4.0*n*newpairs,
        "FP_leading_loop_work"=>fp,"BLOCK_order_proxy"=>Float64(H)*A^2)
end
function rc_check_work(s,w)
    k=length(s.order);alg=s.algorithm;c=s.counts
    c["assignments"]==k && c["terminal_numerical_updates_skipped"]==1 || return false
    alg in (:C,:R) && return c["direct_frontier_pairs"]==1+w["post_update_frontier_pairs"]
    alg==:M && return c["retained_pair_refreshes"]==w["retained_pairs"] && c["new_pair_exposures"]==1+w["new_pairs"]
    alg==:FP && return c["retained_refreshes"]==w["FP_retained_comparisons"] && c["new_comparisons"]==w["FP_new_comparisons"] &&
        c["reference_dot_elements"]==w["FP_reference_dot_elements"] &&
        c["materialization_elements"]==w["FP_materialization_elements"]
    alg==:BLOCK && return c["monotonicity_checks"]==k-1
    false
end
