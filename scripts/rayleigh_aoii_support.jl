"""Average-cost, capped, fully observed Rayleigh/AoII validation module.

No existing numerical/timing source is changed. State h has dense label h+1;
state zero admits only gear zero, even at negative charges. Exact routines use
Rational{BigInt}. They are validation algorithms, not machine-time benchmarks.
"""
module RayleighAoII
using LinearAlgebra
using ..MultiGearBandits
const Q = Rational{BigInt}
export AoIIModel, parse_q, from_success, dense_model, excursion, threshold_policies,
       exact_certificate, curvature_bound, exact_downshift, BlockWorkspace,
       block_frontier!, block_shift!, validation_case, load_spec, float_path_audit

function parse_q(x::AbstractString)
    occursin(r"^[+-]?[0-9]+(/[0-9]+)?$",x) || throw(ArgumentError("Use an exact integer/fraction string: $x"))
    parts=split(x,'/'); n=parse(BigInt,parts[1]); d=length(parts)==1 ? BigInt(1) : parse(BigInt,parts[2])
    d>0 || throw(ArgumentError("Denominator must be positive."))
    return n//d
end
parse_q(x::Integer)=BigInt(x)//BigInt(1)
parse_q(x::Rational)=BigInt(numerator(x))//BigInt(denominator(x))

struct AoIIModel{T<:Real}
    H::Int
    nsrc::Int
    wR::T
    wT::T
    chi::T
    success::Vector{T}
    p::Vector{T}
    c::Vector{T}
    kappa::T
    epsilon::T
    function AoIIModel(H::Integer,nsrc::Integer,wR::T,chi::T,s::Vector{T},k::T,e::T) where {T<:Real}
        H>=1 && H<=10000 || throw(ArgumentError("Cap must lie in 1:10000."))
        nsrc>=2 || throw(ArgumentError("At least two source values required."))
        length(s)>=2 || throw(ArgumentError("At least one active gear required."))
        all(isfinite,vcat(s,[wR,chi,k,e])) || throw(ArgumentError("Nonfinite primitive."))
        wt=(one(T)-wR)/T(nsrc-1)
        zero(T)<wt<wR<one(T) || throw(ArgumentError("Require 0<wT<wR<1."))
        chi>0 && k>0 && e>=0 || throw(ArgumentError("Require chi,kappa>0 and epsilon>=0."))
        s[1]==0 && all(0<s[a]<1 && s[a]>s[a-1] for a in 2:length(s)) ||
            throw(ArgumentError("Success menu must start at zero and increase strictly below one."))
        p=[wt+(wR-wt)*v for v in s]; x=p.-p[1]; c=k.*x.+e.*x.^2
        all(0<p[a]<1 for a in eachindex(p)) && all(c[a]>c[a-1] for a in 2:length(c)) ||
            throw(ArgumentError("Invalid reset/resource menu after conversion."))
        new{T}(Int(H),Int(nsrc),wR,wt,chi,copy(s),p,c,k,e)
    end
end
function from_success(H;nsrc=2,wR=3//4,success=[0//1,1//4,1//2,3//4],
                      kappa=2//1,epsilon=0//1,chi=1//1)
    AoIIModel(H,nsrc,parse_q(wR),parse_q(chi),Q[parse_q(v) for v in success],parse_q(kappa),parse_q(epsilon))
end
maxa(m::AoIIModel)=length(m.p)-1
function floating(m::AoIIModel)
    AoIIModel(m.H,m.nsrc,Float64(m.wR),Float64(m.chi),Float64.(m.success),Float64(m.kappa),Float64(m.epsilon))
end
function dense_model(m::AoIIModel;exact::Bool=true)
    T=exact ? Q : Float64; H=m.H; A=maxa(m); n=H+1
    P=zeros(T,n,n,A+1); cost=zeros(T,n,A+1); resource=zeros(T,n,A+1)
    for a in 0:A
        P[1,1,a+1]=T(m.wR);P[1,2,a+1]=one(T)-T(m.wR)
        for h in 1:H
            P[h+1,1,a+1]=T(m.p[a+1]);P[h+1,min(h+1,H)+1,a+1]=one(T)-T(m.p[a+1])
            cost[h+1,a+1]=T(h);resource[h+1,a+1]=T(m.c[a+1])
        end
    end
    mask=trues(n);mask[1]=false
    return exact ? ExactMultiGearModel(P,cost,resource;controllable=mask) :
                   FiniteMultiGearModel(P,cost,resource;controllable=mask)
end
function check_actions(m,actions;threshold=false)
    length(actions)==m.H && all(a isa Integer && 0<=a<=maxa(m) for a in actions) ||
        throw(ArgumentError("Expected one admissible gear per positive age."))
    !threshold || issorted(actions) || throw(ArgumentError("Policy is not nondecreasing in age."))
end
function excursion(m::AoIIModel{T},actions;costs=m.c) where {T}
    check_actions(m,actions);length(costs)==maxa(m)+1 || throw(DimensionMismatch("Resource menu."))
    H=m.H; t=zeros(T,H);w=zeros(T,H);e=zeros(T,H)
    a=actions[H]+1;t[H]=one(T)/m.p[a];w[H]=T(H)/m.p[a];e[H]=costs[a]/m.p[a]
    for h in (H-1):-1:1
        a=actions[h]+1;q=one(T)-m.p[a]
        t[h]=one(T)+q*t[h+1];w[h]=T(h)+q*w[h+1];e[h]=costs[a]+q*e[h+1]
    end
    u=one(T)-m.wR;den=one(T)+u*t[1];hb=u*w[1]/den;cb=u*e[1]/den
    (time=t,age=w,resource=e,hbar=hb,cbar=cb,phi=w.-hb.*t,gamma=e.-cb.*t)
end
function threshold_policies(H::Integer,A::Integer;limit::Integer=10000)
    H>=1 && A>=1 && limit>=1 || throw(ArgumentError("Positive dimensions/limit required."))
    count=binomial(BigInt(H+A),BigInt(A));count<=limit || throw(ArgumentError("Threshold-family limit exceeded: $count"))
    out=Vector{Int}[];v=zeros(Int,H)
    function visit(h,lo)
        if h>H;push!(out,copy(v));return;end
        for a in lo:A;v[h]=a;visit(h+1,a);end
    end
    visit(1,0);length(out)==count || error("Family enumeration count mismatch.")
    out
end
function scalar_frontier(m,actions,e)
    T=eltype(m.p);out=Tuple{Int,Int,T,T,T}[]
    for h in 1:m.H
        a=actions[h]
        if a>0 && (h==1 || actions[h-1]<a)
            s=min(h+1,m.H);dp=m.p[a+1]-m.p[a]
            f=dp*e.phi[s];g=m.c[a+1]-m.c[a]-dp*e.gamma[s]
            g>0 || error("Nonpositive required frontier resource at ($h,$a).")
            push!(out,(h,a,f,g,f/g))
        end
    end
    out
end

# One mutable statistic triple for each interior gear block. The cap is separate.
mutable struct BlockWorkspace{T<:Real}
    model::AoIIModel{T}
    lengths::Vector{Int}
    powers::Vector{T}
    sums::Vector{T}
    weighted::Vector{T}
    starts::Vector{Int}
    head::Matrix{T}
    terminal_gear::Int
    growth_updates::Int
    shrink_updates::Int
    block_visits::Int
    candidate_evaluations::Int
end
function grow!(w::BlockWorkspace,a)
    s=a+1;n=w.lengths[s];q=one(eltype(w.powers))-w.model.p[s]
    w.weighted[s]+=n*w.powers[s];w.sums[s]+=w.powers[s];w.powers[s]*=q
    w.lengths[s]+=1;w.growth_updates+=1
end
function shrink!(w::BlockWorkspace,a)
    s=a+1;n=w.lengths[s];n>0 || error("Cannot shrink empty block.")
    if n==1
        w.powers[s]=one(eltype(w.powers));w.sums[s]=zero(eltype(w.powers));w.weighted[s]=zero(eltype(w.powers))
    else
        oldpower=w.powers[s]/(one(eltype(w.powers))-w.model.p[s])
        w.powers[s]=oldpower;w.sums[s]-=oldpower;w.weighted[s]-=(n-1)*oldpower
    end
    w.lengths[s]-=1;w.shrink_updates+=1
end
function BlockWorkspace(m::AoIIModel{T}) where {T}
    A=maxa(m)
    w=BlockWorkspace(m,zeros(Int,A+1),ones(T,A+1),zeros(T,A+1),zeros(T,A+1),
        zeros(Int,A+1),zeros(T,3,A+1),A,0,0,0,0)
    for _ in 1:(m.H-1);grow!(w,A);end
    w
end
function block_frontier!(w::BlockWorkspace{T}) where {T}
    m=w.model;A=maxa(m);H=m.H
    sum(w.lengths)==H-1 || error("Interior length invariant failed.")
    all(w.lengths[a+1]==0 for a in (w.terminal_gear+1):A) || error("Nonthreshold cap gear.")
    pos=1
    for a in 0:A;w.starts[a+1]=pos;pos+=w.lengths[a+1];end
    s=w.terminal_gear+1;cap=(one(T)/m.p[s],T(H)/m.p[s],m.c[s]/m.p[s]);tail=cap
    for a in A:-1:0
        w.block_visits+=1;s=a+1
        if w.lengths[s]>0
            l=w.starts[s];L=w.sums[s];J=w.weighted[s];Qn=w.powers[s]
            tail=(L+Qn*tail[1],l*L+J+Qn*tail[2],m.c[s]*L+Qn*tail[3])
            for k in 1:3;w.head[k,s]=tail[k];end
        end
    end
    u=one(T)-m.wR;den=one(T)+u*tail[1];hb=u*tail[2]/den;cb=u*tail[3]/den
    out=Tuple{Int,Int,T,T,T}[]
    for a in 1:A
        s=a+1
        if w.lengths[s]>0
            h=w.starts[s];q=one(T)-m.p[s]
            nxt=((w.head[1,s]-one(T))/q,(w.head[2,s]-T(h))/q,(w.head[3,s]-m.c[s])/q)
        elseif w.terminal_gear==a
            h=H;nxt=cap
        else
            continue
        end
        dp=m.p[s]-m.p[s-1];f=dp*(nxt[2]-hb*nxt[1]);g=m.c[s]-m.c[s-1]-dp*(nxt[3]-cb*nxt[1])
        isfinite(f) && isfinite(g) && g>0 || error("Unresolved/nonpositive block frontier metric at ($h,$a).")
        push!(out,(h,a,f,g,f/g));w.candidate_evaluations+=1
    end
    (frontier=out,hbar=hb,cbar=cb)
end
function block_shift!(w::BlockWorkspace,h::Int,a::Int)
    m=w.model;1<=h<=m.H && 1<=a<=maxa(m) || throw(ArgumentError("Invalid downshift."))
    if h==m.H
        w.terminal_gear==a && w.lengths[a+1]==0 || error("Infeasible cap shift.")
        w.terminal_gear-=1
    else
        w.lengths[a+1]>0 && w.starts[a+1]==h || error("Shift not at block head.")
        shrink!(w,a);grow!(w,a-1)
    end
    nothing
end

"""Exact scalar or block TDS, with true lexicographic ties and no tolerance merging.
O(A)-size block workspace; O(AH) returned logs. Exact rational bit costs are NOT
covered by the real-arithmetic operation bound. Float64 live selection is not
exposed in this validation milestone; float_path_audit replays an exact path.
"""
function exact_downshift(m::AoIIModel{Q};method::Symbol=:block)
    method in (:scalar,:block) || throw(ArgumentError("Unknown evaluator."))
    H=m.H;A=maxa(m);actions=method==:scalar ? fill(A,H) : nothing
    mpi=Matrix{Q}(undef,H,A);order=Tuple{Int,Int}[]
    fs=Q[];gs=Q[];vals=Q[];rates_h=Q[];rates_c=Q[];w=method==:block ? BlockWorkspace(m) : nothing
    for _ in 1:(A*H)
        if method==:block
            e=block_frontier!(w);front=e.frontier
        else
            e=excursion(m,actions);front=scalar_frontier(m,actions,e)
        end
        isempty(front) && error("Empty frontier before all assignments.")
        chosen=first(front)
        for row in front
            if isless((row[5],row[1],row[2]),(chosen[5],chosen[1],chosen[2]));chosen=row;end
        end
        h,a,f,g,v=chosen
        method==:scalar && actions[h]!=a && error("Policy/log mismatch.")
        push!(order,(h,a));push!(fs,f);push!(gs,g);push!(vals,v);push!(rates_h,e.hbar);push!(rates_c,e.cbar)
        mpi[h,a]=v
        if method==:block;block_shift!(w,h,a);else;actions[h]-=1;end
    end
    if method==:block
        w.terminal_gear==0 && w.lengths[1]==H-1 && all(iszero,w.lengths[2:end]) || error("Incomplete exact block path.")
    else
        all(iszero,actions) || error("Incomplete exact scalar path.")
    end
    counts=method==:block ? Dict("block_visits"=>w.block_visits,"candidate_evaluations"=>w.candidate_evaluations,
        "growth_updates"=>w.growth_updates,"shrink_updates"=>w.shrink_updates) : Dict("full_policy_evaluations"=>A*H)
    (mpi=mpi,order=order,f=fs,g=gs,assignments=vals,hbar=rates_h,cbar=rates_c,
     monotone=issorted(vals),counts=counts)
end

"""Enumerate the whole threshold family; verify positivity and sufficient urgency
inequalities in exact arithmetic. This is finite-instance certification, not an
asymptotically inexpensive analytic test. Returns failures instead of discarding them.
"""
function exact_certificate(m::AoIIModel{Q};limit::Int=10000)
    H=m.H;A=maxa(m);r=[(m.c[a+1]-m.c[a])/(m.p[a+1]-m.p[a]) for a in 1:A]
    gamma_bound=maximum(m.c./m.p);failures=String[];ming=nothing;minurg=nothing
    !issorted(r) && push!(failures,"unordered resource/reset ratios")
    !(r[1]>gamma_bound) && push!(failures,"primitive resource-bias bound not satisfied")
    ng=0;nu=0;cap_equalities=0;policies=threshold_policies(H,A;limit=limit)
    for acts in policies
        e=excursion(m,acts)
        all(e.phi.>0) || push!(failures,"nonpositive holding-cost bias")
        for h in 1:H,a in 1:A
            s=min(h+1,H);g=m.c[a+1]-m.c[a]-(m.p[a+1]-m.p[a])*e.gamma[s]
            ng+=1;ming=ming===nothing ? g : min(ming,g)
            g>0 || push!(failures,"nonpositive adjacent resource")
        end
        for h in 1:(H-1)
            a=acts[h]
            if a>0 && acts[h+1]==a
                s=min(h+1,H);v=min(h+2,H)
                margin=e.phi[v]*(r[a]-e.gamma[s])-e.phi[s]*(r[a]-e.gamma[v]);nu+=1
                margin>=0 || push!(failures,"within-block urgency violation")
                if s==v
                    margin==0 || error("Structural cap equality lost.");cap_equalities+=1
                else
                    minurg=minurg===nothing ? margin : min(minurg,margin)
                end
            end
        end
    end
    Dict{String,Any}("passed"=>isempty(failures),"threshold_policies"=>length(policies),"resource_checks"=>ng,
        "urgency_checks"=>nu,"structural_cap_equalities"=>cap_equalities,"minimum_g"=>string(ming),
        "minimum_noncap_urgency_crossproduct"=>(minurg===nothing ? "not_applicable" : string(minurg)),
        "r1_minus_Gamma"=>string(r[1]-gamma_bound),"failures"=>unique(failures))
end

"""Construct an exact open upper curvature bound from ALL threshold policies.
For each affine margin z0+epsilon*z1, require z0>0 (or an identically zero cap
equality). Restrictive slopes give epsilon<z0/(-z1). Select a prespecified dyadic
value <=min(kappa/16,bound/2); never search on timing or index-output quality.
"""
function curvature_bound(base::AoIIModel{Q};limit::Int=10000)
    base.epsilon==0 || throw(ArgumentError("Curvature bound requires baseline epsilon=0."))
    H=base.H;A=maxa(base);k=base.kappa;x=base.p.-base.p[1];bound=nothing;constraints=0;zeros_count=0
    function constrain(z0,z1)
        constraints+=1
        if z0==0
            z1==0 || error("Nonstructural zero baseline margin.");zeros_count+=1
        elseif z0>0
            if z1<0;b=z0/(-z1);bound=bound===nothing ? b : min(bound,b);end
        else
            error("Negative baseline margin.")
        end
    end
    for a in 0:A;constrain(k-k*x[a+1]/base.p[a+1],x[2]-x[a+1]^2/base.p[a+1]);end
    policies=threshold_policies(H,A;limit=limit)
    for acts in policies
        e0=excursion(base,acts);e2=excursion(base,acts;costs=x.^2)
        for h in 1:H,a in 1:A
            s=min(h+1,H);constrain(k-e0.gamma[s],x[a]+x[a+1]-e2.gamma[s])
        end
        for h in 1:(H-1)
            a=acts[h]
            if a>0 && a==acts[h+1]
                s=min(h+1,H);v=min(h+2,H);r2=x[a]+x[a+1]
                constrain(e0.phi[v]*(k-e0.gamma[s])-e0.phi[s]*(k-e0.gamma[v]),
                          e0.phi[v]*(r2-e2.gamma[s])-e0.phi[s]*(r2-e2.gamma[v]))
            end
        end
    end
    target=bound===nothing ? k/16 : min(k/16,bound/2);selected=one(Q)
    while selected>target;selected/=2;end
    (upper=bound,epsilon=selected,threshold_policies=length(policies),constraints=constraints,structural_zeros=zeros_count)
end

scaled(x,y)=abs(Float64(x)-Float64(y))/(1+max(abs(Float64(x)),abs(Float64(y))))
function float_path_audit(m::AoIIModel{Q},exact;tol=1e-9)
    w=BlockWorkspace(floating(m));actions=fill(maxa(m),m.H);err=0.0
    for (k,(h,a)) in enumerate(exact.order)
        b=block_frontier!(w);q=excursion(m,actions);front=scalar_frontier(m,actions,q)
        length(b.frontier)==length(front) || error("Float frontier size mismatch.")
        for (f,e) in zip(b.frontier,front)
            f[1:2]==e[1:2] || error("Float frontier identity mismatch.")
            for j in 3:5;err=max(err,scaled(f[j],e[j]));end
        end
        err=max(err,scaled(b.hbar,q.hbar),scaled(b.cbar,q.cbar));block_shift!(w,h,a);actions[h]-=1
    end
    Dict("passed"=>err<=tol,"maximum_scaled_error"=>err,"tolerance"=>tol,
         "scope"=>"Float64 metrics replayed along exact choices; not independent Float64 selection or timing")
end
function load_spec(spec::AbstractDict)
    base=from_success(Int(spec["H"]);nsrc=Int(spec["source_states"]),wR=spec["wR"],success=spec["success"],
        kappa=spec["kappa"],chi=spec["chi"],epsilon="0")
    rule=spec["curvature_rule"];proof=curvature_bound(base)
    epsilon=rule=="zero" ? zero(Q) : rule=="certified_dyadic" ? proof.epsilon :
        throw(ArgumentError("Unknown curvature rule."))
    m=from_success(base.H;nsrc=base.nsrc,wR=base.wR,success=base.success,kappa=base.kappa,chi=base.chi,epsilon=epsilon)
    m,proof
end
function validation_case(spec;max_policies::Int=1000,oracle_seconds::Real=180.0,check_generic::Bool=true)
    m,proof=load_spec(spec);H=m.H;A=maxa(m);cert=exact_certificate(m)
    cert["passed"] || error("Exact sufficient-condition certificate failed: $(cert["failures"])")
    scalar=exact_downshift(m;method=:scalar);block=exact_downshift(m;method=:block)
    for field in (:mpi,:order,:f,:g,:assignments,:hbar,:cbar)
        getproperty(scalar,field)==getproperty(block,field) || error("Exact block/scalar mismatch: $field")
    end
    block.monotone || error("Exact assignments are not monotone.")
    reference=enumerate_bellman(dense_model(m);criterion=:average,max_states=8,
        max_policies=max_policies,time_limit_s=oracle_seconds)
    reference.complete && reference.coverage==:exact_pass && reference.sign_indexability==:exact_pass ||
        error("Independent full-policy oracle incomplete/failed: $(reference.status), $(reference.sign_indexability)")
    reference.dai==block.mpi || error("Structured MPI differs from independent DAI.")
    # Verify excursion formulas for EVERY admissible policy, including nonthreshold policies.
    for record in reference.records
        e=excursion(m,record.actions[2:end])
        e.hbar==record.cost_rate && e.cbar==record.resource_rate &&
        vcat(zero(Q),e.phi)==record.cost_vector && vcat(zero(Q),e.gamma)==record.resource_vector ||
            error("Excursion/independent stationarity-Poisson mismatch.")
    end
    generic=Dict{String,Any}[]
    if check_generic
        fm=dense_model(m;exact=false);qm=ExactMultiGearModel(fm);em=reference.model
        qm.P==em.P && qm.h==em.h && qm.c==em.c || error("Float64 copy is not the same exact dyadic model.")
        family=OrderedThresholdFamily(collect(2:(H+1)))
        expected=[(h+1,a) for (h,a) in block.order]
        methods=(("C",direct_downshift),("R",rankone_downshift),("M",cached_downshift),("FP",reduced_fp_downshift))
        for (name,fn) in methods
            run=name=="C" ? fn(fm;criterion=:average,family=family) : fn(fm;criterion=:average,family=family,audit=true)
            run.complete || error("Generic $name incomplete: $(run.status)")
            order=name=="FP" ? run.order : [s.selected for s in run.steps]
            order==expected || error("Generic $name differs at an exact tie/selection.")
            v=validate_downshift(run,reference)
            v.execution==:exact_pass && v.bellman_path==:exact_pass && v.family_wide_positivity==:exact_pass &&
            v.mpi_dai_agreement==:numerical_pass || error("Generic $name full-charge validation failed.")
            push!(generic,Dict("algorithm"=>name,"passed"=>true,"status"=>string(run.status),
                "max_scaled_index_error"=>v.max_scaled_index_error,"bellman_path"=>string(v.bellman_path),
                "exact_comparison_solves"=>(name=="FP" ? run.workspace.counts.exact_comparison_solves : run.exact_comparison_solves)))
        end
    end
    floats=float_path_audit(m,block);floats["passed"] || error("Float64 path-metric check failed.")
    intervals=Dict{String,Any}[]
    for r in reference.records
        r.interval.empty && continue
        push!(intervals,Dict("positive_age_gears"=>r.actions[2:end],
            "lower"=>(r.interval.lower===nothing ? "-Inf" : string(r.interval.lower)),
            "upper"=>(r.interval.upper===nothing ? "+Inf" : string(r.interval.upper)),
            "holding_cost_rate"=>string(r.cost_rate),"resource_rate"=>string(r.resource_rate)))
    end
    Dict{String,Any}("id"=>spec["id"],"passed"=>true,"H"=>H,"states_including_zero"=>H+1,"A"=>A,
        "criterion"=>"average","resource_interpretation"=>"constructed synchronization-gain resource; NOT calibrated battery energy",
        "wR"=>string(m.wR),"wT"=>string(m.wT),"source_states"=>m.nsrc,"chi"=>string(m.chi),
        "success_exact"=>string.(m.success),"reset_exact"=>string.(m.p),"resource_exact"=>string.(m.c),
        "power_definition"=>"P^0=0; P^a=-chi/log(s^a) exactly; displayed decimal powers are approximations",
        "power_approx"=>vcat(0.0,[-Float64(m.chi)/log(Float64(s)) for s in m.success[2:end]]),
        "kappa"=>string(m.kappa),"epsilon"=>string(m.epsilon),
        "curvature_open_upper"=>(proof.upper===nothing ? "+Inf" : string(proof.upper)),
        "curvature_constraints"=>proof.constraints,"certificate"=>cert,
        "all_policy_oracle_policies"=>length(reference.records),"all_real_charge_coverage"=>string(reference.coverage),
        "sign_indexability"=>string(reference.sign_indexability),"dai_verified"=>true,
        "strict_adjacent_DAI_order_at_all_ages"=>all(block.mpi[h,a]>block.mpi[h,a+1] for h in 1:H for a in 1:(A-1)),
        "dai_exact_by_age"=>[string.(block.mpi[h,:]) for h in 1:H],
        "shift_log"=>[Dict("age"=>p[1],"upper_gear"=>p[2],"index_exact"=>string(block.assignments[k]),
            "f_exact"=>string(block.f[k]),"g_exact"=>string(block.g[k])) for (k,p) in enumerate(block.order)],
        "block_counts"=>block.counts,"float_path_audit"=>floats,"generic_checks"=>generic,
        "optimal_policy_intervals"=>intervals,
        "scope"=>"exact small rational models; average criterion only; no throughput/timing/allocation performance claim")
end
end # module
