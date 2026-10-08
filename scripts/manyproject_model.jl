"""Certified exact single-project tables and proportionally scaled dual data.
No joint-state enumeration. All data needed by an online allocation are compiled
once per fixed cap/primitive/load family. Exact model/certificate work is outside
simulation/online-allocation diagnostics.
"""
module ManyProjectModel
using ..RayleighAoII, ..RayleighBlock, ..RayleighTiming, ..RayleighAllocation
using ..ManyProjectAllocator
const RA=RayleighAoII;const RT=RayleighTiming;const AL=RayleighAllocation;const MA=ManyProjectAllocator
const Q=Rational{BigInt}
qs(x::Q)=AL.qs(x)
parseq(x)=AL.parseq(x)
const REVISION="rayleigh-manyproject-preflight-v1"
struct Family
    id::String
    source_group::String
    single::AL.SingleData
    weights::Vector{Q}
    base_counts::Vector{Int}
    budget_per_project::Q
    lambda::Q
    bound_per_project::Q
    bias::Vector{Vector{Q}}
    gains::Vector{Q}
    residual::Array{Q,3}       # class, state+1, gear+1
    nextbias::Array{Q,3}
    occupations::Vector{Vector{Q}}
    scores::Dict{Symbol,MA.AbstractScores}
    priority_rank::Matrix{Int}
    priority_positive::BitMatrix
    units::Vector{Int}
    tick::Q
    record::Dict{String,Any}
end
function exact_single(spec)
    H=Int(spec["H"]);1<=H<=64 || error("This first exact-score many-project stage admits caps 1--64; larger-cap preparation is a later gate.")
    m=RA.from_success(H;nsrc=Int(spec["source_states"]),wR=spec["wR"],success=spec["success"],
        kappa=spec["kappa"],epsilon=spec["epsilon"],chi="1")
    A=length(m.p)-1;1<=A<=3 || error("Only the saved 1--3 active-gear validation menus are supported.")
    slopes=Q[(m.c[a+1]-m.c[a])/(m.p[a+1]-m.p[a]) for a in 1:A]
    issorted(slopes) && slopes[1]>maximum(m.c./m.p) || error("All-policy marginal-resource primitive test failed.")
    all(Q(Float64(p))==p && denominator(p)<=BigInt(2)^53 for p in vcat([m.wR],m.p)) || error("Transition sampler requires exact 53-bit dyadic probabilities; no rounding accepted.")
    exact=RA.exact_downshift(m;method=:scalar)
    exact.monotone || error("Exact candidate roots decrease; no DAI or simulation admitted.")
    bellman=RT.path_bellman_certificate(m,exact)
    actions=fill(A,H);policies=Vector{Int}[];hb=Q[];cb=Q[];phi=Vector{Q}[];gamma=Vector{Q}[]
    for k in 0:A*H
        e=RA.excursion(m,actions);push!(policies,copy(actions));push!(hb,e.hbar);push!(cb,e.cbar)
        push!(phi,copy(e.phi));push!(gamma,copy(e.gamma))
        if k<A*H;h,a=exact.order[k+1];actions[h]==a || error("Candidate path is not adjacent.");actions[h]-=1;end
    end
    all(iszero,actions) || error("Incomplete exact index path.")
    record=Dict{String,Any}("H"=>H,"A"=>A,"reset_probabilities"=>qs.(m.p),"resource_menu"=>qs.(m.c),
        "source_stay_probability"=>qs(m.wR),"indices"=>[qs.(exact.mpi[h,:]) for h in 1:H],
        "roots"=>qs.(exact.assignments),"order"=>[collect(x) for x in exact.order],
        "path_policies"=>policies,"path_age_rates"=>qs.(hb),"path_resource_rates"=>qs.(cb),
        "bellman_certificate"=>bellman,"primitive_resource_margin"=>qs(slopes[1]-maximum(m.c./m.p)),
        "dai_verified"=>true,"scope"=>"exact fixed-instance whole-charge optimal chain; NOT whole-family PCLI2 enumeration or a uniform-in-cap theorem")
    AL.SingleData(m,exact.mpi,copy(exact.assignments),copy(exact.order),policies,hb,cb,phi,gamma,record)
end
function class_dual(d,weights::Vector{Q},counts::Vector{Int},b::Q)
    length(weights)==length(counts)>0 && all(weights.>0) && all(counts.>0) && b>=0 || error("Invalid class mixture/budget.")
    base=sum(counts);B=base*b
    points=sort!(unique!(vcat(Q[0],[w*x for w in weights for x in d.roots])))
    vals=Q[sum(n*minimum(w*d.hbar[k]+lam*d.cbar[k] for k in eachindex(d.hbar)) for (w,n) in zip(weights,counts))-lam*B for lam in points]
    best=maximum(vals);lam=points[findfirst(==(best),vals)]
    right=[1+count(x->w*x<=lam,d.roots) for w in weights];left=[1+count(x->w*x<lam,d.roots) for w in weights]
    low=sum(n*d.cbar[k] for (n,k) in zip(counts,right));high=sum(n*d.cbar[k] for (n,k) in zip(counts,left))
    low<=B && (lam==0 || high>=B) || error("Dual resource subgradient bracket failed.")
    theta=lam==0 || high==low ? zero(Q) : (B-low)/(high-low)
    0<=theta<=1 || error("Invalid relaxation mixture.")
    rresource=(1-theta)*low+theta*high
    rcost=sum(n*w*((1-theta)*d.hbar[r]+theta*d.hbar[l]) for (n,w,r,l) in zip(counts,weights,right,left))
    rcost==best && rresource<=B && lam*(B-rresource)==0 || error("Relaxation primal/dual equality failed.")
    m=d.model;K=length(weights);A=length(m.p)-1
    bias=Vector{Q}[];gains=Q[];occ=Vector{Q}[];checks=0
    residual=zeros(Q,K,m.H+1,A+1);nextbias=similar(residual);fill!(nextbias,0)
    for (k,(w,r,l)) in enumerate(zip(weights,right,left))
        v=vcat(zero(Q),w.*d.phi[r].+lam.*d.gamma[r]);g=w*d.hbar[r]+lam*d.cbar[r]
        v==vcat(zero(Q),w.*d.phi[l].+lam.*d.gamma[l]) || error("Bias differs at a tied dual charge.")
        push!(bias,v);push!(gains,g)
        pl=AL.single_stationary(m,d.policies[r]);ph=AL.single_stationary(m,d.policies[l])
        mixed=(1-theta).*pl.+theta.*ph;sum(mixed)==1 && all(mixed.>=0) || error("Relaxed initial distribution invalid.")
        push!(occ,mixed)
        nextbias[k,1,1]=(1-m.wR)*v[2];residual[k,1,1]=nextbias[k,1,1]-v[1]-g
        residual[k,1,1]==0 || error("Zero-state Bellman equality.")
        for h in 1:m.H,a in 0:A
            en=(1-m.p[a+1])*v[min(h+1,m.H)+1]
            nextbias[k,h+1,a+1]=en
            residual[k,h+1,a+1]=w*h+lam*m.c[a+1]+en-v[h+1]-g
            residual[k,h+1,a+1]>=0 || error("Negative exact charged Bellman slack.")
            if a==d.policies[r][h];residual[k,h+1,a+1]==0 || error("Chosen Bellman row not equal.");end
            checks+=1
        end
    end
    # Every point's objective scales identically, including exact maximizing ties.
    for scale in (1,2,4,8,16,32)
        scaled=Q[sum(scale*n*minimum(w*d.hbar[k]+p*d.cbar[k] for k in eachindex(d.hbar)) for (w,n) in zip(weights,counts))-p*(scale*B) for p in points]
        scaled==scale.*vals && points[findfirst(==(maximum(scaled)),scaled)]==lam || error("Proportional dual invariance failed.")
    end
    record=Dict{String,Any}("lambda_star"=>qs(lam),"bound_per_project"=>qs(best/base),"budget_per_project"=>qs(b),
        "base_population"=>base,"base_class_counts"=>counts,"class_weights"=>qs.(weights),"charge_points"=>qs.(points),
        "base_dual_values"=>qs.(vals),"right_policy_indices"=>right,"left_policy_indices"=>left,
        "relaxed_mixture_weight"=>qs(theta),"relaxed_cost"=>qs(rcost),"relaxed_resource"=>qs(rresource),
        "relaxed_primal_equals_dual"=>true,"proportional_scaling_verified"=>true,"all_action_Bellman_checks"=>checks,
        "class_biases"=>[qs.(v) for v in bias],"class_gains"=>qs.(gains),"relaxed_class_occupancies"=>[qs.(v) for v in occ],
        "charge_convention"=>"same smallest nonnegative maximizing charge at every proportional scale; right policy processes all exact ties")
    (lambda=lam,ell=best/base,bias=bias,gains=gains,residual=residual,nextbias=nextbias,occupations=occ,record=record)
end
function score_table(d,D,weights,policy;max_population=1600)
    m=d.model;A=length(m.p)-1;K=length(weights);v=zeros(Q,K*m.H,A+1)
    for k in 1:K,h in 1:m.H
        row=(k-1)*m.H+h;s=min(h+1,m.H)
        for a in 1:A
            if policy==:WHITTLE_K
                v[row,a+1]=v[row,a]+(m.c[a+1]-m.c[a])*weights[k]*d.mpi[h,a]
            elseif policy in (:LAGRANGIAN_K,:LAGRANGIAN_NI,:LAGRANGIAN_FC)
                v[row,a+1]=(m.p[a+1]-m.p[1])*D.bias[k][s+1]-D.lambda*m.c[a+1]
            elseif policy==:MYOPIC_K
                v[row,a+1]=weights[k]*s*(m.p[a+1]-m.p[1])
            else;error("Unknown score policy.")
            end
        end
    end
    MA.compile_scores(v;max_population=max_population)
end
function prepare_family(spec;max_population=1600)
    d=exact_single(spec);m=d.model;K=length(spec["weights"]);weights=parseq.(spec["weights"]);counts=Int.(spec["base_counts"])
    length(counts)==K && sum(counts)==50 || error("Preserve the declared base population and class proportions.")
    b=haskey(spec,"budget_per_project") ? parseq(spec["budget_per_project"]) : parseq(spec["load_factor"])*d.cbar[1]
    D=class_dual(d,weights,counts,b)
    tables=Dict{Symbol,MA.AbstractScores}()
    for name in (:WHITTLE_K,:LAGRANGIAN_K,:MYOPIC_K);tables[name]=score_table(d,D,weights,name;max_population=max_population);end
    tables[:LAGRANGIAN_NI]=tables[:LAGRANGIAN_K];tables[:LAGRANGIAN_FC]=tables[:LAGRANGIAN_K]
    A=length(m.p)-1;values=Q[weights[k]*d.mpi[h,a] for k in 1:K for h in 1:m.H for a in 1:A]
    all(>(0),values) || error("AoII DAI must be positive in this verified family.")
    vals=sort!(unique!(copy(values)));idx=Dict(v=>i for (i,v) in enumerate(vals))
    rank=zeros(Int,K*m.H,A);positive=trues(K*m.H,A)
    for k in 1:K,h in 1:m.H,a in 1:A;rank[(k-1)*m.H+h,a]=idx[weights[k]*d.mpi[h,a]];end
    units,tick=MA.resource_encoding(m.c)
    enc=Dict{String,Any}()
    for p in (:WHITTLE_K,:LAGRANGIAN_K,:MYOPIC_K)
        t=tables[p];enc[String(p)]=Dict("integer_type"=>t.encoding,"denominator"=>string(t.denominator),
            "max_population"=>t.max_population,"unique_score_ranks"=>maximum(t.rank),
            "scope"=>"lossless score encoding; input-specific strict common order checked before each fast allocation; exact DP otherwise")
    end
    record=Dict{String,Any}("id"=>spec["id"],"specification"=>Dict(spec),"single_project"=>d.evidence,"dual"=>D.record,
        "integer_resource_units"=>units,"resource_quantum"=>qs(tick),"score_encodings"=>enc,
        "joint_state_enumeration"=>false,"exact_online_objective"=>true,"offline_tables_reused_across_scales"=>true)
    Family(String(spec["id"]),String(spec["source_group"]),d,weights,counts,b,D.lambda,D.ell,D.bias,D.gains,D.residual,D.nextbias,
        D.occupations,tables,rank,positive,units,tick,record)
end
function class_labels(f::Family,L::Int)
    base=sum(f.base_counts);L>0 && L%base==0 || error("Population must be an exact positive multiple of the base composition.")
    vcat([fill(k,n*div(L,base)) for (k,n) in enumerate(f.base_counts)]...)
end
mutable struct Allocators
    work::Dict{Symbol,Any}
    priority_actions::Vector{Int}
    heap::Vector{NTuple{3,Int}}
    max_spend::Dict{Int,Int}
    limits::MA.Limits
end
function Allocators(f::Family,L::Int;limits=MA.Limits())
    w=Dict{Symbol,Any}(p=>MA.Workspace(f.scores[p],L) for p in MA.POLICIES if p!=:WHITTLE_P)
    Allocators(w,zeros(Int,L),NTuple{3,Int}[],Dict{Int,Int}(),limits)
end
function decide!(w::Allocators,f::Family,policy::Symbol,types::Vector{Int},C::Int)
    if policy==:WHITTLE_P
        r=MA.priority!(w.priority_actions,w.heap,types,f.priority_rank,f.priority_positive,f.units,C)
        return w.priority_actions,r
    end
    domain=policy==:LAGRANGIAN_NI ? :ni : (policy==:LAGRANGIAN_FC ? :fc : :k)
    r=MA.allocate!(w.work[policy],f.scores[policy],types,f.units,C,domain;limits=w.limits)
    w.work[policy].actions,r
end
function capacity_diagnostics(w::Allocators,f::Family,types,actions,C,spent)
    eligible=count(>(0),types);maximum=get!(w.max_spend,eligible) do
        MA.maximum_spend(f.units,eligible,C)
    end
    idle=any(types[i]>0 && actions[i]<length(f.units)-1 && C-spent>=f.units[actions[i]+2]-f.units[actions[i]+1] for i in eachindex(types))
    spent<=maximum<=C || error("Attainable-spend diagnostic contradiction.")
    (nonmaximal=idle,belowmax=spent<maximum,attainable_gap_units=maximum-spent,maximum_units=maximum)
end
end # module
