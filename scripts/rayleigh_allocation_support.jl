"""Exact small-system hard-budget allocation acceptance, NOT a scalable scheduler.
All rational policy scores/weights and budgets are preserved exactly. Existing
single-project kernels/certificates are reused without edits. The six heuristics
use the same hard-feasible complete gear vectors, with explicit NI/FC subdomains. No capacity relaxation/repair
or state-dependent recalibration of the average Lagrange multiplier is implicit.
"""
module RayleighAllocation
using LinearAlgebra, Random, Statistics, TOML, Dates, SHA
using ..MultiGearBandits
using ..RayleighAoII
using ..RayleighBlock
using ..RayleighTiming
using ..RayleighLargeCap
const Q=Rational{BigInt}
const RA=RayleighAoII
const RT=RayleighTiming
const RB=RayleighBlock
const LC=RayleighLargeCap
const REVISION="rayleigh-allocation-capacity-preflight-v2"
const LAGRANGIAN_POLICIES=(:LAGRANGIAN_K,:LAGRANGIAN_NI,:LAGRANGIAN_FC)
const POLICIES=(:WHITTLE_K,:WHITTLE_P,:LAGRANGIAN_K,:LAGRANGIAN_NI,:LAGRANGIAN_FC,:MYOPIC_K)
const ALL_POLICIES=(POLICIES...,:OPTIMAL)
const IDS=["binary_unit_H2","anchor_H3_A2_heterogeneous","three_source_H3_A2_homogeneous",
    "three_source_H3_A2_heterogeneous","three_source_H5_A2_heterogeneous",
    "three_source_H3_A3_heterogeneous","persistent_H3_A2","persistent_H5_A2",
    "zero_budget_H2_A2","slack_budget_H2_A2","capacity_tradeoff_H1_A2"]
parseq(x)=parse_q(x isa AbstractString ? replace(String(x),"//"=>"/") : x)
qs(x::Q)=string(numerator(x),"/",denominator(x))
lexless(a,b)=isless(Tuple(a),Tuple(b))
function require0(test,msg)
    test || error("Allocation validation: "*msg)
end
function configuration(path)
    c=TOML.parsefile(path)
    c["revision"]==REVISION && c["criterion"]=="average" || error("Unknown allocation protocol.")
    c["resource_scope"]=="constructed_sync_gain_not_battery_calibration" || error("Resource interpretation changed.")
    Tuple(Symbol.(c["policies"]))==POLICIES || error("Required policies/order changed.")
    c["expected_systems"]==11 && [s["id"] for s in c["system"]]==IDS || error("Use the eleven-system acceptance grid only.")
    c["max_joint_states"]==40 && c["max_joint_actions"]==64 && c["max_policy_iterations"]==128 &&
    c["max_knapsack_states"]==10000 && c["max_family_policies"]==10000 || error("Acceptance limits changed.")
    c["seconds_per_system"]==180.0 && c["master_seed"]==20260923 || error("Saved budget/seed changed.")
    t=c["simulation"]
    t["replications"]==16 && t["burn_in"]==2000 && t["horizon"]==10000 &&
      t["initial_state"]=="all_zero" && t["common_random_numbers"] && t["include_optimal"] || error("Simulation-validation plan changed.")
    for s in c["system"]
        1<=s["H"]<=5 && 2<=length(s["success"])<=4 && length(s["weights"])==2 || error("This stage is a tiny two-project validation, not production.")
        all(parseq(x)>0 for x in s["weights"]) && parseq(s["budget_maxgear_multiple"])>=0 || error("Invalid weights/budget.")
        occursin(r"^[A-Za-z0-9_-]+$",s["id"]) || error("Unsafe case ID.")
    end
    c
end

function solve_exact(A::Matrix{Q},b::Vector{Q})
    n=length(b);size(A)==(n,n) && n>0 || throw(DimensionMismatch("Square exact system required."))
    M=hcat(copy(A),copy(b))
    for c in 1:n
        p=findfirst(i->M[i,c]!=0,c:n)
        p===nothing && error("Singular exact system.")
        pivot=c+p-1
        if pivot!=c
            for j in c:n+1;M[c,j],M[pivot,j]=M[pivot,j],M[c,j];end
        end
        scale=M[c,c]
        for j in c:n+1;M[c,j]/=scale;end
        for i in c+1:n
            z=M[i,c]
            z==0 && continue
            for j in c:n+1;M[i,j]-=z*M[c,j];end
        end
    end
    x=zeros(Q,n)
    for i in n:-1:1
        x[i]=M[i,n+1]-sum((M[i,j]*x[j] for j in i+1:n);init=zero(Q))
    end
    A*x==b || error("Exact linear-solve residual.")
    x
end

struct SingleData
    model::AoIIModel{Q}
    mpi::Matrix{Q}
    roots::Vector{Q}
    order::Vector{Tuple{Int,Int}}
    policies::Vector{Vector{Int}}
    hbar::Vector{Q}
    cbar::Vector{Q}
    phi::Vector{Vector{Q}}
    gamma::Vector{Vector{Q}}
    evidence::Dict{String,Any}
end

function single_data(spec;family_limit=10000)
    m=from_success(Int(spec["H"]);nsrc=Int(spec["source_states"]),wR=spec["wR"],
        success=spec["success"],kappa=spec["kappa"],epsilon=spec["epsilon"],chi="1")
    family=exact_certificate(m;limit=family_limit)
    require0(family["passed"],"whole-family PCL certificate")
    start=time_ns();live=numerical_downshift(m);native_seconds=(time_ns()-start)/1e9
    exact,err=RB.check_live_against_exact(m,live)
    bellman=RT.path_bellman_certificate(m,exact)
    A=length(m.p)-1;actions=fill(A,m.H);policies=Vector{Int}[]
    hb=Q[];cb=Q[];phi=Vector{Q}[];gamma=Vector{Q}[]
    for k in 0:length(exact.order)
        e=excursion(m,actions);push!(policies,copy(actions));push!(hb,e.hbar);push!(cb,e.cbar)
        push!(phi,copy(e.phi));push!(gamma,copy(e.gamma))
        if k<length(exact.order);h,a=exact.order[k+1];require0(actions[h]==a,"single path state");actions[h]-=1;end
    end
    require0(all(iszero,actions),"single path endpoint")
    # All simulation Bernoulli probabilities are exactly dyadic, not silently rounded.
    require0(all(Q(Float64(x))==x for x in vcat(m.p,m.c,[m.wR,1-m.wR])),"exact representability of frozen primitives")
    record=Dict{String,Any}("model"=>RT.model_record(m),"whole_family_certificate"=>family,
        "bellman_certificate"=>bellman,"dai_verified"=>true,"native_index_construction_seconds_diagnostic"=>native_seconds,
        "native_scaled_error"=>err,"native_counts"=>live.counts,"exact_indices"=>[qs.(exact.mpi[h,:]) for h in 1:m.H],
        "exact_order"=>[collect(x) for x in exact.order],"exact_roots"=>qs.(exact.assignments),
        "path_policies"=>policies,"path_age_rates"=>qs.(hb),"path_resource_rates"=>qs.(cb),
        "scope"=>"exact small-model tables validated against accepted numerical BLOCK; not large-system allocation timing")
    SingleData(m,exact.mpi,copy(exact.assignments),copy(exact.order),policies,hb,cb,phi,gamma,record)
end

function single_stationary(m,actions)
    e=excursion(m,actions);u=1-m.wR
    pi=zeros(Q,m.H+1);pi[1]=1/(1+u*e.time[1]);x=pi[1]*u
    for h in 1:m.H
        if h==m.H;pi[h+1]=x/m.p[actions[h]+1]
        else;pi[h+1]=x;x*=1-m.p[actions[h]+1];end
    end
    require0(sum(pi)==1 && all(pi.>0),"single-policy stationary normalization/positivity")
    require0(sum(Q(h)*pi[h+1] for h in 1:m.H)==e.hbar &&
      sum(m.c[actions[h]+1]*pi[h+1] for h in 1:m.H)==e.cbar,"single-policy rates")
    # Direct flow equations; this is independent of the formula used to obtain pi.
    flows=zeros(Q,m.H+1)
    flows[1]+=pi[1]*m.wR;flows[2]+=pi[1]*(1-m.wR)
    for h in 1:m.H
        a=actions[h]+1;flows[1]+=pi[h+1]*m.p[a];flows[min(h+1,m.H)+1]+=pi[h+1]*(1-m.p[a])
    end
    require0(flows==pi,"single-policy flow equations")
    pi
end

struct DualData
    lambda::Q
    bound::Q
    bias::Vector{Vector{Q}}
    right::Vector{Int}
    evidence::Dict{String,Any}
end

function dual_bound(d::SingleData,weights::Vector{Q},B::Q)
    B>=0 && !isempty(weights) && all(weights.>0) || throw(ArgumentError("Nonnegative budget and positive holding weights required."))
    points=sort!(unique!(vcat(Q[0],[w*x for w in weights for x in d.roots])))
    vals=Q[sum(minimum(w*d.hbar[k]+lam*d.cbar[k] for k in eachindex(d.hbar)) for w in weights)-lam*B for lam in points]
    best=maximum(vals);at=findfirst(==(best),vals);lam=points[at]
    right=[1+count(x->w*x<=lam,d.roots) for w in weights]
    left=[1+count(x->w*x<lam,d.roots) for w in weights]
    low=sum(d.cbar[k] for k in right);high=sum(d.cbar[k] for k in left)
    require0(low<=B && (lam==0 || high>=B),"dual subgradient bracket")
    theta=lam==0 || low==high ? zero(Q) : (B-low)/(high-low)
    require0(0<=theta<=1,"relaxed mixture weight")
    cc=sum(w*((1-theta)*d.hbar[kl]+theta*d.hbar[kh]) for (w,kl,kh) in zip(weights,right,left))
    rr=(1-theta)*low+theta*high
    require0(cc==best && rr<=B && lam*(B-rr)==0,"relaxed primal/dual equality and complementary slackness")
    biases=Vector{Q}[];occupations=Dict{String,Any}[]
    m=d.model;A=length(m.p)-1
    for (w,kl,kh) in zip(weights,right,left)
        bias=vcat(zero(Q),w.*d.phi[kl].+lam.*d.gamma[kl]);push!(biases,bias)
        gain=w*d.hbar[kl]+lam*d.cbar[kl]
        require0(gain==minimum(w*d.hbar[k]+lam*d.cbar[k] for k in eachindex(d.hbar)),"right-policy lower envelope")
        require0(bias==vcat(zero(Q),w.*d.phi[kh].+lam.*d.gamma[kh]),"bias consistency at tied charge")
        require0(gain==m.wR*bias[1]+(1-m.wR)*bias[2]-bias[1],"zero-state charged Bellman equality")
        for h in 1:m.H,a in 0:A
            q=w*h+lam*m.c[a+1]+(1-m.p[a+1])*bias[min(h+1,m.H)+1]
            require0(q>=gain+bias[h+1],"all-action charged single-project Bellman inequality")
            a==d.policies[kl][h] && require0(q==gain+bias[h+1],"selected single-project Bellman equality")
        end
        pl=single_stationary(m,d.policies[kl]);ph=single_stationary(m,d.policies[kh])
        # Occupation mixtures satisfy single-project stationarity exactly; they
        # need not satisfy the period-by-period joint budget (they are a relaxation).
        occ=zeros(Q,m.H+1,A+1)
        occ[1,1]=(1-theta)*pl[1]+theta*ph[1]
        for h in 1:m.H
            occ[h+1,d.policies[kl][h]+1]+=(1-theta)*pl[h+1]
            occ[h+1,d.policies[kh][h]+1]+=theta*ph[h+1]
        end
        stationary=vec(sum(occ;dims=2));flow=zeros(Q,m.H+1)
        flow[1]=occ[1,1]*m.wR;flow[2]=occ[1,1]*(1-m.wR)
        for h in 1:m.H,a in 0:A
            flow[1]+=occ[h+1,a+1]*m.p[a+1]
            flow[min(h+1,m.H)+1]+=occ[h+1,a+1]*(1-m.p[a+1])
        end
        require0(flow==stationary && sum(stationary)==1 && all(occ.>=0),"relaxed occupation feasibility")
        push!(occupations,Dict("weight"=>qs(w),"right_policy"=>d.policies[kl],"left_policy"=>d.policies[kh],
            "occupation_by_state"=>[qs.(occ[h,:]) for h in 1:m.H+1]))
    end
    e=Dict{String,Any}("lambda_star"=>qs(lam),"bound"=>qs(best),"charge_convention"=>"smallest nonnegative maximizer; right policy processes every exact tied breakpoint",
        "points"=>qs.(points),"dual_values"=>qs.(vals),"right_resource"=>qs(low),"left_resource"=>qs(high),
        "right_derivative"=>qs(low-B),"left_derivative"=>qs(high-B),"relaxed_mixture_weight"=>qs(theta),
        "relaxed_cost"=>qs(cc),"relaxed_resource"=>qs(rr),"relaxed_primal_equals_dual"=>true,
        "relaxed_occupations"=>occupations,"biases"=>[qs.(x) for x in biases],
        "scope"=>"average expected-resource relaxation; not a hard-feasible randomized scheduler")
    DualData(lam,best,biases,right,e)
end

struct KSRecord
    score::Q
    actions::Vector{Int}
end
"""Exact multiple-choice knapsack on reachable rational resource totals.
One complete gear per project enforces all prefix constraints automatically.
Tie: maximum exact score, then least resource, then lexicographically least
gear vector in fixed project order. No float equality or tolerance-based merging.
The state limit fails explicitly; it never truncates the reachable set.
"""
function exact_knapsack(costs::Vector{Vector{Q}},scores::Vector{Vector{Q}},B::Q;max_states::Int=10000)
    length(costs)==length(scores)>0 && B>=0 && max_states>=1 || throw(ArgumentError("Invalid allocation dimensions/budget."))
    states=Dict{Q,KSRecord}(zero(Q)=>KSRecord(zero(Q),Int[]));ties=0;peak=1
    for (cr,sr) in zip(costs,scores)
        length(cr)==length(sr)>0 && cr[1]==0 && all(cr.>=0) && issorted(cr) || throw(ArgumentError("Invalid complete-gear menu."))
        nxt=Dict{Q,KSRecord}()
        for spent in sort!(collect(keys(states)))
            entry=states[spent]
            for a in 0:length(cr)-1
                cost=spent+cr[a+1];cost>B && continue
                score=entry.score+sr[a+1];actions=vcat(entry.actions,a)
                old=get(nxt,cost,nothing)
                if old===nothing || score>old.score || (score==old.score && lexless(actions,old.actions))
                    nxt[cost]=KSRecord(score,actions)
                end
                old!==nothing && score==old.score && (ties+=1)
            end
        end
        isempty(nxt) && error("No feasible gear allocation.")
        length(nxt)<=max_states || error("Exact knapsack reachable-state budget exceeded; no truncated result returned.")
        peak=max(peak,length(nxt));states=nxt
    end
    spent=minimum(keys(states));best=states[spent]
    for c in sort!(collect(keys(states)))
        rec=states[c]
        if rec.score>best.score || (rec.score==best.score && (c<spent || (c==spent && lexless(rec.actions,best.actions))))
            spent=c;best=rec
        end
    end
    (actions=copy(best.actions),score=best.score,resource=spent,tie_comparisons=ties,peak_states=peak)
end

# A maximal allocation admits no componentwise increase. With strictly increasing
# complete-gear resource menus, it suffices to test the next adjacent increment.
# `nothing` is an exact +infinity sentinel (every selected gear is already maximal).
const CapacityKey=Tuple{Q,Union{Nothing,Q}}
capacity_key_order(k)=(k[1],k[2]===nothing,something(k[2],zero(Q)))
min_gap(x,y)=x===nothing ? y : (y===nothing ? x : min(x,y))

"""Exact capacity-restricted multiple-choice knapsack.
`domain=:non_idling` maximizes score over componentwise-maximal allocations.
`domain=:max_resource` first maximizes spend, then score. Negative scores remain
eligible. Exact score ties prefer least spend, then lexicographically least gears.

NI states retain BOTH (spend, minimum next-upgrade cost); a best-score state per
spend alone would discard potentially necessary maximal solutions. A complete
state is NI iff remaining budget is STRICTLY smaller than the minimum upgrade,
or no upgrade exists. Nothing is rounded and no result is greedily 'topped up'.
"""
function exact_capacity_knapsack(costs::Vector{Vector{Q}},scores::Vector{Vector{Q}},B::Q;
        domain::Symbol,max_states::Int=10000)
    domain in (:non_idling,:max_resource) || throw(ArgumentError("Unknown capacity domain."))
    length(costs)==length(scores)>0 && B>=0 && max_states>=1 || throw(ArgumentError("Invalid allocation dimensions/budget."))
    states=Dict{CapacityKey,KSRecord}((zero(Q),nothing)=>KSRecord(zero(Q),Int[]))
    ties=0;peak=1
    for (cr,sr) in zip(costs,scores)
        length(cr)==length(sr)>0 && cr[1]==0 && all(diff(cr).>0) ||
            throw(ArgumentError("Capacity rules require strictly increasing complete-gear costs beginning at zero."))
        nxt=Dict{CapacityKey,KSRecord}()
        for key in sort!(collect(keys(states));by=capacity_key_order)
            spent,gap=key;entry=states[key]
            for a in 0:length(cr)-1
                cost=spent+cr[a+1];cost>B && continue
                own=a+1<length(cr) ? cr[a+2]-cr[a+1] : nothing
                nextgap=domain==:non_idling ? min_gap(gap,own) : nothing
                nk=(cost,nextgap);score=entry.score+sr[a+1];acts=vcat(entry.actions,a)
                old=get(nxt,nk,nothing)
                if old===nothing || score>old.score || (score==old.score && lexless(acts,old.actions))
                    nxt[nk]=KSRecord(score,acts)
                end
                old!==nothing && score==old.score && (ties+=1)
            end
        end
        isempty(nxt) && error("No hard-feasible capacity allocation.")
        length(nxt)<=max_states || error("Capacity-knapsack state budget exceeded; no truncation or greedy repair.")
        peak=max(peak,length(nxt));states=nxt
    end
    best=nothing;spent=zero(Q)
    for key in sort!(collect(keys(states));by=capacity_key_order)
        cost,gap=key
        domain==:non_idling && gap!==nothing && B-cost>=gap && continue
        rec=states[key]
        better=best===nothing
        if best!==nothing
            if domain==:max_resource && cost!=spent
                better=cost>spent
            else
                better=rec.score>best.score || (rec.score==best.score &&
                    (cost<spent || (cost==spent && lexless(rec.actions,best.actions))))
            end
        end
        if better;best=rec;spent=cost;end
    end
    best===nothing && error("No capacity-domain allocation; no fallback to the unconstrained rule.")
    (actions=copy(best.actions),score=best.score,resource=spent,tie_comparisons=ties,peak_states=peak)
end

function solve_allocation_knapsack(costs,scores,B,name;max_states=10000)
    name==:LAGRANGIAN_NI && return exact_capacity_knapsack(costs,scores,B;domain=:non_idling,max_states=max_states)
    name==:LAGRANGIAN_FC && return exact_capacity_knapsack(costs,scores,B;domain=:max_resource,max_states=max_states)
    name in (:WHITTLE_K,:LAGRANGIAN_K,:MYOPIC_K) || throw(ArgumentError("Unknown score policy."))
    exact_knapsack(costs,scores,B;max_states=max_states)
end

function allocation_domain(name)
    name==:LAGRANGIAN_NI && return "componentwise_maximal"
    name==:LAGRANGIAN_FC && return "maximum_attainable_resource"
    name==:WHITTLE_P && return "exposed_DAI_downshift_priority"
    name==:OPTIMAL && return "all_hard_feasible_dynamic_Bellman"
    "all_hard_feasible"
end

# Independent domain comparator: enumerate componentwise domination among ALL
# hard-feasible complete vectors, not the DP's minimum-upgrade statistic.
function capacity_domain_indices(actions,spends,name)
    name==:LAGRANGIAN_FC && return findall(==(maximum(spends)),spends)
    if name==:LAGRANGIAN_NI
        return [k for k in eachindex(actions) if !any(
            b!=actions[k] && all(b.>=actions[k]) for b in actions)]
    end
    collect(eachindex(actions))
end

function score_menus(d::SingleData,w::Vector{Q},dual::DualData,state::Vector{Int},policy::Symbol)
    policy in (:WHITTLE_K,LAGRANGIAN_POLICIES...,:MYOPIC_K) || throw(ArgumentError("Unknown score policy."))
    m=d.model;A=length(m.p)-1;costs=Vector{Q}[];scores=Vector{Q}[]
    for (i,h) in enumerate(state)
        cr=h==0 ? Q[0] : copy(m.c);sr=Q[0];next=min(h+1,m.H)
        for a in 1:length(cr)-1
            val=if policy==:WHITTLE_K
                sr[end]+(m.c[a+1]-m.c[a])*w[i]*d.mpi[h,a]
            elseif policy in LAGRANGIAN_POLICIES
                (m.p[a+1]-m.p[1])*dual.bias[i][next+1]-dual.lambda*m.c[a+1]
            else
                w[i]*next*(m.p[a+1]-m.p[1])
            end
            push!(sr,val)
        end
        push!(costs,cr);push!(scores,sr)
    end
    costs,scores
end
function downshift_allocation(d,w,state,B)
    m=d.model;A=length(m.p)-1;actions=[h==0 ? 0 : A for h in state]
    while any(actions.>0)
        exposed=[(w[i]*d.mpi[state[i],a],i,a) for (i,a) in enumerate(actions) if a>0]
        value,i,a=minimum(exposed)
        sum(m.c[a+1] for a in actions)<=B && value>0 && break
        actions[i]-=1
    end
    require0(sum(m.c[a+1] for a in actions)<=B,"DAI priority feasibility")
    actions
end

struct JointData
    single::SingleData
    weights::Vector{Q}
    budget::Q
    states::Vector{Vector{Int}}
    actions::Vector{Vector{Vector{Int}}}
    transitions::Vector{Matrix{Q}}
    costs::Vector{Q}
end
function joint_data(d::SingleData,weights::Vector{Q},B::Q;max_states=40,max_actions=64)
    m=d.model;L=length(weights);A=length(m.p)-1;n=BigInt(m.H+1)^L
    n<=max_states && 0<L<=3 && all(weights.>0) && B>=0 || throw(ArgumentError("Joint-state/weight/budget admission failed."))
    ranges=[0:m.H for _ in 1:L]
    states=sort!(vec([Int[t...] for t in Iterators.product(ranges...)]);by=x->Tuple(x))
    index=Dict(Tuple(x)=>i for (i,x) in enumerate(states));choices=Vector{Vector{Int}}[];rows=Matrix{Q}[]
    cost=Q[sum(w*h for (w,h) in zip(weights,x)) for x in states]
    for x in states
        menus=[h==0 ? (0:0) : (0:A) for h in x]
        actions=sort!(vec([Int[a...] for a in Iterators.product(menus...) if sum(m.c[aa+1] for aa in a)<=B]);by=a->Tuple(a))
        length(actions)<=max_actions && !isempty(actions) || error("Joint-action admission failed.")
        rr=zeros(Q,length(actions),length(states))
        for (k,a) in enumerate(actions)
            targets=[h==0 ? (0,1) : (0,min(h+1,m.H)) for h in x]
            for y in Iterators.product(targets...)
                prob=one(Q)
                for i in 1:L
                    reset=x[i]==0 ? m.wR : m.p[a[i]+1]
                    prob*=y[i]==0 ? reset : 1-reset
                end
                rr[k,index[y]]+=prob
            end
            require0(sum(rr[k,:])==1 && rr[k,1]>0,"joint stochastic row and common return state")
        end
        push!(choices,actions);push!(rows,rr)
    end
    JointData(d,copy(weights),B,states,choices,rows,cost)
end
function policy_matrix(j::JointData,pi::Vector{Int})
    n=length(j.states);length(pi)==n || throw(DimensionMismatch("One action per joint state required."))
    P=zeros(Q,n,n)
    for i in 1:n
        1<=pi[i]<=length(j.actions[i]) || throw(ArgumentError("Invalid joint action."))
        P[i,:]=j.transitions[i][pi[i],:]
    end
    P
end
function poisson(P::Matrix{Q},cost::Vector{Q})
    n=length(cost);A=zeros(Q,n,n);A[:,1].=one(Q)
    for i in 1:n,k in 2:n;A[i,k]=Q(i==k)-P[i,k];end
    v=solve_exact(A,cost);g=v[1];b=vcat(zero(Q),v[2:end])
    require0(all(g+b[i]==cost[i]+sum(P[i,k]*b[k] for k in 1:n) for i in 1:n),"Poisson equations")
    g,b
end
function stationary(P::Matrix{Q})
    n=size(P,1);A=zeros(Q,n,n)
    for i in 1:n-1,k in 1:n;A[i,k]=Q(i==k)-P[k,i];end
    A[n,:].=one(Q);rhs=zeros(Q,n);rhs[n]=one(Q);pi=solve_exact(A,rhs)
    require0(all(pi.>0) && sum(pi)==1 && transpose(P)*pi==pi,"joint stationary equations/irreducibility")
    pi
end
function capacity_evaluation(j::JointData,pi::Vector{Int},P,occ,resources)
    maxima=Q[maximum(sum(j.single.model.c[a+1] for a in acts) for acts in menu) for menu in j.actions]
    nonmaximal=Q[any(b!=j.actions[i][pi[i]] && all(b.>=j.actions[i][pi[i]]) for b in j.actions[i]) ? 1 : 0 for i in eachindex(pi)]
    belowmax=Q[resources[i]<maxima[i] ? 1 : 0 for i in eachindex(pi)]
    gaps=maxima.-resources
    require0(all(gaps.>=0) && all(nonmaximal.<=belowmax),"non-idling versus maximum expenditure domains")
    gi,bi=poisson(P,nonmaximal);gg,bg=poisson(P,gaps);gf,bf=poisson(P,belowmax)
    require0(gi==dot(occ,nonmaximal) && gg==dot(occ,gaps) && gf==dot(occ,belowmax),"stationary capacity statistics")
    (maxima=maxima,nonmaximal=nonmaximal,belowmax=belowmax,gaps=gaps,
        idling_probability=gi,idling_bias=bi,mean_attainable_gap=gg,gap_bias=bg,
        belowmax_probability=gf,belowmax_bias=bf,mean_maximum_resource=dot(occ,maxima))
end

function evaluate_policy(j::JointData,pi::Vector{Int})
    P=policy_matrix(j,pi);g,b=poisson(P,j.costs);occ=stationary(P)
    require0(dot(occ,j.costs)==g,"stationary cost matches Poisson gain")
    resources=Q[sum(j.single.model.c[a+1] for a in j.actions[i][pi[i]]) for i in eachindex(pi)]
    require0(all(resources.<=j.budget),"pointwise hard budget")
    gr,br=poisson(P,resources);require0(gr==dot(occ,resources) && gr<=j.budget,"stationary resource")
    cap=Q[any(==(j.single.model.H),x) ? 1 : 0 for x in j.states]
    gc,bc=poisson(P,cap);require0(gc==dot(occ,cap),"stationary cap occupancy")
    (cost=g,bias=b,stationary=occ,P=P,resource=gr,resource_bias=br,resources=resources,
        cap_any=gc,cap_bias=bc,cap_indicator=cap,pi=copy(pi),
        capacity=capacity_evaluation(j,pi,P,occ,resources))
end
function optimal_policy(j::JointData;limit=128,seconds=180.0)
    n=length(j.states);pi=ones(Int,n);started=time_ns()
    for iteration in 1:limit
        (time_ns()-started)/1e9<=seconds || error("Exact coupled policy iteration exceeded cooperative budget.")
        P=policy_matrix(j,pi);g,b=poisson(P,j.costs);new=copy(pi)
        for i in 1:n
            vals=Q[j.costs[i]+dot(j.transitions[i][a,:],b) for a in eachindex(j.actions[i])]
            best=argmin(vals)
            # Change only strictly improving actions. This avoids cycling on ties.
            vals[best]<vals[pi[i]] && (new[i]=best)
        end
        if new==pi
            result=evaluate_policy(j,pi)
            checks=0
            for i in 1:n,a in eachindex(j.actions[i])
                residual=j.costs[i]+dot(j.transitions[i][a,:],result.bias)-result.cost-result.bias[i]
                require0(residual>=0 && (a!=pi[i] || residual==0),"coupled optimality certificate")
                checks+=1
            end
            return (policy=pi,evaluation=result,iterations=iteration,bellman_checks=checks)
        end
        pi=new
    end
    error("Exact policy iteration iteration limit; no optimality claim.")
end

function action_table(j::JointData,dual::DualData,name::Symbol;max_ks_states=10000)
    d=j.single;pi=Int[];ties=0;peak=1;started=time_ns()
    for (i,state) in enumerate(j.states)
        if name==:WHITTLE_P
            acts=downshift_allocation(d,j.weights,state,j.budget)
        else
            costs,scores=score_menus(d,j.weights,dual,state,name)
            k=solve_allocation_knapsack(costs,scores,j.budget,name;max_states=max_ks_states);acts=k.actions
            ties+=k.tie_comparisons;peak=max(peak,k.peak_states)
            # Exhaustive feasible-action comparator is independent of the DP.
            values=Q[sum(scores[ell][a[ell]+1] for ell in eachindex(state)) for a in j.actions[i]]
            spends=Q[sum(costs[ell][a[ell]+1] for ell in eachindex(state)) for a in j.actions[i]]
            eligible=capacity_domain_indices(j.actions[i],spends,name)
            require0(!isempty(eligible),"nonempty hard-feasible capacity domain")
            best=first(eligible)
            for q in eligible
                if values[q]>values[best] || (values[q]==values[best] &&
                    (spends[q]<spends[best] || (spends[q]==spends[best] && lexless(j.actions[i][q],j.actions[i][best]))))
                    best=q
                end
            end
            require0(acts==j.actions[i][best] && k.score==values[best] && k.resource==spends[best],"DP/brute-force exact optimizer and tie equivalence")
            if name==:MYOPIC_K
                nextmeans=Q[dot(j.transitions[i][a,:],j.costs) for a in eachindex(j.actions[i])]
                require0(nextmeans[best]==minimum(nextmeans),"myopic minimizes next-slot cost, not action-independent current cost")
            elseif name in LAGRANGIAN_POLICIES
                jointbias=Q[sum(dual.bias[ell][h+1] for (ell,h) in enumerate(x)) for x in j.states]
                qvals=Q[j.costs[i]+dual.lambda*spends[a]+dot(j.transitions[i][a,:],jointbias) for a in eachindex(j.actions[i])]
                require0(qvals[best]==minimum(qvals[eligible]),"charged-Q minimization over the SPECIFIED capacity domain")
                if name==:LAGRANGIAN_FC
                    # On fixed expenditure only, dropping the charge is equivalent.
                    uncharged=qvals.-dual.lambda.*spends
                    require0(spends[best]==maximum(spends) && uncharged[best]==minimum(uncharged[eligible]),"forced capacity and charge cancellation on fixed-expenditure domain")
                end
            end
        end
        idx=findfirst(==(acts),j.actions[i]);require0(idx!==nothing,"allocated action admissible")
        push!(pi,idx)
    end
    (pi=pi,tie_comparisons=ties,peak_knapsack_states=peak,diagnostic_seconds=(time_ns()-started)/1e9)
end

function check_expected(j,dual,optimal,evaluations,expected)
    getq(k)=parseq(expected[k])
    require0(length(j.states)==expected["joint_states"] && sum(length,j.actions)==expected["joint_actions"],"independent inventory")
    require0(j.budget==getq("budget") && dual.lambda==getq("lambda_star") && dual.bound==getq("lagrangian_bound") &&
        optimal.evaluation.cost==getq("optimal_cost"),"independent rational budget/dual/coupled optimum")
    for name in POLICIES
        e=evaluations[name];r=expected["policies"][String(name)]
        require0(e.cost==parseq(r["cost"]) && e.resource==parseq(r["resource"]) && e.cap_any==parseq(r["stationary_cap_any"]),"independent rational policy rates")
        require0(e.stationary==parseq.(r["stationary"]) && e.bias==parseq.(r["cost_bias"]),"independent policy stationary distribution/bias")
        acts=[j.actions[i][e.pi[i]] for i in eachindex(e.pi)]
        require0(acts==r["actions"],"independent allocation decisions/ties")
        cp=e.capacity
        require0(cp.idling_probability==parseq(r["nonmaximal_probability"]) &&
            cp.belowmax_probability==parseq(r["below_maximum_resource_probability"]) &&
            cp.mean_attainable_gap==parseq(r["mean_attainable_resource_gap"]) &&
            cp.mean_maximum_resource==parseq(r["mean_maximum_attainable_resource"]),"independent stationary capacity diagnostics")
    end
    true
end

include("rayleigh_allocation_simulation.jl")

function validate_system(spec,config;expected=nothing,simulate::Bool=true,system_number=1)
    started=time_ns();seconds=Float64(config["seconds_per_system"])
    function check_budget(phase)
        (time_ns()-started)/1e9<=seconds || error("System cooperative wall budget exceeded at "*phase)
    end
    d=single_data(spec;family_limit=Int(config["max_family_policies"]))
    check_budget("single-project verification")
    w=Q[parseq(x) for x in spec["weights"]];B=parseq(spec["budget_maxgear_multiple"])*d.model.c[end]
    dual_started=time_ns();dual=dual_bound(d,w,B);dual_seconds=(time_ns()-dual_started)/1e9
    j=joint_data(d,w,B;max_states=Int(config["max_joint_states"]),max_actions=Int(config["max_joint_actions"]))
    check_budget("joint-model construction")
    optimal=optimal_policy(j;limit=Int(config["max_policy_iterations"]),seconds=max(0.01,seconds-(time_ns()-started)/1e9))
    require0(dual.bound<=optimal.evaluation.cost,"dual cannot exceed true hard-budget optimum")
    lbias=Q[sum(dual.bias[ell][h+1] for (ell,h) in enumerate(x)) for x in j.states]
    lower_checks=0
    for i in eachindex(j.states),a in eachindex(j.actions[i])
        slack=j.costs[i]-dual.bound-lbias[i]+dot(j.transitions[i][a,:],lbias)
        require0(slack>=0,"Lagrangian bound is a subsolution for EVERY feasible joint action")
        lower_checks+=1
    end
    evaluations=Dict{Symbol,Any}();pol_records=Dict{String,Any}();tabletimes=Float64[]
    for name in POLICIES
        check_budget("policy evaluation")
        table=action_table(j,dual,name;max_ks_states=Int(config["max_knapsack_states"]))
        e=evaluate_policy(j,table.pi);evaluations[name]=e;push!(tabletimes,table.diagnostic_seconds)
        require0(e.cost>=optimal.evaluation.cost && e.cost>=dual.bound,"policy value sandwich")
        pol_records[String(name)]=policy_record(j,e,dual,optimal.evaluation.cost;table=table,name=name)
    end
    evaluations[:OPTIMAL]=optimal.evaluation
    pol_records["OPTIMAL"]=policy_record(j,optimal.evaluation,dual,optimal.evaluation.cost;name=:OPTIMAL)
    require0(iszero(optimal.evaluation.capacity.idling_probability),"optimal stationary policy is non-idling in this model")
    for name in (:LAGRANGIAN_NI,:LAGRANGIAN_FC)
        require0(all(iszero,evaluations[name].capacity.nonmaximal),"forced rule leaves no componentwise upgrade unused")
    end
    require0(all(iszero,evaluations[:LAGRANGIAN_FC].capacity.gaps),"FC uses maximum ATTAINABLE resource at every state")
    expected!==nothing && check_expected(j,dual,optimal,evaluations,expected)
    # Exact binary/unit-capacity specialization, including ties and nonpositive Lag scores.
    if spec["id"]=="binary_unit_H2"
        require0(d.model.c==Q[0,1] && B==1,"binary normalization")
        for (i,x) in enumerate(j.states),name in (:WHITTLE_K,:LAGRANGIAN_K,:MYOPIC_K)
            _,scores=score_menus(d,w,dual,x,name);benefits=Q[length(v)>1 ? v[2] : 0 for v in scores]
            aa=j.actions[i][evaluations[name].pi[i]]
            require0(sum(aa)<=1 && sum(benefits[k]*aa[k] for k in eachindex(aa))==max(zero(Q),maximum(benefits)),"binary highest-positive-index specialization")
        end
    end
    if spec["id"]=="binary_unit_H2"
        for (i,x) in enumerate(j.states)
            ani=j.actions[i][evaluations[:LAGRANGIAN_NI].pi[i]]
            afc=j.actions[i][evaluations[:LAGRANGIAN_FC].pi[i]]
            require0(ani==afc && sum(ani)==min(1,count(>(0),x)),"equal-unit binary NI/FC specialization")
        end
    end
    if spec["id"]=="capacity_tradeoff_H1_A2"
        i=findfirst(==([1,1]),j.states);optact=j.actions[i][optimal.policy[i]]
        fc=evaluations[:LAGRANGIAN_FC]
        require0(optact==[2,0] && optimal.evaluation.cost==Q(257//217) && fc.cost==Q(116//97),"strict counterexample to maximum-expenditure optimality")
        require0(fc.cost>optimal.evaluation.cost && optimal.evaluation.capacity.belowmax_probability>0,"optimal non-idling can leave attainable expenditure unused")
    end
    check_budget("simulation start")
    simulation=simulate ? simulate_system(j,dual,evaluations,config["simulation"];seed=UInt64(config["master_seed"])+UInt64(1000003)*UInt64(system_number),seconds_limit=max(0.01,seconds-(time_ns()-started)/1e9)) :
        Dict{String,Any}("executed"=>false)
    check_budget("system completion")
    comparisons=Dict{String,Any}[]
    for (a,b) in ((:LAGRANGIAN_NI,:LAGRANGIAN_K),(:LAGRANGIAN_FC,:LAGRANGIAN_K),(:LAGRANGIAN_FC,:LAGRANGIAN_NI))
        x=evaluations[a];y=evaluations[b]
        push!(comparisons,Dict("comparison"=>String(a)*" minus "*String(b),"cost_difference"=>qs(x.cost-y.cost),
            "resource_difference"=>qs(x.resource-y.resource),"nonmaximal_probability_difference"=>qs(x.capacity.idling_probability-y.capacity.idling_probability),
            "same_lambda_star"=>qs(dual.lambda),"same_lagrangian_bound"=>qs(dual.bound),
            "scope"=>"descriptive exact policy comparison; no universal ordering or forced-usage optimality claim"))
    end
    elapsed=(time_ns()-started)/1e9
    record=Dict{String,Any}("id"=>spec["id"],"passed"=>true,"specification"=>Dict(spec),"revision"=>REVISION,
        "criterion"=>"average","scope"=>"tiny coupled-system exact allocation/bound validation; not scalable allocation timing",
        "joint_states"=>length(j.states),"joint_actions"=>sum(length,j.actions),"states"=>j.states,
        "budget"=>qs(B),"holding_weights"=>qs.(w),
        "weighted_indices_by_project"=>[[qs.(ww .* d.mpi[h,:]) for h in 1:d.model.H] for ww in w],
        "single_project"=>d.evidence,"dual"=>dual.evidence,
        "optimal_cost"=>qs(optimal.evaluation.cost),"coupled_policy_iterations"=>optimal.iterations,
        "coupled_bellman_checks"=>optimal.bellman_checks,"lagrangian_all_feasible_action_subsolution_checks"=>lower_checks,
        "independent_python_reference_matched"=>(expected!==nothing),"joint_bias_subsolution"=>qs.(lbias),
        "policies"=>pol_records,"lagrangian_capacity_comparisons"=>comparisons,"simulation"=>simulation,"dual_seconds_diagnostic"=>dual_seconds,
        "action_table_seconds_diagnostic"=>sum(tabletimes),"diagnostic_elapsed_seconds"=>elapsed,
        "scope_of_times"=>"diagnostic only; exact validation and table construction are not online floating-point scheduler timings")
    record
end

function policy_record(j,e,dual,gstar;table=nothing,name=:OPTIMAL)
    L=length(j.weights);m=j.single.model;A=length(m.p)-1
    gear=zeros(Q,L,A+1);cap=zeros(Q,L)
    for i in eachindex(j.states),ell in 1:L
        gear[ell,j.actions[i][e.pi[i]][ell]+1]+=e.stationary[i]
        j.states[i][ell]==m.H && (cap[ell]+=e.stationary[i])
    end
    Dict{String,Any}("average_cost"=>qs(e.cost),"average_resource"=>qs(e.resource),
        "absolute_bound_gap"=>qs(e.cost-dual.bound),"relative_bound_gap"=>qs((e.cost-dual.bound)/dual.bound),
        "absolute_optimality_gap"=>qs(e.cost-gstar),"relative_optimality_gap"=>qs((e.cost-gstar)/gstar),
        "resource_utilization"=>(j.budget==0 ? "undefined_zero_budget" : qs(e.resource/j.budget)),
        "stationary"=>qs.(e.stationary),"cost_bias"=>qs.(e.bias),"resource_bias"=>qs.(e.resource_bias),
        "cap_occupancy_any"=>qs(e.cap_any),"cap_occupancy_by_project"=>qs.(cap),
        "gear_occupancy_by_project"=>[qs.(gear[k,:]) for k in 1:L],
        "actions"=>[j.actions[i][e.pi[i]] for i in eachindex(e.pi)],"hard_feasible_all_states"=>true,
        "capacity_domain"=>allocation_domain(name),
        "nonmaximal_state_count"=>count(==(one(Q)),e.capacity.nonmaximal),
        "below_maximum_resource_state_count"=>count(==(one(Q)),e.capacity.belowmax),
        "nonmaximal_probability"=>qs(e.capacity.idling_probability),
        "below_maximum_resource_probability"=>qs(e.capacity.belowmax_probability),
        "mean_maximum_attainable_resource"=>qs(e.capacity.mean_maximum_resource),
        "mean_attainable_resource_gap"=>qs(e.capacity.mean_attainable_gap),
        "mean_unused_budget"=>qs(j.budget-e.resource),
        "selected_resource_by_state"=>qs.(e.resources),
        "maximum_attainable_resource_by_state"=>qs.(e.capacity.maxima),
        "tie_rule"=>(name==:OPTIMAL ? "exact policy iteration: strict improvements only; retain current action at equal Bellman value" :
            name==:WHITTLE_P ? "least exact exposed DAI, then fixed project label, then upper gear" :
            name==:LAGRANGIAN_FC ? "maximum resource, maximum exact charged score, lexicographically least gears" :
            name==:LAGRANGIAN_NI ? "among componentwise maximal vectors: maximum exact charged score, least resource, lexicographically least gears" :
            "score policies: exact max score, least resource, lexicographically least gears; priority: least exact exposed DAI, then project, then gear"),
        "allocation_table_seconds_diagnostic"=>(table===nothing ? 0.0 : table.diagnostic_seconds),
        "knapsack_tie_comparisons"=>(table===nothing ? 0 : table.tie_comparisons),
        "peak_reachable_knapsack_states"=>(table===nothing ? 0 : table.peak_knapsack_states))
end

end # module
