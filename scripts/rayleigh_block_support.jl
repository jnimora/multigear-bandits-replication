"""Autonomous safeguarded numerical block downshifting for capped Rayleigh/AoII.

Float64 values carry outward-rounded enclosures of the frozen source primitives.
No exact path or reference DAI is an input to the selector. Bounded machine-Int
constants take an exact, allocation-free Float64 conversion route. Rational input
conversion, exact slope-equality tests and budgeted current-policy fallback retain
their prior semantics. No BigFloat precision or global rounding mode is changed.
This module computes a threshold-family MPI path, not an indexability certificate.
"""
module RayleighBlock
using ..MultiGearBandits
using ..RayleighAoII
const Q = Rational{BigInt}
const BLOCK_EFFICIENCY_REVISION = "block-workspace-efficiency-v1"
export BlockOptions, NumericBlockWorkspace, numerical_downshift, numerical_frontier!,
       numerical_shift!, topology_performance_input, frozen_actions, choose_frontier!,
       geometric, geometric_shrink, Enclosed, contains_exact, accept_numerical_case,
       stress_cases, evaluator_stress_checks, topology_rejection_checks

struct NumericalIssue <: Exception
    status::Symbol
    message::String
end
Base.showerror(io::IO,e::NumericalIssue)=print(io,e.status,": ",e.message)

# Enclosures use ordinary IEEE operations widened by one representable neighbour.
# No @fastmath / reassociation / directed global rounding. NaN/Inf is a rejection,
# never a finite substitute. Known nonnegative statistic bounds are intersected.
struct Enclosed
    value::Float64
    lo::Float64
    hi::Float64
    function Enclosed(v::Float64,lo::Float64,hi::Float64)
        all(isfinite,(v,lo,hi)) && lo<=hi || throw(NumericalIssue(:nonfinite_enclosure,"invalid or overflowed interval"))
        new(clamp(v,lo,hi),lo,hi)
    end
end
function Enclosed(x::Q)
    v=Float64(x)
    isfinite(v) || throw(ArgumentError("Source primitive is outside Float64 range."))
    # Test conversion by exact comparison; do not assume a platform conversion's
    # last-bit direction. At most four neighbouring expansions are admitted.
    lo=v;hi=v
    for _ in 1:4
        Q(lo)<=x<=Q(hi) && return Enclosed(v,lo,hi)
        Q(lo)>x && (lo=prevfloat(lo))
        Q(hi)<x && (hi=nextfloat(hi))
        all(isfinite,(lo,hi)) || break
    end
    throw(ArgumentError("Could not enclose source rational in finite Float64."))
end
# Every integer in [-2^53,2^53] is exactly representable in binary64.
# Test in the integer domain (no abs(typemin(Int)) or float round-trip overflow).
# The model's ages/lengths use Int and lie well inside this range. Preserve the
# original rational conversion for large machine integers and all other Integers.
function Enclosed(x::Int)
    if -9007199254740992 <= x <= 9007199254740992
        v=Float64(x)
        return Enclosed(v,v,v)
    end
    Enclosed(Q(x))
end
Enclosed(x::Integer)=Enclosed(Q(x))
const BZERO=Enclosed(0.0,0.0,0.0)
const BONE=Enclosed(1.0,1.0,1.0)
Base.:+(a::Enclosed,b::Enclosed)=Enclosed(a.value+b.value,prevfloat(a.lo+b.lo),nextfloat(a.hi+b.hi))
Base.:-(a::Enclosed,b::Enclosed)=Enclosed(a.value-b.value,prevfloat(a.lo-b.hi),nextfloat(a.hi-b.lo))
function Base.:*(a::Enclosed,b::Enclosed)
    z=(a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi)
    Enclosed(a.value*b.value,prevfloat(minimum(z)),nextfloat(maximum(z)))
end
function Base.:/(a::Enclosed,b::Enclosed)
    b.lo>0 || throw(NumericalIssue(:unresolved_denominator,"denominator not separated from zero"))
    z=(a.lo/b.lo,a.lo/b.hi,a.hi/b.lo,a.hi/b.hi)
    Enclosed(a.value/b.value,prevfloat(minimum(z)),nextfloat(maximum(z)))
end
function intersect_known(x::Enclosed,lo::Float64,hi::Float64)
    l=max(lo,x.lo);u=min(hi,x.hi)
    l<=u || throw(NumericalIssue(:invalid_statistic,"geometric enclosure contradicts analytic bounds"))
    Enclosed(clamp(x.value,l,u),l,u)
end
contains_exact(x::Enclosed,y::Q)=Q(x.lo)<=y<=Q(x.hi)
function accurate(x::Enclosed,atol,rtol)
    # A sufficient absolute-plus-relative-to-returned-value bound, not a screen
    # on agreement between two floating approximations.
    max(abs(x.value-x.lo),abs(x.hi-x.value))<=atol+rtol*abs(x.value)
end

struct Geometric
    n::Int
    power::Enclosed
    total::Enclosed
    weighted::Enclosed
end
function geom_bounds(n,p,l,j)
    n==0 && return Geometric(0,BONE,BZERO,BZERO)
    n==1 && return Geometric(1,intersect_known(p,0.0,1.0),BONE,BZERO)
    Geometric(n,intersect_known(p,0.0,1.0),intersect_known(l,1.0,Float64(n)),
        intersect_known(j,0.0,Float64(n*(n-1)÷2)))
end
function geom_concat(x::Geometric,y::Geometric)
    x.n==0 && return y
    y.n==0 && return x
    geom_bounds(x.n+y.n,x.power*y.power,x.total+x.power*y.total,
        x.weighted+x.power*(Enclosed(x.n)*y.total+y.weighted))
end
"""Positive-sum doubling: no (1-q^n)/(1-q) or cancellation-prone closed form.
O(log(n+1)) arithmetic; also used for explicitly counted numerical rebuilds.
"""
function geometric(q::Enclosed,n::Int)
    0<=n<=10000 || throw(ArgumentError("Geometric length outside 0:10000."))
    0<q.lo<=q.hi<1 || throw(ArgumentError("Survival probability must be separated from 0 and 1."))
    out=Geometric(0,BONE,BZERO,BZERO);b=Geometric(1,q,BONE,BZERO);k=n
    while k>0
        isodd(k) && (out=geom_concat(out,b))
        k>>=1
        k>0 && (b=geom_concat(b,b))
    end
    out
end
"""Remove the last term. Rebuild instead of dividing a lost/subnormal power.
Returns (statistics, rebuilt_for_underflow). A vanished power is NEVER assumed
irrecoverable: shrinking back toward a shorter block recomputes it from q,n.
"""
function geometric_shrink(g::Geometric,q::Enclosed)
    g.n>0 || throw(ArgumentError("Cannot shorten empty block."))
    g.n<=2 && return (geometric(q,g.n-1),false)
    if g.power.lo<=4 * floatmin(Float64)
        return (geometric(q,g.n-1),true)
    end
    last=g.power/q
    (geom_bounds(g.n-1,last,g.total-last,g.weighted-Enclosed(g.n-1)*last),false)
end

Base.@kwdef struct BlockOptions
    value_atol::Float64=1e-12
    value_rtol::Float64=1e-10
    refresh_every::Int=64
    exact_fallback::Bool=true
    exact_max_H::Int=64
    exact_max_A::Int=8
    exact_max_steps::Int=128
    exact_max_input_bits::Int=2048
    seconds_limit::Float64=120.0
end
function check_options(o::BlockOptions)
    all(isfinite,(o.value_atol,o.value_rtol,o.seconds_limit)) &&
    o.value_atol>=0 && o.value_rtol>=0 && o.value_atol+o.value_rtol>0 &&
    1<=o.refresh_every<=100000 && 0<=o.exact_max_H<=10000 &&
    0<=o.exact_max_A<=64 && 0<=o.exact_max_steps<=100000 &&
    1<=o.exact_max_input_bits<=100000 && o.seconds_limit>0 || throw(ArgumentError("Invalid block options."))
    nothing
end
struct FrozenSource
    H::Int
    p::Vector{Q}
    c::Vector{Q}
    u::Q
end
function FrozenSource(m::AoIIModel)
    # Q(Float64) is the EXACT stored binary number. Do not rationalize to a nearby
    # fraction or regenerate p,c from rounded success probabilities.
    p=Q.(m.p);c=Q.(m.c);u=one(Q)-Q(m.wR)
    0<u<1 && length(p)==length(c)>=2 && all(0<x<1 for x in p) && c[1]==0 &&
    all(p[a]>p[a-1] && c[a]>c[a-1] for a in 2:length(p)) || throw(ArgumentError("Invalid frozen AoII primitives."))
    # The ordinary Float64 representation must preserve its positive topology and
    # distinct gears. Extreme under-resolved inputs are rejected, not perturbed.
    fp=Float64.(p);fc=Float64.(c);fq=Float64.(one(Q).-p)
    all(0<x<1 for x in vcat(fp,fq,[Float64(u),Float64(1-u)])) &&
    all(fp[a]>fp[a-1] && fc[a]>fc[a-1] for a in 2:length(p)) ||
        throw(ArgumentError("Float64 conversion collapses a probability, edge, or resource/gear increment."))
    FrozenSource(m.H,copy(p),copy(c),u)
end
struct Candidate
    h::Int
    a::Int
    f::Enclosed
    g::Enclosed
    index::Enclosed
end
mutable struct NumericBlockWorkspace
    source::FrozenSource
    p::Vector{Enclosed}
    q::Vector{Enclosed}
    c::Vector{Enclosed}
    dp::Vector{Enclosed}
    slope::Vector{Enclosed}
    exact_slope::Vector{Q}
    linear_resource::Bool
    u::Enclosed
    lengths::Vector{Int}
    stats::Vector{Geometric}
    short::Vector{Geometric}
    updates::Vector{Int}
    starts::Vector{Int}
    terminal_gear::Int
    # Task-private scratch. Only candidate slots 1:count returned by
    # _numerical_frontier! are valid; unused slots must never be inspected.
    successors::Vector{NTuple{3,Enclosed}}
    frontier_ages::Vector{Int}
    candidates::Vector{Candidate}
    options::BlockOptions
    counts::Dict{String,Int}
end
function ieee_check()
    get_zero_subnormals() && throw(ArgumentError("Flush-to-zero is enabled on the calling thread. Use IEEE subnormal arithmetic; the package does not change this setting."))
    nothing
end
function NumericBlockWorkspace(m::AoIIModel;options=BlockOptions(),actions=nothing)
    check_options(options);ieee_check();s=FrozenSource(m);A=length(s.p)-1;H=s.H
    act=actions===nothing ? fill(A,H) : Int.(actions)
    length(act)==H && issorted(act) && all(a->0<=a<=A,act) || throw(ArgumentError("Expected a nondecreasing positive-age policy."))
    p=Enclosed.(s.p);q=Enclosed.(one(Q).-s.p);c=Enclosed.(s.c)
    dp=[Enclosed(s.p[a+1]-s.p[a]) for a in 1:A]
    r=[(s.c[a+1]-s.c[a])/(s.p[a+1]-s.p[a]) for a in 1:A]
    lengths=zeros(Int,A+1)
    for h in 1:H-1;lengths[act[h]+1]+=1;end
    stats=[geometric(q[a],lengths[a]) for a in 1:A+1]
    short=[geometric(q[a],max(0,lengths[a]-1)) for a in 1:A+1]
    counts=Dict(k=>0 for k in ("block_visits","candidate_evaluations","growth_updates","shrink_updates",
        "periodic_stat_rebuilds","underflow_stat_rebuilds","comparison_refreshes","exact_policy_evaluations",
        "interval_selections","structural_tie_selections","exact_selections","exact_tie_selections"))
    NumericBlockWorkspace(s,p,q,c,dp,Enclosed.(r),r,all(==(r[1]),r),Enclosed(s.u),lengths,stats,short,
        zeros(Int,A+1),zeros(Int,A+1),act[end],
        Vector{NTuple{3,Enclosed}}(undef,A),zeros(Int,A),Vector{Candidate}(undef,A),options,counts)
end
function frozen_actions(w::NumericBlockWorkspace)
    acts=Int[]
    for a in 0:length(w.lengths)-1;append!(acts,fill(a,w.lengths[a+1]));end
    push!(acts,w.terminal_gear)
    length(acts)==w.source.H || error("Block-length invariant failed.")
    acts
end
function rebuild_stats!(w)
    for s in eachindex(w.lengths)
        w.stats[s]=geometric(w.q[s],w.lengths[s])
        w.short[s]=geometric(w.q[s],max(0,w.lengths[s]-1));w.updates[s]=0
    end
    nothing
end
function block_apply(g::Geometric,l::Int,c::Enclosed,tail)
    g.n==0 && return tail
    (g.total+g.power*tail[1],Enclosed(l)*g.total+g.weighted+g.power*tail[2],
        c*g.total+g.power*tail[3])
end
"""Evaluate candidates from a directly constructed short block, NOT by subtracting
its head and dividing by q. All positive-age/cap conventions remain unchanged.
"""
# Internal workspace-backed scratch route used only by the live selector.
# Arithmetic expressions and their evaluation order below match the v1 kernel.
function _numerical_frontier!(w::NumericBlockWorkspace)
    ieee_check();H=w.source.H;A=length(w.p)-1
    sum(w.lengths)==H-1 && all(n->n>=0,w.lengths) || error("Interior-length invariant failed.")
    all(w.lengths[a+1]==0 for a in (w.terminal_gear+1):A) || error("Invalid terminal gear.")
    pos=1
    for s in 1:A+1;w.starts[s]=pos;pos+=w.lengths[s];end
    a=w.terminal_gear+1;cap=(BONE/w.p[a],Enclosed(H)/w.p[a],w.c[a]/w.p[a]);tail=cap
    nxt=w.successors;ages=w.frontier_ages;fill!(ages,0)
    for a in A:-1:0
        w.counts["block_visits"]+=1;s=a+1;n=w.lengths[s]
        if n>0
            h=w.starts[s]
            if a>0;ages[a]=h;nxt[a]=block_apply(w.short[s],h+1,w.c[s],tail);end
            tail=block_apply(w.stats[s],h,w.c[s],tail)
        elseif w.terminal_gear==a && a>0
            ages[a]=H;nxt[a]=cap
        end
    end
    den=BONE+w.u*tail[1];hb=w.u*tail[2]/den;cb=w.u*tail[3]/den
    # If all exact resource/reset slopes equal k, E_s=k*(1-p0*T_s),
    # hence k-Gamma_s=k*(p0+u)*T_s/(1+u*T_1). Evaluate this positive
    # expression instead of subtracting almost equal k and Gamma_s.
    linear_scale=w.linear_resource ? w.slope[1]*(w.p[1]+w.u)/den : BZERO
    front=w.candidates;count=0
    for a in 1:A
        ages[a]==0 && continue
        t,v,e=nxt[a];phi=v-hb*t;gamma=e-cb*t
        d=w.linear_resource ? linear_scale*t : w.slope[a]-gamma
        d.lo>0 || throw(NumericalIssue(:unresolved_resource,"adjacent resource not separated from zero at ($(ages[a]),$a)"))
        f=w.dp[a]*phi;g=w.dp[a]*d;idx=phi/d
        g.lo>0 || throw(NumericalIssue(:unresolved_resource,"scaled marginal resource underflows/is unresolved"))
        count+=1;front[count]=Candidate(ages[a],a,f,g,idx);w.counts["candidate_evaluations"]+=1
    end
    (count=count,hbar=hb,cbar=cb)
end
"""Owned diagnostic frontier, preserving the existing public API.
The candidate vector is a copy: later evaluation/shifting of w cannot mutate it.
The live selector uses _numerical_frontier! directly and never pays for this copy.
"""
function numerical_frontier!(w::NumericBlockWorkspace)
    e=_numerical_frontier!(w)
    (frontier=w.candidates[1:e.count],hbar=e.hbar,cbar=e.cbar)
end
function symbolic_tie(w,x::Candidate,y::Candidate)
    min(x.h+1,w.source.H)==min(y.h+1,w.source.H) && w.exact_slope[x.a]==w.exact_slope[y.a]
end
# Stable in-place reduction: same overlap test, first-candidate equality test,
# and lexicographic tie rule as v1; no vector of possible minima and no sorting.
# A valid length is explicit so stale scratch slots cannot influence a decision.
function interval_choice(w,front,n::Int=length(front))
    0<=n<=length(front) || throw(ArgumentError("Invalid frontier prefix length."))
    n>0 || throw(NumericalIssue(:empty_frontier,"no feasible candidate"))
    minhi=front[1].index.hi
    for j in 2:n
        minhi=min(minhi,front[j].index.hi)
    end
    first=0;best=0;possible=0;all_tied=true
    for j in 1:n
        x=front[j]
        if x.index.lo<=minhi
            possible+=1
            if first==0
                first=j;best=j
            else
                all_tied && (all_tied=symbolic_tie(w,front[first],x))
                isless((x.h,x.a),(front[best].h,front[best].a)) && (best=j)
            end
        end
    end
    possible==1 && return (front[first],:interval)
    all_tied && return (front[best],:structural_tie)
    nothing
end
function sufficient_output(c::Candidate,o)
    c.g.lo>0 && all(x->accurate(x,o.value_atol,o.value_rtol),(c.f,c.g,c.index))
end
function exact_current_frontier(w)
    s=w.source;H=s.H;A=length(s.p)-1;o=w.options
    o.exact_fallback && H<=o.exact_max_H && A<=o.exact_max_A &&
        w.counts["exact_policy_evaluations"]<o.exact_max_steps ||
        throw(NumericalIssue(:unresolved_comparison,"exact current-policy fallback is disabled or outside its saved limits"))
    all(max(ndigits(abs(numerator(x));base=2),ndigits(denominator(x);base=2))<=o.exact_max_input_bits
        for x in vcat(s.p,s.c,[s.u])) || throw(NumericalIssue(:exact_input_budget,"source rational bit limit exceeded"))
    w.counts["exact_policy_evaluations"]+=1
    acts=frozen_actions(w);t=zeros(Q,H);v=zeros(Q,H);e=zeros(Q,H)
    a=acts[H]+1;t[H]=1/s.p[a];v[H]=H/s.p[a];e[H]=s.c[a]/s.p[a]
    for h in H-1:-1:1
        a=acts[h]+1;q=1-s.p[a]
        t[h]=1+q*t[h+1];v[h]=h+q*v[h+1];e[h]=s.c[a]+q*e[h+1]
    end
    den=1+s.u*t[1];hb=s.u*v[1]/den;cb=s.u*e[1]/den
    rows=Tuple{Int,Int,Q,Q,Q}[]
    for h in 1:H
        a=acts[h]
        if a>0 && (h==1 || acts[h-1]<a)
            k=min(h+1,H);dp=s.p[a+1]-s.p[a]
            f=dp*(v[k]-hb*t[k]);g=s.c[a+1]-s.c[a]-dp*(e[k]-cb*t[k])
            g>0 || throw(NumericalIssue(:nonpositive_resource,"exact frontier resource at ($h,$a) is $g"))
            push!(rows,(h,a,f,g,f/g))
        end
    end
    isempty(rows) && throw(NumericalIssue(:empty_frontier,"exact frontier empty"))
    sort!(rows;by=x->(x[5],x[1],x[2]));r=first(rows)
    c=Candidate(r[1],r[2],Enclosed(r[3]),Enclosed(r[4]),Enclosed(r[5]))
    sufficient_output(c,o) || throw(NumericalIssue(:unrepresentable_output,"exact answer cannot meet requested Float64 output accuracy"))
    tied=count(x->x[5]==r[5],rows)>1
    w.counts[tied ? "exact_tie_selections" : "exact_selections"]+=1
    (candidate=c,mode=tied ? :exact_tie : :exact,hbar=Enclosed(hb),cbar=Enclosed(cb))
end
"""Select live, certify separation/equality, refresh once if needed, then resolve
only this current policy exactly within saved limits. No tolerance-based tie merge.
"""
function choose_frontier!(w::NumericBlockWorkspace)
    for attempt in 1:2
        try
            e=_numerical_frontier!(w);choice=interval_choice(w,w.candidates,e.count)
            if choice!==nothing
                c,mode=choice
                if sufficient_output(c,w.options)
                    w.counts[mode==:interval ? "interval_selections" : "structural_tie_selections"]+=1
                    return (candidate=c,mode=mode,hbar=e.hbar,cbar=e.cbar)
                end
            end
        catch err
            err isa NumericalIssue || rethrow()
        end
        if attempt==1
            rebuild_stats!(w);w.counts["comparison_refreshes"]+=1
        end
    end
    exact_current_frontier(w)
end
function change_geometric(w,s,delta)
    n=w.lengths[s]+delta;n>=0 || error("Negative block length.")
    if w.updates[s]+1>=w.options.refresh_every
        w.counts["periodic_stat_rebuilds"]+=1
        return geometric(w.q[s],n),geometric(w.q[s],max(0,n-1)),0
    end
    if delta==1
        return geom_concat(w.stats[s],Geometric(1,w.q[s],BONE,BZERO)),w.stats[s],w.updates[s]+1
    end
    current=w.short[s]
    if n==0;return current,current,w.updates[s]+1;end
    short,rebuilt=geometric_shrink(current,w.q[s])
    rebuilt && (w.counts["underflow_stat_rebuilds"]+=1)
    current,short,w.updates[s]+1
end
function numerical_shift!(w::NumericBlockWorkspace,h::Int,a::Int)
    H=w.source.H;A=length(w.p)-1
    1<=h<=H && 1<=a<=A || throw(ArgumentError("Invalid shift label."))
    if h==H
        w.terminal_gear==a && w.lengths[a+1]==0 || error("Infeasible cap shift.")
        w.terminal_gear-=1
    else
        w.lengths[a+1]>0 && 1+sum(@view(w.lengths[1:a]))==h || error("Shift is not the block head.")
        hi=change_geometric(w,a+1,-1);lo=change_geometric(w,a,1)
        # No partial mutation of policy if statistic construction failed.
        w.stats[a+1],w.short[a+1],w.updates[a+1]=hi
        w.stats[a],w.short[a],w.updates[a]=lo
        w.lengths[a+1]-=1;w.lengths[a]+=1
        w.counts["shrink_updates"]+=1;w.counts["growth_updates"]+=1
    end
    nothing
end
function numerical_downshift(m::AoIIModel;options=BlockOptions())
    w=NumericBlockWorkspace(m;options=options);H=w.source.H;A=length(w.p)-1
    indices=Matrix{Union{Missing,Float64}}(undef,H,A);fill!(indices,missing)
    order=Tuple{Int,Int}[];assigned=Float64[];fs=Float64[];gs=Float64[]
    bounds=Vector{Float64}[];modes=Symbol[];status=:running;message="";started=time_ns()
    for _ in 1:H*A
        if (time_ns()-started)/1e9>options.seconds_limit
            status=:time_limit;message="cooperative numerical-loop time limit reached";break
        end
        try
            selected=choose_frontier!(w);c=selected.candidate
            numerical_shift!(w,c.h,c.a)
            push!(order,(c.h,c.a));push!(assigned,c.index.value);push!(fs,c.f.value);push!(gs,c.g.value)
            push!(bounds,[c.index.lo,c.index.hi,c.f.lo,c.f.hi,c.g.lo,c.g.hi]);push!(modes,selected.mode)
            indices[c.h,c.a]=c.index.value
        catch err
            err isa NumericalIssue || rethrow()
            status=err.status;message=err.message;break
        end
    end
    complete=length(order)==H*A
    if complete
        w.terminal_gear==0 && w.lengths[1]==H-1 && all(iszero,w.lengths[2:end]) || error("Incomplete terminal policy.")
        status=:complete
    end
    (complete=complete,status=status,message=message,mpi=indices,order=order,assignments=assigned,f=fs,g=gs,
     bounds=bounds,modes=modes,counts=copy(w.counts),certified_prefix=true,dai_verified=false,
     scope="numerical threshold-family MPI path; DAI verification is separate",workspace=w)
end

"""Opt-in structural admission. The default dense-positive guard stays in force
for ordinary PerformanceInput(model); no structural zero is changed.
"""
function topology_performance_input(m::AoIIModel)
    model=dense_model(m;exact=false)
    PerformanceInput(model;average_topology=:reset_or_increment)
end

# Acceptance utilities are separate from the selector. The exact model/path is
# constructed AFTER the live numerical run; none is supplied to its decisions.
include("rayleigh_block_acceptance_support.jl")
end # module
