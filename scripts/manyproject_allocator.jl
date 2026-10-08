"""Exact many-project allocation for a COMMON resource menu.
The ordered fast path is used only after an exact increasing-differences check
on the active score rows. Otherwise a budgeted exact DP solves the SAME rule.
No floating score rounding, relaxed knapsack, greedy NI fill, or altered tie rule.
"""
module ManyProjectAllocator
using SHA
const Q=Rational{BigInt}
const POLICIES=(:WHITTLE_K,:WHITTLE_P,:LAGRANGIAN_K,:LAGRANGIAN_NI,:LAGRANGIAN_FC,:MYOPIC_K)
const REVISION="manyproject-exact-allocation-v2-dual-core"
struct AllocationLimit <: Exception
    message::String
end
Base.showerror(io::IO,e::AllocationLimit)=print(io,e.message)
struct Limits
    max_states::Int
    max_updates::Int
    seconds::Float64
end
Limits()=Limits(200000,20000000,30.0)
abstract type AbstractScores end
struct Scores{T<:Integer} <: AbstractScores
    values::Matrix{T}           # rows are (class,positive age); action zero is col 1
    denominator::BigInt
    rank::Vector{Int}           # descending full increment-vector order
    compatible::BitMatrix      # strict componentwise dominance or full equality
    max_population::Int
    encoding::String
end
function resource_encoding(c::Vector{Q})
    length(c)>=2 && c[1]==0 && all(diff(c).>0) || throw(ArgumentError("Strict common resource menu required."))
    den=foldl(lcm,(denominator(x) for x in c);init=BigInt(1))
    v=BigInt[numerator(x*den) for x in c];g=foldl(gcd,v;init=BigInt(0))
    units=div.(v,g)
    maximum(units)<=div(typemax(Int),8192) || error("Resource integer encoding exceeds saved arithmetic range.")
    Int.(units),g//den
end
function budget_units(B::Q,tick::Q,c::Vector{Int},L::Int)
    B>=0 && L>=0 || throw(ArgumentError("Nonnegative budget/population required."))
    v=fld(numerator(B/tick),denominator(B/tick))
    v<=div(typemax(Int),4) && BigInt(L)*maximum(c)<=div(typemax(Int),4) || error("Budget/population integer arithmetic overflow risk.")
    Int(v)
end
function compile_scores(values::Matrix{Q};max_population::Int=1600,max_bits::Int=524288)
    n,cols=size(values);n>0 && 2<=cols<=4 && max_population>=1 && all(iszero,values[:,1]) || throw(ArgumentError("Invalid complete score table."))
    den=foldl(lcm,(denominator(x) for x in values);init=BigInt(1))
    ndigits(den;base=2)<=max_bits || throw(AllocationLimit("Exact score-denominator bit budget exceeded."))
    v=BigInt[numerator(x*den) for x in values];v=reshape(v,size(values))
    bound=BigInt(32)*max_population*maximum(abs,v;init=BigInt(0))
    T=bound<=typemax(Int64) ? Int64 : (bound<=typemax(Int128) ? Int128 : BigInt)
    ndigits(max(bound,1);base=2)<=max_bits || throw(AllocationLimit("Exact score-numerator bit budget exceeded."))
    delta=[Tuple(v[i,a+1]-v[i,a] for a in 1:cols-1) for i in 1:n]
    ord=sortperm(1:n;by=i->delta[i],rev=true);rank=zeros(Int,n);r=0;prev=0
    for i in ord
        (prev==0 || delta[i]!=delta[prev]) && (r+=1)
        rank[i]=r;prev=i
    end
    comp=falses(n,n)
    for i in 1:n,j in 1:n
        comp[i,j]=delta[i]==delta[j] || all(delta[i][a]>delta[j][a] for a in eachindex(delta[i]))
    end
    Scores(T.(v),den,rank,comp,max_population,string(T))
end
mutable struct Workspace{T<:Integer}
    order::Vector{Int}
    actions::Vector{Int}
    prefix::Matrix{T}
    scratch::Vector{Int}
end
Workspace(s::Scores{T},L::Int) where T=Workspace{T}(Int[],zeros(Int,L),zeros(T,L+1,size(s.values,2)),zeros(Int,L))
function actions_from_counts!(dest,order,n1,n2,n3)
    fill!(dest,0);k3=n3;k2=n3+n2;k1=k2+n1
    for (k,i) in enumerate(order)
        dest[i]=k<=k3 ? 3 : (k<=k2 ? 2 : (k<=k1 ? 1 : 0))
    end
    dest
end
function lex_counts_less(w,counts,bestcounts)
    actions_from_counts!(w.scratch,w.order,counts...)
    actions_from_counts!(w.actions,w.order,bestcounts...)
    for i in eachindex(w.actions)
        w.scratch[i]==w.actions[i] && continue
        return w.scratch[i]<w.actions[i]
    end
    false
end
"""Return ordered solution statistics, or nothing if the exact order test fails.
For A=2, count search is O(L), following sorting and prefix construction. A=3
uses O(L^2) count search. The labels in a fully tied row group are descending,
so high gears go to later labels, preserving lexicographically least outputs.
"""
function ordered!(w::Workspace{T},s::Scores{T},types::Vector{Int},c::Vector{Int},C::Int,domain::Symbol) where T
    L=length(types);L==length(w.actions) && L<=s.max_population || throw(DimensionMismatch("Allocation workspace/population mismatch."))
    domain in (:k,:ni,:fc) || throw(ArgumentError("Unknown domain."))
    empty!(w.order)
    for i in eachindex(types)
        0<=types[i]<=size(s.values,1) || throw(ArgumentError("Invalid score-table row."))
        types[i]>0 && push!(w.order,i)
    end
    sort!(w.order;by=i->(s.rank[types[i]],-i))
    for j in 2:length(w.order)
        s.compatible[types[w.order[j-1]],types[w.order[j]]] || return nothing
    end
    n=length(w.order);A=length(c)-1
    A+1==size(s.values,2) || throw(DimensionMismatch("Resource/score gear count mismatch."))
    if n==0;fill!(w.actions,0);return (engine=:ordered_exact,work=0,peak_states=0,resource_units=0);end
    # Binary exact selection needs neither a count search nor BigInt prefixes.
    if A==1
        cap=min(n,div(C,c[2]));positive=count(i->s.values[types[i],2]>0,w.order)
        n1=domain==:k ? min(cap,positive) : cap
        actions_from_counts!(w.actions,w.order,n1,0,0)
        return (engine=:ordered_exact,work=n,peak_states=0,resource_units=n1*c[2])
    end
    for a in 1:A+1
        w.prefix[1,a]=zero(T)
        for k in 1:n;w.prefix[k+1,a]=w.prefix[k,a]+s.values[types[w.order[k]],a];end
    end
    positive=count(i->s.values[types[i],2]>0,w.order)
    have=false;bestscore=zero(T);bestspent=0;bestcounts=(0,0,0);work=0
    max3=A==3 ? min(n,div(C,c[4])) : 0
    for n3 in 0:max3
        fixed3=A==3 ? n3*c[4] : 0
        max2=min(n-n3,div(C-fixed3,c[3]))
        for n2 in 0:max2
            off=n3+n2;remaining=n-off;fixed=fixed3+n2*c[3]
            max1=min(remaining,div(C-fixed,c[2]))
            segments=domain==:ni ? 3 : 1
            for seg in 1:segments
                lo=0;hi=max1
                if domain==:fc
                    lo=hi
                elseif domain==:ni
                    if seg==1;lo=0;hi=min(max1,0)
                    elseif seg==2;lo=1;hi=min(max1,remaining-1)
                    else;lo=remaining;hi=min(max1,remaining)
                    end
                    lo>hi && continue
                    gap=0
                    if A==3 && n2>0;gap=c[4]-c[3];end
                    if lo>0
                        own=c[3]-c[2];gap=gap==0 ? own : min(gap,own)
                    end
                    if hi<remaining
                        own=c[2];gap=gap==0 ? own : min(gap,own)
                    end
                    gap>0 && (lo=max(lo,fld(C-fixed-gap,c[2])+1))
                end
                lo>hi && continue
                n1=domain==:fc ? hi : clamp(positive-off,lo,hi)
                value=w.prefix[off+n1+1,2]-w.prefix[off+1,2] + w.prefix[off+1,3]-w.prefix[n3+1,3]
                A==3 && (value+=w.prefix[n3+1,4])
                spent=fixed+n1*c[2];counts=(n1,n2,n3);work+=1
                better=!have
                if have
                    if domain==:fc && spent!=bestspent;better=spent>bestspent
                    elseif value!=bestscore;better=value>bestscore
                    elseif spent!=bestspent;better=spent<bestspent
                    else;better=lex_counts_less(w,counts,bestcounts)
                    end
                end
                if better;have=true;bestscore=value;bestspent=spent;bestcounts=counts;end
            end
        end
    end
    have || error("Empty ordered allocation domain.")
    actions_from_counts!(w.actions,w.order,bestcounts...)
    (engine=:ordered_exact,work=work,peak_states=0,resource_units=bestspent)
end
min_gap(x::Int,y::Int)=x==0 ? y : (y==0 ? x : min(x,y))
"""Exact DP on (resource units, minimum next-upgrade gap). Lexicographic paths
are encoded in base A+1 integers, not copied length-L vectors. No state pruning
by score tolerance or greedy capacity repair is permitted. Limits throw.
"""
function dp!(w::Workspace{T},s::Scores{T},types::Vector{Int},c::Vector{Int},C::Int,domain::Symbol,limits::Limits) where T
    A=length(c)-1;base=A+1;active=findall(>(0),types)
    states=Dict{Tuple{Int,Int},Tuple{T,BigInt}}((0,0)=>(zero(T),BigInt(0)))
    work=0;peak=1;started=time_ns()
    for i in active
        (time_ns()-started)/1e9<=limits.seconds || throw(AllocationLimit("Exact allocator cooperative time limit; no approximate action returned."))
        nxt=Dict{Tuple{Int,Int},Tuple{T,BigInt}}()
        for ((spent,gap),(score,code)) in states,a in 0:A
            work+=1;work<=limits.max_updates || throw(AllocationLimit("Exact allocator update limit; no action returned."))
            cost=spent+c[a+1];cost>C && continue
            own=a<A ? c[a+2]-c[a+1] : 0
            ng=domain==:ni ? min_gap(gap,own) : 0;key=(cost,ng)
            value=score+s.values[types[i],a+1];path=code*base+a
            prev=get(nxt,key,nothing)
            if prev===nothing || value>prev[1] || (value==prev[1] && path<prev[2]);nxt[key]=(value,path);end
            length(nxt)<=limits.max_states || throw(AllocationLimit("Exact allocator state limit; no truncation or greedy substitute."))
        end
        isempty(nxt) && error("No hard-feasible DP state.")
        peak=max(peak,length(nxt));states=nxt
    end
    have=false;bs=zero(T);bc=0;bp=BigInt(0)
    for ((cost,gap),(score,path)) in states
        domain==:ni && gap>0 && C-cost>=gap && continue
        better=!have
        if have
            if domain==:fc && cost!=bc;better=cost>bc
            elseif score!=bs;better=score>bs
            elseif cost!=bc;better=cost<bc
            else;better=path<bp
            end
        end
        if better;have=true;bs=score;bc=cost;bp=path;end
    end
    have || error("Empty capacity-domain DP.")
    fill!(w.actions,0);path=bp
    for i in Iterators.reverse(active)
        w.actions[i]=Int(rem(path,base));path=div(path,base)
    end
    path==0 || error("DP backtrace overflow.")
    (engine=:integer_dp_exact,work=work,peak_states=peak,resource_units=bc)
end
# A checked exact refinement of the ordinary (:k) three-gear knapsack fallback.
# The original DP below remains the reference and the route for unsupported rows.
mutable struct CoreWorkBudget
    limits::Limits
    work::Int
    started::UInt64
end
function core_tick!(b::CoreWorkBudget,n::Int=1)
    b.work+=n
    b.work<=b.limits.max_updates || throw(AllocationLimit("Exact dual-core allocator update limit; no action returned."))
    if (b.work & 1023)==0
        core_time_check(b)
    end
    nothing
end
function core_time_check(b::CoreWorkBudget)
    (time_ns()-b.started)/1e9<=b.limits.seconds || throw(AllocationLimit("Exact dual-core allocator cooperative time limit; no approximate action returned."))
    nothing
end

"""Solve the SAME ordinary knapsack with a common three-gear menu, when its
present rows have strictly positive, nonincreasing per-resource increments.

Repeated rows are aggregated only to build an exact Lagrangian upper bound and
a feasible incumbent. An action is excluded only when its individual reduced
loss is greater than the entire upper-bound/incumbent gap. The residual DP
retains all optimal actions, including ties, with the original label ordering.
Nothing is rounded. A greedy incumbent is never returned as a heuristic answer.

Returns nothing (no workspace mutation) outside this structural scope. Work,
state and time limits are the same Limits object used by the old exact DP.
"""
function dual_core!(w::Workspace{T},s::Scores{T},types::Vector{Int},c::Vector{Int},C::Int,
                    domain::Symbol,limits::Limits; evidence=nothing) where T
    domain==:k && length(c)==3 || return nothing
    b=CoreWorkBudget(limits,0,time_ns())
    counts=Dict{Int,Int}()
    for t in types
        core_tick!(b)
        t>0 && (counts[t]=get(counts,t,0)+1)
    end
    isempty(counts) && return nothing # handled by the existing ordered route
    rows=sort!(collect(keys(counts)))
    dc=(c[2],c[3]-c[2])
    delta=Dict{Int,NTuple{2,BigInt}}()
    for t in rows
        d1=BigInt(s.values[t,2]);d2=BigInt(s.values[t,3])-d1
        d1>0 && d2>0 && d1*dc[2]>=d2*dc[1] || return nothing
        delta[t]=(d1,d2);core_tick!(b,3)
    end
    # Events aggregate identical score rows. Exact density comparison; tied
    # first increments precede second increments. This is only an incumbent.
    events=Tuple{Int,Int}[(t,a) for t in rows for a in 1:2]
    function event_lt(x,y)
        core_tick!(b)
        lhs=delta[x[1]][x[2]]*dc[y[2]]
        rhs=delta[y[1]][y[2]]*dc[x[2]]
        lhs!=rhs ? lhs>rhs : isless((x[2],x[1]),(y[2],y[1]))
    end
    sort!(events;lt=event_lt)
    inventory=Dict(t=>[counts[t],0,0] for t in rows)
    spent=0;incumbent=BigInt(0);price_num=BigInt(0);price_den=1;price_set=false
    for (t,a) in events
        core_tick!(b)
        available=inventory[t][a]
        take=min(available,div(C-spent,dc[a]))
        if take<available && !price_set
            price_num=delta[t][a];price_den=dc[a];price_set=true
        end
        inventory[t][a]-=take;inventory[t][a+1]+=take
        spent+=take*dc[a];incumbent+=take*delta[t][a]
    end
    0<=spent<=C || error("Invalid feasible incumbent in exact dual-core allocator.")
    core_time_check(b)
    # D * upper_bound = z*C + sum_i max_a(D*v_ia-z*c_a), with D>0,z>=0.
    maxima=Dict{Int,BigInt}();losses=Dict{Int,Vector{BigInt}}()
    upper=price_num*C
    for t in rows
        v=BigInt[price_den*BigInt(s.values[t,a])-price_num*c[a] for a in 1:3]
        z=maximum(v);maxima[t]=z;losses[t]=BigInt[z-x for x in v]
        upper+=counts[t]*z;core_tick!(b,3)
    end
    gap=upper-price_den*incumbent
    gap>=0 || error("Exact Lagrangian bound below feasible incumbent.")
    allowed=Dict{Int,Vector{Int}}();minimum_gear=Dict{Int,Int}()
    base_spent=0;base_score=zero(T);excluded=0
    for t in rows
        menu=Int[a for a in 0:2 if losses[t][a+1]<=gap]
        isempty(menu) && error("Exact screening incorrectly removed every action.")
        allowed[t]=menu;minimum_gear[t]=first(menu);excluded+=3-length(menu)
        base_spent+=counts[t]*c[first(menu)+1]
        base_score+=counts[t]*s.values[t,first(menu)+1]
        core_tick!(b,3)
    end
    base_spent<=C || error("Exact screening removed the feasible incumbent.")
    capacity=C-base_spent
    active=Int[i for i in eachindex(types) if types[i]>0]
    core=Int[i for i in active if length(allowed[types[i]])>1]
    fixed_loss=sum((losses[types[i]][minimum_gear[types[i]]+1] for i in active
                    if length(allowed[types[i]])==1);init=BigInt(0))
    fixed_loss<=gap || error("Screened singleton actions exceed the exact loss bound.")
    states=Dict{Int,Tuple{T,BigInt,BigInt}}(0=>(zero(T),BigInt(0),BigInt(0)))
    peak=1;dp_updates=0
    for i in core # original label order: base-three code is exactly the old tie rule
        core_time_check(b)
        t=types[i];firstgear=minimum_gear[t]
        nxt=Dict{Int,Tuple{T,BigInt,BigInt}}()
        for (used,(score,code,loss)) in states,a in allowed[t]
            core_tick!(b);dp_updates+=1
            cost=used+c[a+1]-c[firstgear+1]
            cost>capacity && continue
            newloss=loss+losses[t][a+1]
            newloss+fixed_loss>gap && continue
            value=score+(s.values[t,a+1]-s.values[t,firstgear+1])
            path=code*3+a
            prev=get(nxt,cost,nothing)
            if prev===nothing || value>prev[1] || (value==prev[1] && path<prev[2])
                nxt[cost]=(value,path,newloss)
            end
            length(nxt)<=limits.max_states || throw(AllocationLimit("Exact dual-core allocator state limit; no truncation or substitute."))
        end
        isempty(nxt) && error("Exact loss-pruned DP lost every feasible incumbent prefix.")
        peak=max(peak,length(nxt));states=nxt
    end
    have=false;bestscore=zero(T);bestcost=0;bestcode=BigInt(0)
    for (used,(score,code,loss)) in states
        core_tick!(b)
        if !have || score>bestscore || (score==bestscore && (used<bestcost || (used==bestcost && code<bestcode)))
            have=true;bestscore=score;bestcost=used;bestcode=code
        end
    end
    have && BigInt(base_score)+BigInt(bestscore)>=incumbent || error("Exact core result below its feasible incumbent.")
    action=zeros(Int,length(types))
    for i in active;action[i]=minimum_gear[types[i]];end
    path=bestcode
    for i in Iterators.reverse(core);action[i]=Int(rem(path,3));path=div(path,3);end
    path==0 || error("Exact core backtrace overflow.")
    total=base_spent+bestcost
    sum(c[a+1] for a in action)==total<=C || error("Exact core resource reconciliation failed.")
    score=sum((s.values[types[i],action[i]+1] for i in active);init=zero(T))
    score==base_score+bestscore || error("Exact core score reconciliation failed.")
    core_time_check(b)
    if evidence!==nothing
        merge!(evidence,Dict{String,Any}("engine"=>"dual_core_dp_exact","rows"=>length(rows),
            "core_projects"=>length(core),"fixed_projects"=>length(active)-length(core),
            "core_rows"=>count(t->length(allowed[t])>1,rows),"excluded_row_actions"=>excluded,
            "price_numerator"=>string(price_num),"price_denominator"=>price_den,
            "scaled_upper_bound"=>string(upper),"scaled_gap"=>string(gap),
            "incumbent_score_units"=>string(incumbent),"incumbent_resource_units"=>spent,
            "optimal_score_units"=>string(score),"resource_units"=>total,
            "residual_capacity"=>capacity,"work"=>b.work,"dp_updates"=>dp_updates,"peak_states"=>peak,
            "allowed_row_actions"=>[Dict("row"=>t,"count"=>counts[t],"actions"=>allowed[t]) for t in rows],
            "screening_is_exact"=>true,"greedy_incumbent_is_not_the_policy"=>true))
    end
    copyto!(w.actions,action) # no returned allocation is changed before success
    (engine=:dual_core_dp_exact,work=b.work,peak_states=peak,resource_units=total)
end

function allocate!(w::Workspace{T},s::Scores{T},types::Vector{Int},c::Vector{Int},C::Int,domain::Symbol;limits=Limits(),force_dp=false) where T
    C>=0 && length(c)>=2 && c[1]==0 && all(diff(c).>0) || throw(ArgumentError("Invalid budget/menu."))
    domain in (:k,:ni,:fc) || throw(ArgumentError("Unknown domain."))
    length(c)==size(s.values,2) || throw(DimensionMismatch("Resource/score gear count mismatch."))
    BigInt(length(types))*maximum(c)<=div(typemax(Int),4) && C<=div(typemax(Int),4) || throw(ArgumentError("Allocation resource overflow risk."))
    limits.max_states>=1 && limits.max_updates>=1 && limits.seconds>0 || throw(ArgumentError("Invalid allocator budgets."))
    length(types)<=s.max_population && length(types)==length(w.actions) || throw(DimensionMismatch("Population mismatch."))
    all(t->0<=t<=size(s.values,1),types) || throw(ArgumentError("Invalid type."))
    r=force_dp ? nothing : ordered!(w,s,types,c,C,domain)
    r===nothing && !force_dp && (r=dual_core!(w,s,types,c,C,domain,limits))
    r===nothing && (r=dp!(w,s,types,c,C,domain,limits))
    sum(c[a+1] for a in w.actions)==r.resource_units<=C || error("Returned allocation violates budget.")
    all(types[i]>0 || w.actions[i]==0 for i in eachindex(types)) || error("Synchronized project received nonzero gear.")
    if domain==:ni
        left=C-r.resource_units
        all(types[i]==0 || w.actions[i]==length(c)-1 || left<c[w.actions[i]+2]-c[w.actions[i]+1] for i in eachindex(types)) || error("NI result has a feasible componentwise upgrade.")
    end
    r
end
function allocate(s::Scores,types::Vector{Int},c::Vector{Int},C::Int,domain::Symbol;kwargs...)
    w=Workspace(s,length(types));r=allocate!(w,s,types,c,C,domain;kwargs...)
    merge(r,(actions=copy(w.actions),))
end
"""Maximum attainable resource for N eligible identical menus; independent of
scores. Used for diagnostics, not substituted for the NI domain. O(N^(A-1))."""
function maximum_spend(c::Vector{Int},N::Int,C::Int)
    A=length(c)-1;1<=A<=3 && N>=0 && C>=0 || throw(ArgumentError("Unsupported spend inventory."))
    A==1 && return min(N,div(C,c[2]))*c[2]
    best=0
    for n3 in 0:(A==3 ? min(N,div(C,c[4])) : 0)
        r3=A==3 ? n3*c[4] : 0
        for n2 in 0:min(N-n3,div(C-r3,c[3]))
            spent=r3+n2*c[3];n1=min(N-n3-n2,div(C-spent,c[2]))
            best=max(best,spent+n1*c[2])
        end
    end
    best
end
# Heap nodes use the exact precomputed index rank and then the original project
# label. Only exposed adjacent increments are inserted, preserving precedence.
function heappush!(heap,node)
    push!(heap,node);i=length(heap)
    while i>1
        p=div(i,2);isless(heap[i],heap[p]) || break
        heap[i],heap[p]=heap[p],heap[i];i=p
    end
end
function heappop!(heap)
    first=heap[1];last=pop!(heap);isempty(heap) && return first
    heap[1]=last;i=1
    while 2i<=length(heap)
        j=2i
        j<length(heap) && isless(heap[j+1],heap[j]) && (j+=1)
        isless(heap[j],heap[i]) || break
        heap[i],heap[j]=heap[j],heap[i];i=j
    end
    first
end
function priority!(actions::Vector{Int},heap::Vector{NTuple{3,Int}},types::Vector{Int},rank::Matrix{Int},positive::BitMatrix,c::Vector{Int},C::Int)
    A=length(c)-1;empty!(heap);spent=0;work=0
    for i in eachindex(types)
        t=types[i];actions[i]=t==0 ? 0 : A
        if t>0;spent+=c[A+1];heappush!(heap,(rank[t,A],i,A));end
    end
    while !isempty(heap)
        _,i,a=heap[1];t=types[i]
        spent<=C && positive[t,a] && break
        heappop!(heap);actions[i]-=1;spent-=c[a+1]-c[a];work+=1
        a>1 && heappush!(heap,(rank[t,a-1],i,a-1))
    end
    spent<=C || error("Whittle priority infeasible.")
    (engine=:exposed_priority_exact,work=work,peak_states=length(heap),resource_units=spent)
end
end # module
