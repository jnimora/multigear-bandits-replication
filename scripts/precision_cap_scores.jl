"""Cap-stage exact scores WITHOUT a table-wide rational common denominator.
Ordered count candidates are screened with certified enclosures. Only overlapping
candidate scores are compared exactly, cancelling unchanged rows first. No score
rounding, approximate ties, greedy NI completion, or priority substitution.
This is a separately gated interface, not a change to the accepted allocator.
"""
module PrecisionCapScores
using ..RayleighBlock, ..ManyProjectAllocator
const RB=RayleighBlock; const MA=ManyProjectAllocator
const Q=Rational{BigInt}; const E=RB.Enclosed
struct Table
    values::Matrix{Q}
    delta::Matrix{Q}
    bounds::Matrix{E}
    delta_bounds::Matrix{E}
    rank::Vector{Int}
    max_population::Int
end
function compile(v::Matrix{Q};max_population=1600)
    RB.ieee_check()
    n,cols=size(v)
    n>0 && cols in (2,3) && max_population>0 && all(iszero,v[:,1]) || error("Direct cap score table shape.")
    d=diff(v;dims=2);eb=E.(v);db=E.(d)
    function row_less(i,j)
        for a in 1:cols-1
            db[i,a].lo>db[j,a].hi && return true
            db[i,a].hi<db[j,a].lo && return false
            d[i,a]!=d[j,a] && return d[i,a]>d[j,a]
        end
        i<j
    end
    perm=sortperm(1:n;lt=row_less);rank=zeros(Int,n);k=0;previous=0
    for i in perm
        if previous==0 || any(d[i,a]!=d[previous,a] for a in 1:cols-1);k+=1;end
        rank[i]=k;previous=i
    end
    Table(v,d,eb,db,rank,max_population)
end
mutable struct Workspace
    order::Vector{Int}
    actions::Vector{Int}
    scratch::Vector{Int}
    prefix::Matrix{E}
end
Workspace(t::Table,L::Int)=Workspace(Int[],zeros(Int,L),zeros(Int,L),fill(RB.BZERO,L+1,size(t.values,2)))
mutable struct Budget
    limits::MA.Limits
    started::UInt64
    work::Int
    exact::Int
    max_exact_bits::Int
end
Budget(l::MA.Limits;max_exact_bits=524288)=Budget(l,time_ns(),0,0,max_exact_bits)
function tick!(b::Budget,n=1)
    b.work+=n
    b.work<=b.limits.max_updates || throw(MA.AllocationLimit("Cap score work limit; no approximate allocation returned."))
    (time_ns()-b.started)/1e9<=b.limits.seconds || throw(MA.AllocationLimit("Cap score cooperative time limit; no approximate allocation returned."))
end
function bounded(q::Q,b::Budget)
    max(ndigits(abs(numerator(q));base=2),ndigits(denominator(q);base=2))<=b.max_exact_bits ||
        throw(MA.AllocationLimit("Selective exact-score bit limit; no approximate tie or action returned."))
    q
end
function dominates(t::Table,i,j)
    t.rank[i]==t.rank[j] && return true
    for a in axes(t.delta,2)
        t.delta_bounds[i,a].lo>t.delta_bounds[j,a].hi && continue
        t.delta[i,a]>t.delta[j,a] || return false
    end
    true
end
gear_at(k,n1,n2)=k<=n2 ? 2 : (k<=n2+n1 ? 1 : 0)
"""Exact candidate score DIFFERENCE after cancelling all unchanged project rows.
A dictionary aggregates only changed row/action coefficients. An ambiguous interval
is not assumed to be a tie. The exact arithmetic has an explicit bit/work budget.
"""
function exact_difference(t,w,types,x,y,b::Budget)
    b.exact+=1
    terms=Dict{Tuple{Int,Int},Int}()
    for (k,i) in enumerate(w.order)
        a=gear_at(k,x...);z=gear_at(k,y...);a==z && continue
        tick!(b)
        a>0 && (key=(types[i],a+1);terms[key]=get(terms,key,0)+1)
        z>0 && (key=(types[i],z+1);terms[key]=get(terms,key,0)-1)
    end
    value=zero(Q)
    for key in sort!(collect(keys(terms)))
        n=terms[key];n==0 && continue
        tick!(b);value=bounded(value+n*t.values[key...],b)
    end
    value
end
function lex_less(w,x,y)
    MA.actions_from_counts!(w.scratch,w.order,x[1],x[2],0)
    MA.actions_from_counts!(w.actions,w.order,y[1],y[2],0)
    for i in eachindex(w.actions)
        w.scratch[i]==w.actions[i] || return w.scratch[i]<w.actions[i]
    end
    false
end
function ordered!(w::Workspace,t::Table,types,c,C,domain,b::Budget)
    empty!(w.order)
    for i in eachindex(types);types[i]>0 && push!(w.order,i);end
    sort!(w.order;by=i->(t.rank[types[i]],-i))
    for j in 2:length(w.order)
        tick!(b)
        dominates(t,types[w.order[j-1]],types[w.order[j]]) || return nothing
    end
    n=length(w.order);A=length(c)-1
    if n==0
        fill!(w.actions,0)
        return (engine=:cap_ordered_verified,work=b.work,peak_states=0,resource_units=0,exact_comparisons=0)
    end
    if A==1
        positive=count(i->t.values[types[i],2]>0,w.order)
        n1=min(n,div(C,c[2]));domain==:k && (n1=min(n1,positive))
        MA.actions_from_counts!(w.actions,w.order,n1,0,0)
        return (engine=:cap_ordered_verified,work=b.work+n,peak_states=0,resource_units=n1*c[2],exact_comparisons=0)
    end
    for a in 1:3
        w.prefix[1,a]=RB.BZERO
        for k in 1:n
            w.prefix[k+1,a]=w.prefix[k,a]+t.bounds[types[w.order[k]],a];tick!(b)
        end
    end
    positive=count(i->t.values[types[i],2]>0,w.order)
    have=false;best=(0,0);best_spent=0;best_value=RB.BZERO;candidate_count=0
    for n2 in 0:min(n,div(C,c[3]))
        remaining=n-n2;fixed=n2*c[3];max1=min(remaining,div(C-fixed,c[2]))
        for seg in 1:(domain==:ni ? 3 : 1)
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
                lo>0 && (gap=c[3]-c[2])
                hi<remaining && (gap=MA.min_gap(gap,c[2]))
                gap>0 && (lo=max(lo,fld(C-fixed-gap,c[2])+1))
            end
            lo>hi && continue
            n1=domain==:fc ? hi : clamp(positive-n2,lo,hi)
            counts=(n1,n2);spent=fixed+n1*c[2]
            value=w.prefix[n2+n1+1,2]-w.prefix[n2+1,2]+w.prefix[n2+1,3]
            tick!(b);candidate_count+=1
            better=!have
            if have
                if domain==:fc && spent!=best_spent
                    better=spent>best_spent
                elseif value.lo>best_value.hi
                    better=true
                elseif value.hi<best_value.lo
                    better=false
                else
                    difference=exact_difference(t,w,types,counts,best,b)
                    better=difference>0 || (difference==0 &&
                        (spent<best_spent || (spent==best_spent && lex_less(w,counts,best))))
                end
            end
            if better;have=true;best=counts;best_spent=spent;best_value=value;end
        end
    end
    have || error("No allocation in cap-score domain.")
    MA.actions_from_counts!(w.actions,w.order,best[1],best[2],0)
    (engine=:cap_ordered_verified,work=b.work,peak_states=candidate_count,resource_units=best_spent,exact_comparisons=b.exact)
end

"""An exact comparison in the individual-row route. Operands/results are bounded
in normalized rational representation, as in the existing exact_difference route.
The bit gate is not a bound on temporary GMP products or a resident-memory limit.
"""
function row_compare(x::Q,y::Q,b::Budget)
    tick!(b);b.exact+=1
    bounded(x,b);bounded(y,b)
    x<y ? -1 : (x>y ? 1 : 0)
end

"""Exact reduced-loss screening followed by a residual spend-DP, WITHOUT an
active-table common denominator. Admitted only for ordinary K with A=2 and
positive nonincreasing marginal score/resource ratios. Other domains/menus keep
the existing checked local encoding and exact optimizer. The incumbent is only
a bound; it is never substituted for the exact policy. Optional evidence is for
acceptance, not used by the selector.
"""
function row_core!(w::Workspace,t::Table,types::Vector{Int},c::Vector{Int},C::Int,
                   domain::Symbol,b::Budget;evidence=nothing,filter_threshold::Int=32)
    domain===:k && length(c)==3 || return nothing
    rows=sort!(unique(filter(>(0),types)))
    isempty(rows) && return nothing
    dc=(c[2],c[3]-c[2]);nr=length(rows)
    mapping=Dict(r=>j for (j,r) in enumerate(rows))
    counts=zeros(Int,nr)
    for r in types
        tick!(b)
        r>0 && (counts[mapping[r]]+=1)
    end
    # Check the sufficient precedence condition exactly. An unsupported input
    # returns to the old route; a resource-budget refusal is never swallowed.
    for r in rows
        for a in 1:3;tick!(b);bounded(t.values[r,a],b);end
        d1=bounded(t.delta[r,1],b);d2=bounded(t.delta[r,2],b)
        d1>0 && d2>0 || return nothing
        row_compare(bounded(d1*dc[2],b),bounded(d2*dc[1],b),b)>=0 || return nothing
    end
    events=Tuple{Int,Int}[(j,a) for j in 1:nr for a in 1:2]
    function before(x,y)
        j,a=x;k,z=y
        r=row_compare(bounded(t.delta[rows[j],a]*dc[z],b),
                      bounded(t.delta[rows[k],z]*dc[a],b),b)
        r==0 ? isless((a,rows[j]),(z,rows[k])) : r>0
    end
    sort!(events;lt=before)
    incumbent=zeros(Int,nr,3);incumbent[:,1].=counts
    spent=0;mu=zero(Q);price_set=false
    for (j,a) in events
        tick!(b)
        n=incumbent[j,a];take=min(n,div(C-spent,dc[a]))
        if take<n && !price_set
            mu=bounded(t.delta[rows[j],a]/dc[a],b);price_set=true
        end
        incumbent[j,a]-=take;incumbent[j,a+1]+=take;spent+=take*dc[a]
    end
    0<=spent<=C && mu>=0 || error("Invalid row-core incumbent.")
    # U - incumbent is formed from NONNEGATIVE losses, not by subtracting two
    # global scores. Scores of projects already maximizing the priced objective
    # cancel before any rational sums are taken.
    losses=Matrix{Q}(undef,nr,3)
    gap=bounded(mu*(C-spent),b)
    for j in 1:nr
        z=ntuple(a->bounded(t.values[rows[j],a]-bounded(mu*c[a],b),b),3)
        best=z[1]
        for a in 2:3;row_compare(z[a],best,b)>0 && (best=z[a]);end
        for a in 1:3
            tick!(b);loss=bounded(best-z[a],b)
            loss>=0 || error("Negative exact reduced loss.")
            losses[j,a]=loss
            if incumbent[j,a]>0 && loss!=0
                gap=bounded(gap+bounded(incumbent[j,a]*loss,b),b)
            end
        end
    end
    menus=Vector{Vector{Int}}(undef,nr)
    for j in 1:nr
        menus[j]=Int[]
        for a in 0:2
            row_compare(losses[j,a+1],gap,b)<=0 && push!(menus[j],a)
        end
        isempty(menus[j]) && error("Screen excluded every action.")
        for a in 0:2
            incumbent[j,a+1]>0 && !(a in menus[j]) && error("Incumbent lost by exact screen.")
        end
    end
    basegear=first.(menus);basecost=sum(counts[j]*c[basegear[j]+1] for j in 1:nr)
    basecost<=spent<=C || error("Invalid residual baseline.")
    fixedloss=zero(Q)
    for j in 1:nr
        tick!(b)
        if length(menus[j])==1
            fixedloss=bounded(fixedloss+bounded(counts[j]*losses[j,basegear[j]+1],b),b)
        end
    end
    coregap=bounded(gap-fixedloss,b);coregap>=0 || error("Invalid fixed loss.")
    core=Int[i for i in eachindex(types) if types[i]>0 && length(menus[mapping[types[i]]])>1]
    filter_threshold>=0 || error("Negative filtered-core threshold.")
    if length(core)>=filter_threshold && !isempty(core)
        return filtered_row_dp!(w,t,types,c,C,b,rows,counts,mapping,incumbent,spent,
            mu,gap,losses,menus,basegear,basecost,fixedloss,coregap,core;evidence=evidence)
    end
    # At a fixed spend, maximizing score equals minimizing total reduced loss.
    # The base-3 code uses ORIGINAL increasing project-label order, so exact ties
    # preserve the lexicographically least FULL vector; all other actions are fixed.
    states=Dict{Int,Tuple{Q,BigInt}}(0=>(zero(Q),BigInt(0)))
    peak=1;extensions=0;remaining=C-basecost
    for i in core
        j=mapping[types[i]];bg=basegear[j]
        next=Dict{Int,Tuple{Q,BigInt}}()
        # Sorting integer keys makes work and evidence independent of Dict order.
        for cost in sort!(collect(keys(states)))
            ls,code=states[cost]
            for a in menus[j]
                tick!(b);extensions+=1
                nc=cost+c[a+1]-c[bg+1];nc<=remaining || continue
                nl=bounded(ls+losses[j,a+1],b)
                row_compare(nl,coregap,b)<=0 || continue
                ncode=code*3+a
                previous=get(next,nc,nothing)
                take=previous===nothing
                if !take
                    cmp=row_compare(nl,previous[1],b)
                    take=cmp<0 || (cmp==0 && ncode<previous[2])
                end
                if take
                    if previous===nothing && length(next)>=b.limits.max_states
                        throw(MA.AllocationLimit("Individual-row core state limit; no approximate action returned."))
                    end
                    next[nc]=(nl,ncode)
                end
            end
        end
        isempty(next) && error("Exact core lost the feasible incumbent.")
        peak=max(peak,length(next));states=next
    end
    have=false;bestkey=zero(Q);bestcost=0;bestcode=BigInt(0);bestloss=zero(Q)
    for cost in sort!(collect(keys(states)))
        tick!(b);ls,code=states[cost]
        key=bounded(ls-bounded(mu*cost,b),b)
        cmp=have ? row_compare(key,bestkey,b) : -1
        if !have || cmp<0 || (cmp==0 && (cost<bestcost || (cost==bestcost && code<bestcode)))
            have=true;bestkey=key;bestcost=cost;bestcode=code;bestloss=ls
        end
    end
    have || error("No exact row-core solution.")
    # Build into private storage. A refusal never publishes a partial action.
    action=zeros(Int,length(types))
    for i in eachindex(types);types[i]>0 && (action[i]=basegear[mapping[types[i]]]);end
    code=bestcode
    for i in Iterators.reverse(core)
        code,a=divrem(code,3);action[i]=Int(a)
    end
    code==0 || error("Residual lexicographic code mismatch.")
    resource=basecost+bestcost
    sum(c[a+1] for a in action)==resource<=C || error("Row-core budget mismatch.")
    deficit=bounded(bounded(mu*(C-resource),b)+bounded(fixedloss+bestloss,b),b)
    0<=deficit<=gap || error("Exact solution below the feasible incumbent.")
    tick!(b)
    if evidence!==nothing
        evidence["rows"]=copy(rows);evidence["counts"]=copy(counts)
        evidence["mu"]=mu;evidence["gap"]=gap;evidence["losses"]=losses
        evidence["incumbent_counts"]=incumbent;evidence["incumbent_resource_units"]=spent
        evidence["menus"]=deepcopy(menus);evidence["core_labels"]=copy(core)
        evidence["actions"]=copy(action);evidence["resource_units"]=resource
        evidence["upper_minus_score"]=deficit;evidence["dp_extensions"]=extensions
        evidence["peak_states"]=peak;evidence["active_common_denominator_constructed"]=false
    end
    copyto!(w.actions,action)
    # Existing accounting recognizes cap_active_exact and dual_core_dp_exact.
    # The new field distinguishes the representation without changing a simulator.
    (engine=:cap_active_exact,underlying_engine=:dual_core_dp_exact,
     score_interface=:individual_rational_core_v1,work=b.work,peak_states=peak,
     resource_units=resource,exact_comparisons=b.exact,core_projects=length(core),
     dp_extensions=extensions)
end

# Exact, absolute dyadic enclosures. Only arithmetic comparisons are filtered:
# the objective is NEVER rounded to these endpoints. BigInt additions have no
# overflow; an overlap invokes the original exact-rational/label comparison.
struct RowDyadicState
    lo::BigInt
    hi::BigInt
    code::BigInt
end
function row_dyadic(q::Q,shift::Int,b::Budget)
    tick!(b);bounded(q,b)
    if shift>=0
        n=numerator(q)<<shift;d=denominator(q)
    else
        n=numerator(q);d=denominator(q)<<(-shift)
    end
    lo,rem=fldmod(n,d)
    hi=rem==0 ? lo : lo+1
    # Endpoints use one common power-of-two SCALE, not a score-denominator LCM.
    # The shifted intermediate is temporary, as with existing rational products.
    lo,hi
end

"""Subtract two equal-depth labelled paths after cancelling identical actions
and repeated row/action coefficients. `other=nothing` requests a single path
sum (needed only for an overlapping pruning/postcondition check or evidence).
Never interpret a numerical overlap as an exact tie.
"""
function row_path_difference(losses,mapping,types,core,depth::Int,
                             code::BigInt,other,b::Budget)
    b.exact+=1
    terms=Dict{Tuple{Int,Int},Int}()
    x=code;y=other===nothing ? BigInt(0) : other
    for k in depth:-1:1
        tick!(b)
        x,a=divrem(x,3)
        if other===nothing
            key=(mapping[types[core[k]]],Int(a)+1)
            terms[key]=get(terms,key,0)+1
        else
            y,z=divrem(y,3)
            a==z && continue
            j=mapping[types[core[k]]]
            key=(j,Int(a)+1);terms[key]=get(terms,key,0)+1
            key=(j,Int(z)+1);terms[key]=get(terms,key,0)-1
        end
    end
    x==0 && (other===nothing || y==0) || error("Filtered exact path length mismatch.")
    total=zero(Q)
    for key in sort!(collect(keys(terms)))
        n=terms[key];n==0 && continue
        tick!(b)
        total=bounded(total+bounded(n*losses[key...],b),b)
    end
    total
end

"""Policy-preserving residual DP with rigorous dyadic bounds on partial losses.
The previously accepted exact screen is unchanged. The per-spend best label code
still determines the complete original-label tie rule. No large rational partial
sum is made on a separated comparison; overlap uses exact cancelled differences.
Small cores use the historical implementation by default; acceptance can exercise
this routine on small fixtures through row_core!(...;filter_threshold=0).
"""
function bigint_filtered_row_dp!(w,t,types,c,C,b::Budget,rows,counts,mapping,incumbent,spent,
        mu,gap,losses,menus,basegear,basecost,fixedloss,coregap,core;evidence=nothing)
    n=length(core);remaining=C-basecost
    n>0 && remaining>=0 || error("Invalid filtered residual problem.")
    # Reserve log2(n) accumulation headroom inside the existing score bit budget.
    # Typical models use 4096 fractional bits. Extreme large magnitudes get a
    # coarser absolute scale (possibly a negative shift) and exact refinement.
    largest=maximum(losses)
    mag(q)=max(0,ndigits(abs(numerator(q));base=2)-ndigits(denominator(q);base=2)+1)
    magnitude=max(mag(largest)+ndigits(n+1;base=2),mag(gap),
                  mag(mu)+ndigits(C+1;base=2),mag(fixedloss))
    shift=min(4096,b.max_exact_bits-8-magnitude)
    abs(shift)<=b.max_exact_bits || throw(MA.AllocationLimit("Dyadic score-scale bit budget; no approximate action returned."))
    lows=Matrix{BigInt}(undef,size(losses));highs=similar(lows)
    for j in axes(losses,1),a in 1:3
        lows[j,a],highs[j,a]=row_dyadic(losses[j,a],shift,b)
    end
    gaplo,gaphi=row_dyadic(coregap,shift,b)
    mulo,muhi=row_dyadic(mu,shift,b)
    states=Dict{Int,RowDyadicState}(0=>RowDyadicState(BigInt(0),BigInt(0),BigInt(0)))
    peak=1;extensions=0;separations=0;refinements=0;ties=0;prune_refinements=0
    for (depth,i) in enumerate(core)
        j=mapping[types[i]];bg=basegear[j]
        next=Dict{Int,RowDyadicState}()
        for cost in sort!(collect(keys(states)))
            old=states[cost]
            for a in menus[j]
                tick!(b);extensions+=1
                nc=cost+c[a+1]-c[bg+1];nc<=remaining || continue
                nl=old.lo+lows[j,a+1];nh=old.hi+highs[j,a+1]
                # The true partial loss lies in [nl,nh]/2^shift. Losses are
                # nonnegative, so exceeding the total loss allowance is safe.
                tick!(b)
                nl>gaphi && (separations+=1;continue)
                ncode=nothing
                if nh>gaplo
                    prune_refinements+=1
                    ncode=old.code*3+a
                    q=row_path_difference(losses,mapping,types,core,depth,ncode,nothing,b)
                    q>coregap && continue
                else
                    separations+=1
                end
                prev=get(next,nc,nothing);take=prev===nothing
                if prev!==nothing
                    tick!(b)
                    if nh<prev.lo
                        take=true;separations+=1
                    elseif nl>prev.hi
                        take=false;separations+=1
                    else
                        refinements+=1
                        ncode===nothing && (ncode=old.code*3+a)
                        q=row_path_difference(losses,mapping,types,core,depth,ncode,prev.code,b)
                        ties+=q==0
                        take=q<0 || (q==0 && ncode<prev.code)
                    end
                end
                if take
                    prev===nothing && length(next)>=b.limits.max_states &&
                        throw(MA.AllocationLimit("Filtered individual-row core state limit; no approximate action returned."))
                    ncode===nothing && (ncode=old.code*3+a)
                    next[nc]=RowDyadicState(nl,nh,ncode)
                end
            end
        end
        isempty(next) && error("Filtered exact core lost feasible incumbent.")
        peak=max(peak,length(next));states=next
    end
    best=nothing;bestcost=0;bestlo=BigInt(0);besthi=BigInt(0)
    for cost in sort!(collect(keys(states)))
        tick!(b);st=states[cost]
        lo=st.lo-muhi*cost;hi=st.hi-mulo*cost
        take=best===nothing
        if best!==nothing
            if hi<bestlo
                take=true;separations+=1
            elseif lo>besthi
                take=false;separations+=1
            else
                refinements+=1
                diff=row_path_difference(losses,mapping,types,core,n,st.code,best.code,b)
                diff=bounded(diff+bounded(mu*(bestcost-cost),b),b)
                ties+=diff==0
                take=diff<0 || (diff==0 && (cost<bestcost || (cost==bestcost && st.code<best.code)))
            end
        end
        if take;best=st;bestcost=cost;bestlo=lo;besthi=hi;end
    end
    best===nothing && error("No filtered exact solution.")
    action=zeros(Int,length(types))
    for i in eachindex(types);types[i]>0 && (action[i]=basegear[mapping[types[i]]]);end
    code=best.code
    for i in Iterators.reverse(core);code,a=divrem(code,3);action[i]=Int(a);end
    code==0 || error("Filtered label code mismatch.")
    resource=basecost+bestcost
    sum(c[a+1] for a in action)==resource<=C || error("Filtered budget mismatch.")
    # Verify incumbent dominance with bounds first; only an unresolved endpoint
    # requires the exact sum. Evidence on small cores retains the historical keys.
    fl,fh=row_dyadic(fixedloss,shift,b);gl,gh=row_dyadic(gap,shift,b)
    dl=best.lo+fl+mulo*(C-resource);dh=best.hi+fh+muhi*(C-resource)
    deficit=nothing
    if dh>gl || evidence!==nothing
        q=row_path_difference(losses,mapping,types,core,n,best.code,nothing,b)
        deficit=bounded(bounded(mu*(C-resource),b)+bounded(fixedloss+q,b),b)
        0<=deficit<=gap || error("Filtered solution below feasible incumbent.")
    end
    tick!(b)
    if evidence!==nothing
        evidence["rows"]=copy(rows);evidence["counts"]=copy(counts)
        evidence["mu"]=mu;evidence["gap"]=gap;evidence["losses"]=losses
        evidence["incumbent_counts"]=incumbent;evidence["incumbent_resource_units"]=spent
        evidence["menus"]=deepcopy(menus);evidence["core_labels"]=copy(core)
        evidence["actions"]=copy(action);evidence["resource_units"]=resource
        evidence["upper_minus_score"]=deficit;evidence["dp_extensions"]=extensions
        evidence["peak_states"]=peak;evidence["active_common_denominator_constructed"]=false
        evidence["comparison_interface"]="certified-dyadic-partial-loss-v1"
    end
    copyto!(w.actions,action)
    (engine=:cap_active_exact,underlying_engine=:dual_core_dp_exact,
     score_interface=:individual_rational_core_v1,comparison_interface=:certified_dyadic_partial_loss_v1,
     work=b.work,peak_states=peak,resource_units=resource,exact_comparisons=b.exact,
     core_projects=n,dp_extensions=extensions,dyadic_fractional_bits=shift,
     separated_comparisons=separations,exact_order_refinements=refinements,
     exact_prune_refinements=prune_refinements,exact_ties=ties)
end


# Bounded, task-private high-precision limb storage.  The serialized Table,
# Workspace and Budget layouts above are intentionally unchanged.  This cache
# holds ONLY plain scratch arrays, never score tables, policies or RNG objects.
const ROW_LIMB_SCRATCH_LIMIT = 64*1024*1024
const ROW_LIMB_TLS_KEY = :multigear_row_limb_scratch_v1
mutable struct RowLimbScratch
    owner::Task
    busy::Bool
    lo1::Vector{UInt64}
    hi1::Vector{UInt64}
    lo2::Vector{UInt64}
    hi2::Vector{UInt64}
    low::Vector{UInt64}
    high::Vector{UInt64}
    constants::Vector{UInt64}
    temp::Vector{UInt64}
    choice::Vector{UInt8}
    mark1::Vector{Bool}
    mark2::Vector{Bool}
    rank1::Vector{Int}
    rank2::Vector{Int}
    rank_slots::Vector{Int}
end
RowLimbScratch()=RowLimbScratch(current_task(),false,UInt64[],UInt64[],UInt64[],UInt64[],
    UInt64[],UInt64[],UInt64[],UInt64[],UInt8[],Bool[],Bool[],Int[],Int[],Int[])
const ROW_LIMB_FIELDS=(:lo1,:hi1,:lo2,:hi2,:low,:high,:constants,:temp,:choice,
    :mark1,:mark2,:rank1,:rank2,:rank_slots)
function row_limb_storage_bytes(s::RowLimbScratch)
    sum(length(getfield(s,k))*sizeof(eltype(getfield(s,k))) for k in ROW_LIMB_FIELDS)
end
function row_limb_scratch()
    tls=task_local_storage()
    s=get(tls,ROW_LIMB_TLS_KEY,nothing)
    if !(s isa RowLimbScratch) || s.owner!==current_task()
        s=RowLimbScratch();tls[ROW_LIMB_TLS_KEY]=s
    end
    s::RowLimbScratch
end
function row_limb_clear!()
    tls=task_local_storage();s=get(tls,ROW_LIMB_TLS_KEY,nothing)
    s===nothing && return nothing
    s isa RowLimbScratch && s.busy && error("Cannot release an active row-DP scratch.")
    delete!(tls,ROW_LIMB_TLS_KEY)
    nothing
end
function row_limb_reserve!(s::RowLimbScratch,W::Int,width::Int,n::Int,nr::Int)
    # BigInt preflight avoids overflow in workspace-size arithmetic.  Failure of
    # this optional representation gate leaves the historical exact route in use.
    needs_big=(BigInt(W)*width,BigInt(W)*width,BigInt(W)*width,BigInt(W)*width,
        BigInt(W)*3nr,BigInt(W)*3nr,BigInt(W)*8,BigInt(W)*8,
        BigInt(width)*n,BigInt(width),BigInt(width),BigInt(width),BigInt(width),BigInt(3)*width+3)
    total=BigInt(0)
    for (k,need) in zip(ROW_LIMB_FIELDS,needs_big)
        v=getfield(s,k);total+=max(BigInt(length(v)),need)*sizeof(eltype(v))
    end
    total<=ROW_LIMB_SCRATCH_LIMIT || return false
    for (k,need) in zip(ROW_LIMB_FIELDS,needs_big)
        v=getfield(s,k);length(v)<need && resize!(v,Int(need))
    end
    true
end

# Little-endian nonnegative integers.  All arithmetic is exact.  The packed word
# count includes a proved extra carry limb; unexpected overflow fails closed.
@inline function row_limb_cmp(x,xoff::Int,y,yoff::Int,W::Int)
    @inbounds for k in W:-1:1
        a=x[xoff+k];z=y[yoff+k]
        a<z && return -1
        a>z && return 1
    end
    0
end
@inline function row_limb_add!(z,zoff::Int,x,xoff::Int,y,yoff::Int,W::Int)
    carry=false
    @inbounds for k in 1:W
        a,c1=Base.Checked.add_with_overflow(x[xoff+k],y[yoff+k])
        v,c2=Base.Checked.add_with_overflow(a,UInt64(carry))
        z[zoff+k]=v;carry=c1|c2
    end
    carry && error("Row limb addition overflow: action not published.")
    nothing
end
@inline function row_limb_mul!(z,zoff::Int,x,xoff::Int,m::Int,W::Int)
    m>=0 || error("Negative row limb multiplier.")
    carry=UInt128(0);mm=UInt128(m);mask=UInt128(typemax(UInt64))
    @inbounds for k in 1:W
        v=UInt128(x[xoff+k])*mm+carry
        z[zoff+k]=UInt64(v & mask);carry=v>>64
    end
    carry==0 || error("Row limb multiplication overflow: action not published.")
    nothing
end
function row_limb_pack!(v,off::Int,x::BigInt,W::Int)
    x>=0 || error("Negative row loss bound.")
    z=x;mask=BigInt(typemax(UInt64))
    @inbounds for k in 1:W
        v[off+k]=UInt64(z & mask);z=z>>64
    end
    z==0 || error("Insufficient row limb width.")
    nothing
end

"""Exact cancelled difference of two equal-depth paths in the byte traceback.
`override` supplies the proposed last gear before its next-layer slot is written.
Ranks are used ONLY for lexicographic ties, never for loss comparisons.
"""
function row_limb_path_difference(s::RowLimbScratch,width,losses,mapping,types,core,
        basegear,c,depth::Int,cost::Int,other,b::Budget;override::Int=-1)
    b.exact+=1;terms=Dict{Tuple{Int,Int},Int}();x=cost;y=other===nothing ? 0 : Int(other)
    shared_prefix=false
    for k in depth:-1:1
        # At equal remaining depth and spend, both stored paths refer to the
        # SAME accepted DP prefix. Its exact loss cancels, so do not replay it.
        # The proposed last action has not been stored yet: process an override
        # once before using this identity, even when cost == other initially.
        if other!==nothing && x==y && (k<depth || override<0)
            shared_prefix=true
            break
        end
        tick!(b);j=mapping[types[core[k]]]
        a=(k==depth && override>=0) ? override : Int(s.choice[(k-1)*width+x+1])
        x-=c[a+1]-c[basegear[j]+1]
        x>=0 || error("Invalid row-DP predecessor.")
        if other===nothing
            key=(j,a+1);terms[key]=get(terms,key,0)+1
        else
            z=Int(s.choice[(k-1)*width+y+1]);y-=c[z+1]-c[basegear[j]+1]
            y>=0 || error("Invalid other row-DP predecessor.")
            if a!=z
                key=(j,a+1);terms[key]=get(terms,key,0)+1
                key=(j,z+1);terms[key]=get(terms,key,0)-1
            end
        end
    end
    (shared_prefix || (x==0 && (other===nothing || y==0))) ||
        error("Row-DP traceback root mismatch.")
    q=zero(Q)
    for key in sort!(collect(keys(terms)))
        m=terms[key];m==0 && continue
        tick!(b);q=bounded(q+bounded(m*losses[key...],b),b)
    end
    q
end

# Dispatcher preserves the historical 4096-bit filter for domains that do not
# fit the optional bounded dense scratch.  Both representations make exact
# decisions and share the original allocation work/time limits.
function filtered_row_dp!(w,t,types,c,C,b::Budget,rows,counts,mapping,incumbent,spent,
        mu,gap,losses,menus,basegear,basecost,fixedloss,coregap,core;evidence=nothing)
    n=length(core);remaining=C-basecost
    n>0 && remaining>=0 || error("Invalid filtered residual problem.")
    # Restrict dense indexing before any size/product conversion or allocation.
    # This is an implementation admission bound, not a change to the feasible set.
    if remaining>65535
        return bigint_filtered_row_dp!(w,t,types,c,C,b,rows,counts,mapping,incumbent,spent,
            mu,gap,losses,menus,basegear,basecost,fixedloss,coregap,core;evidence=evidence)
    end
    mag(q)=max(0,ndigits(abs(numerator(q));base=2)-ndigits(denominator(q);base=2)+1)
    magnitude=max(mag(maximum(losses))+ndigits(n+1;base=2),mag(gap),
                  mag(mu)+ndigits(C+1;base=2),mag(fixedloss))
    shift=min(4096,b.max_exact_bits-8-magnitude)
    abs(shift)<=b.max_exact_bits || throw(MA.AllocationLimit("Dyadic score-scale bit budget; no approximate action returned."))
    lows=Matrix{BigInt}(undef,size(losses));highs=similar(lows)
    for j in axes(losses,1),a in 1:3
        lows[j,a],highs[j,a]=row_dyadic(losses[j,a],shift,b)
    end
    gl,gh=row_dyadic(coregap,shift,b);ml,mh=row_dyadic(mu,shift,b)
    fl,fh=row_dyadic(fixedloss,shift,b);tl,th=row_dyadic(gap,shift,b)
    # Covers partial sums, final cross-multiplied comparisons, the incumbent
    # check, AND the individually stored price endpoints. With C == 0 the
    # product mh*C is zero, but ml and mh are still packed into constants.
    # max(C,1) therefore covers the price itself as well as every price*spend.
    # For C >= 1 this bound is exactly unchanged. Keep the extra carry limb.
    maximum_value=n*maximum(highs)+mh*max(C,1)+fh+gh+th+1
    W=max(1,cld(ndigits(maximum_value;base=2),64)+1);width=remaining+1
    s=row_limb_scratch();s.busy && error("Reentrant use of task-private row-DP scratch.")
    if !row_limb_reserve!(s,W,width,n,length(rows))
        return bigint_filtered_row_dp!(w,t,types,c,C,b,rows,counts,mapping,incumbent,spent,
            mu,gap,losses,menus,basegear,basecost,fixedloss,coregap,core;evidence=evidence)
    end
    s.busy=true
    try
        for j in axes(losses,1),a in 1:3
            off=((j-1)*3+a-1)*W
            row_limb_pack!(s.low,off,lows[j,a],W);row_limb_pack!(s.high,off,highs[j,a],W)
        end
        for (k,x) in enumerate((gl,gh,ml,mh,fl,fh,tl,th))
            row_limb_pack!(s.constants,(k-1)*W,x,W)
        end
        return limb_row_dp!(s,W,width,shift,w,t,types,c,C,b,rows,counts,mapping,
            incumbent,spent,mu,gap,losses,menus,basegear,basecost,fixedloss,coregap,core;evidence=evidence)
    finally
        s.busy=false
    end
end

"""Allocation-stable inner DP.  No dictionaries, rational partial sums, growing
BigInt label codes, or new endpoint objects are created for candidate extensions.
Byte traceback costs (capacity+1)*core_length; two limb layers are reused.
"""
function limb_row_dp!(s::RowLimbScratch,W::Int,width::Int,shift::Int,w,t,types,c,C,b::Budget,
        rows,counts,mapping,incumbent,spent,mu,gap,losses,menus,basegear,basecost,
        fixedloss,coregap,core;evidence=nothing)
    n=length(core);remaining=width-1
    lo=s.lo1;hi=s.hi1;nlo=s.lo2;nhi=s.hi2
    mark=s.mark1;nmark=s.mark2;rank=s.rank1;nrank=s.rank2
    fill!(mark,false);fill!(nmark,false)
    @inbounds for k in 1:W;lo[k]=0;hi[k]=0;end
    mark[1]=true;rank[1]=1;count_states=1
    peak=1;extensions=0;separations=0;refinements=0;ties=0;prune_refinements=0
    for (depth,i) in enumerate(core)
        j=mapping[types[i]];bg=basegear[j];fill!(nmark,false);next_count=0
        for cost in 0:remaining
            # Dense scanning is charged, including unreachable slots.
            tick!(b);idx=cost+1;mark[idx] || continue;off=cost*W
            for a in menus[j]
                tick!(b);extensions+=1
                nc=cost+c[a+1]-c[bg+1];nc<=remaining || continue
                ni=nc+1;no=nc*W;ao=((j-1)*3+a)*W
                row_limb_add!(s.temp,0,lo,off,s.low,ao,W)
                row_limb_add!(s.temp,W,hi,off,s.high,ao,W)
                tick!(b)
                if row_limb_cmp(s.temp,0,s.constants,W,W)>0
                    separations+=1;continue
                end
                if row_limb_cmp(s.temp,W,s.constants,0,W)>0
                    prune_refinements+=1
                    q=row_limb_path_difference(s,width,losses,mapping,types,core,basegear,c,
                        depth,nc,nothing,b;override=a)
                    q>coregap && continue
                else
                    separations+=1
                end
                key=3rank[idx]+a;take=!nmark[ni]
                if !take
                    tick!(b)
                    if row_limb_cmp(s.temp,W,nlo,no,W)<0
                        take=true;separations+=1
                    elseif row_limb_cmp(s.temp,0,nhi,no,W)>0
                        take=false;separations+=1
                    else
                        refinements+=1
                        q=row_limb_path_difference(s,width,losses,mapping,types,core,basegear,c,
                            depth,nc,nc,b;override=a)
                        ties+=q==0
                        take=q<0 || (q==0 && key<nrank[ni])
                    end
                end
                if take
                    if !nmark[ni]
                        next_count>=b.limits.max_states && throw(MA.AllocationLimit("Packed row-core state limit; no approximate action returned."))
                        next_count+=1;nmark[ni]=true
                    end
                    copyto!(nlo,no+1,s.temp,1,W);copyto!(nhi,no+1,s.temp,W+1,W)
                    nrank[ni]=key;s.choice[(depth-1)*width+ni]=UInt8(a)
                end
            end
        end
        next_count>0 || error("Packed exact core lost the feasible incumbent.")
        # Compress (parent lex-rank, last gear) keys in linear bounded storage.
        # A key identifies a unique labelled prefix and hence a unique spend.
        fill!(s.rank_slots,0)
        for nc in 0:remaining
            tick!(b);ni=nc+1;nmark[ni] || continue
            slot=nrank[ni]+1
            s.rank_slots[slot]==0 || error("Duplicate labelled prefix at different spends.")
            s.rank_slots[slot]=ni
        end
        next_rank=0
        for k in 1:(3count_states+3)
            tick!(b);ni=s.rank_slots[k];ni==0 && continue
            next_rank+=1;nrank[ni]=next_rank
        end
        next_rank==next_count || error("Incomplete packed prefix rank inventory.")
        peak=max(peak,next_count);count_states=next_count
        lo,nlo=nlo,lo;hi,nhi=nhi,hi;mark,nmark=nmark,mark;rank,nrank=nrank,rank
    end
    have=false;bestcost=0
    for cost in 0:remaining
        tick!(b);idx=cost+1;mark[idx] || continue;take=!have
        if have
            # Compare loss(cost)+mu*bestcost to loss(best)+mu*cost.  Both sides
            # are nonnegative and the shared term cancels exactly.
            row_limb_mul!(s.temp,0,s.constants,2W,bestcost,W)
            row_limb_mul!(s.temp,W,s.constants,3W,bestcost,W)
            row_limb_add!(s.temp,2W,lo,cost*W,s.temp,0,W)
            row_limb_add!(s.temp,3W,hi,cost*W,s.temp,W,W)
            row_limb_mul!(s.temp,0,s.constants,2W,cost,W)
            row_limb_mul!(s.temp,W,s.constants,3W,cost,W)
            row_limb_add!(s.temp,4W,lo,bestcost*W,s.temp,0,W)
            row_limb_add!(s.temp,5W,hi,bestcost*W,s.temp,W,W)
            if row_limb_cmp(s.temp,3W,s.temp,4W,W)<0
                take=true;separations+=1
            elseif row_limb_cmp(s.temp,2W,s.temp,5W,W)>0
                take=false;separations+=1
            else
                refinements+=1
                q=row_limb_path_difference(s,width,losses,mapping,types,core,basegear,c,n,cost,bestcost,b)
                q=bounded(q+bounded(mu*(bestcost-cost),b),b);ties+=q==0
                take=q<0 || (q==0 && (cost<bestcost || (cost==bestcost && rank[idx]<rank[bestcost+1])))
            end
        end
        if take;have=true;bestcost=cost;end
    end
    have || error("No packed exact solution.")
    action=zeros(Int,length(types))
    for i in eachindex(types);types[i]>0 && (action[i]=basegear[mapping[types[i]]]);end
    cost=bestcost
    for k in n:-1:1
        j=mapping[types[core[k]]];a=Int(s.choice[(k-1)*width+cost+1]);action[core[k]]=a
        cost-=c[a+1]-c[basegear[j]+1]
    end
    cost==0 || error("Packed traceback root mismatch.")
    resource=basecost+bestcost
    sum(c[a+1] for a in action)==resource<=C || error("Packed budget mismatch.")
    row_limb_mul!(s.temp,0,s.constants,3W,C-resource,W)
    row_limb_add!(s.temp,W,hi,bestcost*W,s.constants,5W,W)
    row_limb_add!(s.temp,2W,s.temp,W,s.temp,0,W)
    deficit=nothing
    if row_limb_cmp(s.temp,2W,s.constants,6W,W)>0 || evidence!==nothing
        q=row_limb_path_difference(s,width,losses,mapping,types,core,basegear,c,n,bestcost,nothing,b)
        deficit=bounded(bounded(mu*(C-resource),b)+bounded(fixedloss+q,b),b)
        0<=deficit<=gap || error("Packed solution below feasible incumbent.")
    end
    tick!(b)
    if evidence!==nothing
        evidence["rows"]=copy(rows);evidence["counts"]=copy(counts)
        evidence["mu"]=mu;evidence["gap"]=gap;evidence["losses"]=losses
        evidence["incumbent_counts"]=incumbent;evidence["incumbent_resource_units"]=spent
        evidence["menus"]=deepcopy(menus);evidence["core_labels"]=copy(core)
        evidence["actions"]=copy(action);evidence["resource_units"]=resource
        evidence["upper_minus_score"]=deficit;evidence["dp_extensions"]=extensions
        evidence["peak_states"]=peak;evidence["active_common_denominator_constructed"]=false
        evidence["comparison_interface"]="packed-limb-partial-loss-v1"
    end
    copyto!(w.actions,action)
    (engine=:cap_active_exact,underlying_engine=:dual_core_dp_exact,
     score_interface=:individual_rational_core_v1,comparison_interface=:packed_limb_partial_loss_v1,
     work=b.work,peak_states=peak,resource_units=resource,exact_comparisons=b.exact,
     core_projects=n,dp_extensions=extensions,dyadic_fractional_bits=shift,
     separated_comparisons=separations,exact_order_refinements=refinements,
     exact_prune_refinements=prune_refinements,exact_ties=ties,
     packed_words=W,scratch_storage_bytes=row_limb_storage_bytes(s),
     candidate_endpoint_objects=0,candidate_bigint_label_codes=0)
end

"""Crossing case: compile ONLY active score rows, with a pre-checked LCM budget,
then call the UNCHANGED accepted exact core/DP. The table-wide denominator is never
constructed. If this local encoding is too large the allocation is refused.
"""
function local_exact!(w,t,types,c,C,domain,b::Budget)
    rows=sort!(unique(filter(>(0),types)));den=BigInt(1)
    for r in rows,a in axes(t.values,2)
        tick!(b);d=denominator(t.values[r,a]);reduced=div(d,gcd(den,d))
        ndigits(den;base=2)+ndigits(reduced;base=2)-1<=b.max_exact_bits ||
            throw(MA.AllocationLimit("Active-row exact denominator exceeds cap-interface budget; policy not approximated."))
        den*=reduced
        ndigits(den;base=2)<=b.max_exact_bits || throw(MA.AllocationLimit("Active-row denominator limit."))
    end
    mapping=Dict(r=>j for (j,r) in enumerate(rows));active_types=[x==0 ? 0 : mapping[x] for x in types]
    table=MA.compile_scores(t.values[rows,:];max_population=length(types),max_bits=b.max_exact_bits)
    tick!(b)
    left=b.limits.seconds-(time_ns()-b.started)/1e9
    left>0 || throw(MA.AllocationLimit("Active-row encoding exhausted the allocation clock."))
    limits=MA.Limits(b.limits.max_states,b.limits.max_updates-b.work,left)
    limits.max_updates>0 || throw(MA.AllocationLimit("Active-row encoding exhausted the work allowance."))
    result=MA.allocate(table,active_types,c,C,domain;limits=limits)
    copyto!(w.actions,result.actions)
    (engine=:cap_active_exact,underlying_engine=result.engine,work=b.work+result.work,peak_states=result.peak_states,
        resource_units=result.resource_units,exact_comparisons=b.exact)
end
function allocate!(w::Workspace,t::Table,types::Vector{Int},c::Vector{Int},C::Int,domain::Symbol;
        limits=MA.Limits(),max_exact_bits=524288)
    RB.ieee_check()
    domain in (:k,:ni,:fc) && C>=0 && length(c)==size(t.values,2) && c[1]==0 && all(diff(c).>0) || error("Invalid cap allocation domain/menu.")
    length(types)==length(w.actions)<=t.max_population && all(x->0<=x<=size(t.values,1),types) || error("Cap score row/population mismatch.")
    BigInt(length(types))*maximum(c)<=div(typemax(Int),4) && C<=div(typemax(Int),4) || error("Resource overflow.")
    limits.max_states>0 && limits.max_updates>0 && limits.seconds>0 || error("Invalid cap allocation limits.")
    b=Budget(limits;max_exact_bits=max_exact_bits)
    result=ordered!(w,t,types,c,C,domain,b)
    result===nothing && (result=row_core!(w,t,types,c,C,domain,b))
    result===nothing && (result=local_exact!(w,t,types,c,C,domain,b))
    sum(c[a+1] for a in w.actions)==result.resource_units<=C || error("Cap allocation hard-budget violation.")
    all(types[i]>0 || w.actions[i]==0 for i in eachindex(types)) || error("Positive gear at zero state.")
    if domain==:ni
        all(types[i]==0 || w.actions[i]==length(c)-1 || C-result.resource_units<c[w.actions[i]+2]-c[w.actions[i]+1] for i in eachindex(types)) || error("NI upgrade remains feasible.")
    end
    result
end
end # module
