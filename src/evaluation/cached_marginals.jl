"""
Frontier-only M-DS cache. Entries are (state, upper gear), never just states.
The stored transition rows are fixed primitive differences for the CURRENT
frontier only. All arrays are private working data; do not mutate them outside
these routines. Refreshing creates a new cache and does not edit old snapshots.
The cache is not the all-current-comparisons cache of reduced FP-DS.
"""
struct FrontierMetricCache{T<:AbstractFloat}
    model::FiniteMultiGearModel{T}
    criterion::Symbol
    beta::Union{Nothing,T}
    omega::Union{Nothing,Vector{T}}
    actions::Vector{Int}
    frontier::Vector{Tuple{Int,Int}}
    rows::Vector{Vector{T}}
    immediate_f::Vector{T}
    immediate_g::Vector{T}
    row_l1::Vector{T}
    f::Vector{T}
    g::Vector{T}
end

"""Old-policy data for one cache refresh; prepared BEFORE the inverse update."""
struct MarginalCachePivot{T<:AbstractFloat}
    model::FiniteMultiGearModel{T}
    criterion::Symbol
    beta::Union{Nothing,T}
    omega::Union{Nothing,Vector{T}}
    actions::Vector{Int}
    selected::Tuple{Int,Int}
    column::Vector{T}
    denominator::T
    f::T
    g::T
end

"""
Actual cache work at one policy. New/reentering exposures and extra rebuild
work are separate. A direct evaluation computes BOTH f and g for ONE pair;
a recurrence refresh updates BOTH scalars for ONE retained pair.
"""
struct MarginalCacheRecord
    policy_index::Int
    frontier::Vector{Tuple{Int,Int}}
    exposed::Vector{Tuple{Int,Int}}
    retained::Vector{Tuple{Int,Int}}
    discarded::Vector{Tuple{Int,Int}}
    direct_evaluations::Int
    recurrence_refreshes::Int
    rebuild_evaluations::Int
    rebuilt::Bool
    reason::Symbol
end

_cache_vectors(e::DiscountedEvaluation) = (e.F,e.G)
_cache_vectors(e::AverageEvaluation) = (e.phi,e.gamma)
_cache_criterion(e::DiscountedEvaluation) = :discounted
_cache_criterion(e::AverageEvaluation) = :average
_cache_beta(e::DiscountedEvaluation) = e.beta
_cache_beta(e::AverageEvaluation) = nothing
_cache_omega(e::DiscountedEvaluation) = nothing
_cache_omega(e::AverageEvaluation) = copy(e.omega)

# Cache construction evaluates fixed transition differences on exposure. Rows
# can be retained without reconstruction while their PAIR stays feasible.
function _fresh_cache_entry(e,i,a)
    T = eltype(e.model.h)
    factor = e isa DiscountedEvaluation ? e.beta : one(T)
    row = factor .* (e.model.P[i,:,a+1]-e.model.P[i,:,a])
    f0 = e.model.h[i,a]-e.model.h[i,a+1]
    g0 = e.model.c[i,a+1]-e.model.c[i,a]
    v,w = _cache_vectors(e)
    return row,f0,g0,sum(abs,row),f0-dot(row,v),g0+dot(row,w)
end

function _empty_marginal_cache(e,frontier)
    T = eltype(e.model.h)
    m = length(frontier)
    return FrontierMetricCache(e.model,_cache_criterion(e),_cache_beta(e),_cache_omega(e),
        copy(e.policy.actions),copy(frontier),Vector{Vector{T}}(undef,m),
        zeros(T,m),zeros(T,m),zeros(T,m),zeros(T,m),zeros(T,m))
end

function _fill_cache_entry!(cache,k,e,pair)
    row,f0,g0,l1,f,g = _fresh_cache_entry(e,pair...)
    cache.rows[k] = row
    cache.immediate_f[k],cache.immediate_g[k],cache.row_l1[k] = f0,g0,l1
    cache.f[k],cache.g[k] = f,g
    return nothing
end

"""
    initialize_marginal_cache(e; family=UnrestrictedFamily(), policy_index=0)

Evaluate every feasible current pair directly from one policy evaluation.
Returns (cache, work_record); performs NO policy solve. Empty frontiers are
allowed (e.g. the all-zero endpoint). Invalid policy/family inputs are rejected.
"""
function initialize_marginal_cache(e::Union{DiscountedEvaluation,AverageEvaluation};
                                   family::AbstractPolicyFamily=UnrestrictedFamily(),
                                   policy_index::Integer=0)
    policy_index >= 0 || throw(ArgumentError("policy_index must be nonnegative."))
    _check_family(e.model,family)
    frontier = downshift_frontier(e.model,e.policy.actions,family)
    cache = _empty_marginal_cache(e,frontier)
    for (k,pair) in enumerate(frontier)
        _fill_cache_entry!(cache,k,e,pair)
    end
    record = MarginalCacheRecord(Int(policy_index),copy(frontier),copy(frontier),
        Tuple{Int,Int}[],Tuple{Int,Int}[],length(frontier),0,0,false,:initialization)
    return cache,record
end

"""
    prepare_marginal_pivot(state, i)

Copy the OLD inverse column and compute the OLD selected-pair f, g and
Sherman-Morrison denominator. These match the values used by the existing
rank-one kernel, not a possibly exact-converted ratio used for selection.
This extra selected-pair O(N) work is counted separately by M-DS. No solve and
no update occurs here. The average inverse column INCLUDES its rate coordinate.
"""
function prepare_marginal_pivot(state::RankOnePolicyState{T},i::Integer) where {T}
    n = nstates(state.model)
    1 <= i <= n || throw(BoundsError(state.policy.actions,i))
    a = state.policy.actions[i]
    state.model.controllable[i] && a > 0 || throw(ArgumentError("No downshift at state $i."))
    d = zeros(T,size(state.inverse,1))
    factor = state.criterion == :discounted ? state.beta : one(T)
    d[1:n] = factor*(state.model.P[i,:,a+1]-state.model.P[i,:,a])
    y = copy(@view(state.inverse[:,i]))
    eta = one(T)+dot(d,y)
    f = state.model.h[i,a]-state.model.h[i,a+1]-dot(d,@view(state.values[:,1]))
    g = state.model.c[i,a+1]-state.model.c[i,a]+dot(d,@view(state.values[:,2]))
    return MarginalCachePivot(state.model,state.criterion,state.beta,
        state.omega === nothing ? nothing : copy(state.omega),copy(state.policy.actions),
        (Int(i),a),y,eta,f,g)
end

function _check_cache_transition(cache,pivot,e)
    cache.model === pivot.model && cache.model === e.model ||
        throw(ArgumentError("Cache, pivot and evaluation must refer to the same model."))
    cache.criterion == pivot.criterion == _cache_criterion(e) ||
        throw(ArgumentError("Cache criterion mismatch."))
    cache.beta == pivot.beta == _cache_beta(e) || throw(ArgumentError("Cache beta mismatch."))
    cache.omega == pivot.omega == _cache_omega(e) || throw(ArgumentError("Cache normalization mismatch."))
    cache.actions == pivot.actions || throw(ArgumentError("Cache is not from the pre-pivot policy."))
    successor = copy(pivot.actions)
    i,a = pivot.selected
    successor[i] == a && a > 0 || throw(ArgumentError("Invalid recorded cache pivot."))
    successor[i] -= 1
    e.policy.actions == successor || throw(ArgumentError("Evaluation is not the one-step successor."))
    return nothing
end

"""
    refresh_marginal_cache(cache, pivot, e; family=UnrestrictedFamily(),
                          policy_index=1, rebuild=false, reason=:none)

Return a new frontier-only cache at successor evaluation e and a work record.
For a retained pair use chi = (beta*Delta p)*OLD_y/OLD_eta, or
Delta p*OLD_y_bias/OLD_eta under the average criterion, and update
f_new=f_old-chi*f_pivot, g_new=g_old-chi*g_pivot. Both signs are MINUS.
New/reentering pairs are evaluated directly; departing pairs are discarded.
If the policy workspace was refactorized, the caller MUST set rebuild=true:
all current metrics are then reevaluated, with the extra retained-pair work
reported separately. The runner below enforces this automatically.
No metrics, denominators or ratios are rounded or clipped by a tolerance.
"""
function refresh_marginal_cache(cache::FrontierMetricCache{T},pivot::MarginalCachePivot{T},
                                e::Union{DiscountedEvaluation{T},AverageEvaluation{T}};
                                family::AbstractPolicyFamily=UnrestrictedFamily(),
                                policy_index::Integer=1,rebuild::Bool=false,
                                reason::Symbol=:none) where {T}
    policy_index >= 1 || throw(ArgumentError("A successor policy_index must be positive."))
    _check_cache_transition(cache,pivot,e)
    _check_family(e.model,family)
    frontier = downshift_frontier(e.model,e.policy.actions,family)
    lookup = Dict(pair => k for (k,pair) in enumerate(cache.frontier))
    oldset,newset = Set(cache.frontier),Set(frontier)
    exposed = [pair for pair in frontier if !(pair in oldset)]
    retained = [pair for pair in frontier if pair in oldset]
    discarded = [pair for pair in cache.frontier if !(pair in newset)]
    if !rebuild && !isempty(retained)
        isfinite(pivot.denominator) && pivot.denominator > 0 &&
            isfinite(pivot.f) && isfinite(pivot.g) && all(isfinite,pivot.column) ||
            throw(ArgumentError("Invalid old pivot data for a cache recurrence."))
    end
    next = _empty_marginal_cache(e,frontier)
    n = nstates(e.model)
    y = @view(pivot.column[1:n])
    v,w = _cache_vectors(e)
    zero_f,zero_g = all(iszero,v),all(iszero,w)
    for (k,pair) in enumerate(frontier)
        if !rebuild && haskey(lookup,pair)
            j = lookup[pair]
            next.rows[k] = copy(cache.rows[j])
            next.immediate_f[k],next.immediate_g[k] = cache.immediate_f[j],cache.immediate_g[j]
            next.row_l1[k] = cache.row_l1[j]
            chi = dot(cache.rows[j],y)/pivot.denominator
            next.f[k] = cache.f[j]-chi*pivot.f
            next.g[k] = cache.g[j]-chi*pivot.g
            # Same exact-zero-column identity as the rank-one kernel. This
            # is not clipping small nonzero numbers and costs no dot product.
            zero_f && (next.f[k] = next.immediate_f[k])
            zero_g && (next.g[k] = next.immediate_g[k])
        else
            _fill_cache_entry!(next,k,e,pair)
        end
    end
    extra = rebuild ? length(retained) : 0
    record = MarginalCacheRecord(Int(policy_index),copy(frontier),exposed,retained,discarded,
        length(exposed)+extra,rebuild ? 0 : length(retained),extra,
        rebuild && !isempty(frontier),rebuild ? reason : :rank_one_refresh)
    return next,record
end

# Avoid a hidden resource dot product merely to screen every retained pair.
# The l1/infinity bound dominates the direct absolute-dot cancellation scale
# in real arithmetic; it is NOT an interval-arithmetic error bound. It can be
# more conservative than the C-DS/R-DS scale and trigger extra exact fallbacks.
function _cached_frontier_metrics(cache::FrontierMetricCache{T},e) where {T}
    cache.model === e.model && cache.actions == e.policy.actions ||
        throw(ArgumentError("Cache does not match the evaluation."))
    _,w = _cache_vectors(e)
    wnorm = norm(w,Inf) # once per policy, not once per frontier entry
    scales = abs.(cache.immediate_g) .+ cache.row_l1.*wnorm
    return copy(cache.f),copy(cache.g),scales
end

_cache_finite(cache::FrontierMetricCache) = all(isfinite,cache.f) && all(isfinite,cache.g) &&
    all(isfinite,cache.immediate_f) && all(isfinite,cache.immediate_g) &&
    all(isfinite,cache.row_l1)
