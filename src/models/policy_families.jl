"""A family restricts the downshift path, never the oracle's Bellman deviations."""
abstract type AbstractPolicyFamily end
struct UnrestrictedFamily <: AbstractPolicyFamily end

"""Nondecreasing gears in the supplied ordering of ALL controllable states."""
struct OrderedThresholdFamily <: AbstractPolicyFamily
    states::Vector{Int}
    function OrderedThresholdFamily(states::AbstractVector{<:Integer})
        v = Int.(states)
        !isempty(v) && length(unique(v)) == length(v) ||
            throw(ArgumentError("Provide a nonempty ordering without repeated states."))
        new(v)
    end
end

"""An explicit small family. Both extreme policies must be included."""
struct ExplicitPolicyFamily <: AbstractPolicyFamily
    policies::Set{Tuple}
    function ExplicitPolicyFamily(policies::AbstractVector{<:AbstractVector{<:Integer}})
        isempty(policies) && throw(ArgumentError("The policy family cannot be empty."))
        new(Set{Tuple}(Tuple(Int.(p)) for p in policies))
    end
end

policy_in_family(::UnrestrictedFamily, actions) = true
policy_in_family(f::ExplicitPolicyFamily, actions) = Tuple(actions) in f.policies
function policy_in_family(f::OrderedThresholdFamily, actions)
    return all(actions[f.states[k]] <= actions[f.states[k+1]]
               for k in 1:(length(f.states)-1))
end

function _check_family(model, family::AbstractPolicyFamily)
    if family isa OrderedThresholdFamily
        sort(family.states) == findall(model.controllable) || throw(ArgumentError(
            "The threshold ordering must contain every controllable state exactly once."))
    elseif family isa ExplicitPolicyFamily
        for p in family.policies
            length(p) == nstates(model) || throw(DimensionMismatch("Invalid family policy length."))
            all(p[i] in admissible_gears(model,i) for i in 1:nstates(model)) ||
                throw(ArgumentError("An explicit-family policy uses an inadmissible gear."))
        end
    end
    low = zeros(Int,nstates(model))
    high = [model.controllable[i] ? maxgear(model) : 0 for i in 1:nstates(model)]
    policy_in_family(family,low) && policy_in_family(family,high) ||
        throw(ArgumentError("The family must contain both extreme policies."))
    return nothing
end

"""Return feasible (state, upper-gear) pairs in lexicographic order."""
function downshift_frontier(model, actions::AbstractVector{<:Integer},
                            family::AbstractPolicyFamily=UnrestrictedFamily())
    length(actions) == nstates(model) || throw(DimensionMismatch("Invalid policy length."))
    all(actions[i] in admissible_gears(model,i) for i in 1:nstates(model)) ||
        throw(ArgumentError("Inadmissible policy."))
    policy_in_family(family,actions) || throw(ArgumentError("The policy is outside the family."))
    pairs = Tuple{Int,Int}[]
    for i in 1:nstates(model)
        a = actions[i]
        if model.controllable[i] && a > 0
            successor = Int.(actions)
            successor[i] -= 1
            policy_in_family(family,successor) && push!(pairs,(i,Int(a)))
        end
    end
    return pairs
end


"""Product of ordered gear chains, with no ordering between rows.
Each row lists increasing states; rows must partition controllable states.
This family describes admissible paths, not a claim of PCL-indexability.
"""
struct RowThresholdFamily <: AbstractPolicyFamily
    rows::Vector{Vector{Int}}
    predecessor::Vector{Int}
    row_of::Vector{Int}
    position::Vector{Int}
    function RowThresholdFamily(rows::AbstractVector{<:AbstractVector{<:Integer}})
        rs=[Int.(r) for r in rows]
        !isempty(rs) && all(!isempty,rs) || throw(ArgumentError("Rows must be nonempty."))
        ids=vcat(rs...)
        minimum(ids)>=1 && length(unique(ids))==length(ids) ||
            throw(ArgumentError("Row states must be positive and distinct across rows."))
        pred=zeros(Int,maximum(ids)); row_of=similar(pred); fill!(row_of,0)
        position=similar(pred); fill!(position,0)
        for (b,r) in enumerate(rs), (k,i) in enumerate(r)
            row_of[i]=b; position[i]=k
        end
        for r in rs, k in 2:length(r);pred[r[k]]=r[k-1];end
        new(rs,pred,row_of,position)
    end
end
function policy_in_family(f::RowThresholdFamily, actions)
    length(actions)>=length(f.predecessor) || return false
    all(actions[r[k]]<=actions[r[k+1]] for r in f.rows for k in 1:length(r)-1)
end
function _check_family(model, f::RowThresholdFamily)
    sort(vcat(f.rows...))==findall(model.controllable) ||
        throw(ArgumentError("Rows must partition every controllable state exactly once."))
    nothing
end
function downshift_frontier(model,actions::AbstractVector{<:Integer},f::RowThresholdFamily)
    length(actions)==nstates(model) || throw(DimensionMismatch("Invalid policy length."))
    all(actions[i] in admissible_gears(model,i) for i in 1:nstates(model)) ||
        throw(ArgumentError("Inadmissible policy."))
    _check_family(model,f)
    policy_in_family(f,actions) || throw(ArgumentError("Policy is outside the row family."))
    [(i,Int(actions[i])) for i in findall(model.controllable) if actions[i]>0 &&
        (f.predecessor[i]==0 || actions[f.predecessor[i]]<actions[i])]
end
