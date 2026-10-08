"""
    FiniteMultiGearModel(P, h, c; controllable=trues(size(h, 1)),
                        float_type=Float64, row_atol=1e-12)

Dense finite-state model with gears 0:A. Julia storage uses slot `a + 1`:
`P[i,j,a+1] = p_ij^a`, `h[i,a+1] = h_i^a`, `c[i,a+1] = c_i^a`.
The state labels are 1:N. P has size (N,N,A+1), h and c size (N,A+1).

At controllable states all gears are admissible and resource use must be
nonnegative and strictly increasing. Other states admit only gear 0. Their
unused higher-gear storage slots are filled with copies of their gear-0 data;
these slots are placeholders, NOT extra admissible actions. Only admissible
input rows are validated. There must be at least one controllable state.

The constructor copies its inputs and does not normalize transition rows,
clip negative probabilities, or infer indexability. `row_atol` is an absolute
row-sum tolerance with zero relative tolerance. Even tiny negative entries
are rejected. Float64 is the reference default; BigFloat is also supported.
Do not mutate the stored arrays after construction, particularly while jobs
share a model or evaluations refer to it. Create a new model for new primitives.
"""
struct FiniteMultiGearModel{T<:AbstractFloat}
    P::Array{T,3}
    h::Matrix{T}
    c::Matrix{T}
    controllable::BitVector
    row_atol::T

    # An inner constructor prevents bypassing validation via field arguments.
    function FiniteMultiGearModel(
        ::Type{T}, P::AbstractArray{<:Real,3}, h::AbstractMatrix{<:Real},
        c::AbstractMatrix{<:Real};
        controllable::AbstractVector{Bool}=trues(size(h, 1)),
        row_atol::Real=1e-12,
    ) where {T<:AbstractFloat}
        Base.require_one_based_indexing(P, h, c, controllable)
        n, ngears = size(h)
        n >= 1 || throw(ArgumentError("At least one state is required."))
        ngears >= 2 || throw(ArgumentError("At least gears 0 and 1 are required."))
        size(c) == size(h) || throw(DimensionMismatch("h and c must have equal sizes."))
        size(P) == (n, n, ngears) || throw(DimensionMismatch(
            "P must have size (N,N,A+1) matching h and c."
        ))
        length(controllable) == n || throw(DimensionMismatch("Invalid controllable mask length."))
        any(controllable) || throw(ArgumentError("At least one state must be controllable."))
        isfinite(row_atol) && row_atol >= 0 || throw(ArgumentError(
            "row_atol must be finite and nonnegative."
        ))
        tol = T(row_atol)
        isfinite(tol) && zero(T) <= tol < one(T) || throw(ArgumentError(
            "row_atol must be finite and lie in [0,1)."
        ))

        pp = Array{T,3}(undef, n, n, ngears)
        hh = Matrix{T}(undef, n, ngears)
        cc = Matrix{T}(undef, n, ngears)
        mask = BitVector(controllable)
        for i in 1:n
            for slot in 1:(mask[i] ? ngears : 1)
                c[i,slot] >= 0 || throw(ArgumentError("Negative or invalid input resource."))
                hi, ci = T(h[i,slot]), T(c[i,slot])
                isfinite(hi) || throw(ArgumentError("Nonfinite holding cost at ($i,$(slot-1))."))
                isfinite(ci) && ci >= zero(T) || throw(ArgumentError(
                    "Resource must be finite and nonnegative at ($i,$(slot-1))."
                ))
                hh[i,slot], cc[i,slot] = hi, ci
                for j in 1:n
                    P[i,j,slot] >= 0 || throw(ArgumentError("Negative or invalid input probability."))
                    pij = T(P[i,j,slot])
                    isfinite(pij) && pij >= zero(T) || throw(ArgumentError(
                        "Invalid transition probability at ($i,$j,$(slot-1))."
                    ))
                    pp[i,j,slot] = pij
                end
                rowsum = sum(@view pp[i,:,slot])
                abs(rowsum - one(T)) <= tol || throw(ArgumentError(
                    "Transition row ($i,$(slot-1)) sums to $rowsum, not 1 within row_atol=$tol."
                ))
                if slot > 1 && !(cc[i,slot] > cc[i,slot-1])
                    throw(ArgumentError("Resource must strictly increase with gear at state $i."))
                end
            end
            if !mask[i]
                for slot in 2:ngears
                    pp[i,:,slot] = pp[i,:,1]
                    hh[i,slot], cc[i,slot] = hh[i,1], cc[i,1]
                end
            end
        end
        return new{T}(pp, hh, cc, mask, tol)
    end
end

# The numeric type is a positional argument to the validated inner constructor.
# This convenience interface preserves the documented keyword-based API.
function FiniteMultiGearModel(P::AbstractArray{<:Real,3}, h::AbstractMatrix{<:Real},
                              c::AbstractMatrix{<:Real};
                              controllable::AbstractVector{Bool}=trues(size(h,1)),
                              float_type::Type{<:AbstractFloat}=Float64,
                              row_atol::Real=1e-12)
    return FiniteMultiGearModel(float_type,P,h,c;controllable=controllable,row_atol=row_atol)
end

nstates(model::FiniteMultiGearModel) = size(model.h, 1)
maxgear(model::FiniteMultiGearModel) = size(model.h, 2) - 1
ncontrollable(model::FiniteMultiGearModel) = count(model.controllable)

"""Return actual gear labels, not Julia array slots."""
function admissible_gears(model::FiniteMultiGearModel, i::Integer)
    1 <= i <= nstates(model) || throw(BoundsError(model.controllable, i))
    return 0:(model.controllable[i] ? maxgear(model) : 0)
end

"""Copied primitives of one admissible deterministic stationary policy."""
struct PolicyData{T<:AbstractFloat}
    actions::Vector{Int}
    P::Matrix{T}
    h::Vector{T}
    c::Vector{T}
end

"""
    policy_data(model, actions)

Assemble P^S, h^S, c^S. `actions[i]` is the mathematical gear in 0:A,
not its array position. Uncontrollable states must have action 0.
No resource-capacity coupling is imposed in this single-project routine.
"""
function policy_data(model::FiniteMultiGearModel{T}, actions::AbstractVector{<:Integer}) where {T}
    Base.require_one_based_indexing(actions)
    n = nstates(model)
    length(actions) == n || throw(DimensionMismatch("One gear per state is required."))
    p = Matrix{T}(undef, n, n)
    h, c = Vector{T}(undef, n), Vector{T}(undef, n)
    a = Vector{Int}(undef, n)
    for i in 1:n
        actions[i] in admissible_gears(model, i) || throw(ArgumentError(
            "Gear $(actions[i]) is inadmissible at state $i."
        ))
        a[i] = Int(actions[i])
        slot = a[i] + 1
        p[i,:] = model.P[i,:,slot]
        h[i], c[i] = model.h[i,slot], model.c[i,slot]
    end
    return PolicyData(a, p, h, c)
end
