"""
Diagnostics of the two solved right-hand sides, in holding-cost/resource order.
`residual_inf[k] = norm(M*x_k - b_k, Inf)` and
`backward_error[k] = residual_inf[k] / (opnorm(M,Inf)*norm(x_k,Inf) + norm(b_k,Inf))`.
A zero denominator with zero residual is assigned error zero. These are
floating-point diagnostics, NOT validated bounds on forward error or certificates.
"""
struct SolveDiagnostics{T<:AbstractFloat}
    residual_inf::Vector{T}
    backward_error::Vector{T}
end

struct DiscountedEvaluation{T<:AbstractFloat}
    model::FiniteMultiGearModel{T}
    policy::PolicyData{T}
    beta::T
    F::Vector{T}
    G::Vector{T}
    diagnostics::SolveDiagnostics{T}
end

struct AverageEvaluation{T<:AbstractFloat}
    model::FiniteMultiGearModel{T}
    policy::PolicyData{T}
    omega::Vector{T}
    phi::Vector{T}
    gamma::Vector{T}
    hbar::T
    cbar::T
    recurrent_states::Vector{Int}
    diagnostics::SolveDiagnostics{T}
end

function _direct_solve(M::Matrix{T}, B::Matrix{T}) where {T<:AbstractFloat}
    # One fresh LU factorization, reused for both right-hand sides. No inverse.
    X = lu(M; check=true) \ B
    all(isfinite, X) || error("Policy evaluation produced nonfinite values.")
    residuals = Vector{T}(undef, size(B, 2))
    errors = similar(residuals)
    matrix_norm = opnorm(M, Inf)
    for k in axes(B, 2)
        x, b = @view(X[:,k]), @view(B[:,k])
        residuals[k] = norm(M*x - b, Inf)
        denominator = matrix_norm*norm(x, Inf) + norm(b, Inf)
        isfinite(denominator) || error("Solve diagnostic overflow; rescale or use higher precision.")
        errors[k] = denominator == zero(T) ?
            (residuals[k] == zero(T) ? zero(T) : T(Inf)) : residuals[k]/denominator
    end
    all(isfinite, residuals) && all(isfinite, errors) || error(
        "Nonfinite solve diagnostics; rescale the problem or use higher precision."
    )
    return X, SolveDiagnostics(residuals, errors)
end

"""
    evaluate_discounted(model, actions; beta)

Solve (I-beta*P^S)[F G] = [h^S c^S], with 0 < beta < 1.
Returns separate discounted holding-cost/resource vectors and solve diagnostics.
A specified policy is evaluated, not optimized. No unichain assumption is needed.
This reference routine assembles and factors a fresh dense matrix at every call.
"""
function evaluate_discounted(model::FiniteMultiGearModel{T}, actions::AbstractVector{<:Integer};
                             beta::Real) where {T}
    b = T(beta)
    isfinite(b) && zero(T) < b < one(T) || throw(ArgumentError("beta must lie strictly in (0,1)."))
    p = policy_data(model, actions)
    n = nstates(model)
    M = Matrix{T}(I, n, n) - b*p.P
    X, diagnostics = _direct_solve(M, hcat(p.h, p.c))
    return DiscountedEvaluation(model, p, b, X[:,1], X[:,2], diagnostics)
end

# Iterative Kosaraju decomposition. Positive stored probabilities define edges;
# no threshold suppresses small probabilities. For dense P this costs O(N^2).
function _closed_classes(P::Matrix{T}) where {T<:AbstractFloat}
    n = size(P, 1)
    outgoing, incoming = [Int[] for _ in 1:n], [Int[] for _ in 1:n]
    for i in 1:n, j in 1:n
        if P[i,j] > zero(T)
            push!(outgoing[i], j)
            push!(incoming[j], i)
        end
    end
    seen, order = falses(n), Int[]
    for root in 1:n
        seen[root] && continue
        seen[root] = true
        nodes, next_neighbor = [root], [1]
        while !isempty(nodes)
            u = nodes[end]
            k = next_neighbor[end]
            if k > length(outgoing[u])
                push!(order, pop!(nodes))
                pop!(next_neighbor)
            else
                next_neighbor[end] = k + 1
                v = outgoing[u][k]
                if !seen[v]
                    seen[v] = true
                    push!(nodes, v)
                    push!(next_neighbor, 1)
                end
            end
        end
    end
    component, groups = zeros(Int, n), Vector{Int}[]
    for root in Iterators.reverse(order)
        component[root] != 0 && continue
        push!(groups, Int[])
        id = length(groups)
        component[root] = id
        stack = [root]
        while !isempty(stack)
            u = pop!(stack)
            push!(groups[id], u)
            for v in incoming[u]
                if component[v] == 0
                    component[v] = id
                    push!(stack, v)
                end
            end
        end
    end
    closed = trues(length(groups))
    for i in 1:n, j in outgoing[i]
        component[i] != component[j] && (closed[component[i]] = false)
    end
    return [sort!(groups[k]) for k in eachindex(groups) if closed[k]]
end

"""
    evaluate_average(model, actions; omega=nothing)

Evaluate a specified unichain policy using the augmented Poisson system
    [I-P^S  1; omega'  0] * [phi gamma; hbar cbar] = [h^S c^S; 0 0].
`omega=nothing` means omega=e_1, hence phi[1]=gamma[1]=0. A supplied finite
normalization vector must sum to 1 within model.row_atol; it need not be
nonnegative. It is not silently rescaled.

Exactly one closed communicating class of the stored P^S is required. Transient
states and periodic recurrent classes are allowed. Multichain policies are
rejected before solving. This checks THIS policy only, not all policies or
communicating/indexability/unique-optimal-bias assumptions on the controlled model.
No near-one discount substitution is used.
"""
function evaluate_average(model::FiniteMultiGearModel{T}, actions::AbstractVector{<:Integer};
                          omega::Union{Nothing,AbstractVector{<:Real}}=nothing) where {T}
    p = policy_data(model, actions)
    n = nstates(model)
    if omega === nothing
        w = zeros(T, n)
        w[1] = one(T)
    else
        Base.require_one_based_indexing(omega)
        length(omega) == n || throw(DimensionMismatch("omega must have one entry per state."))
        w = T.(omega)
        all(isfinite, w) || throw(ArgumentError("omega must be finite."))
        abs(sum(w)-one(T)) <= model.row_atol || throw(ArgumentError("omega must sum to 1."))
    end
    classes = _closed_classes(p.P)
    length(classes) == 1 || throw(ArgumentError(
        "Average evaluation requires a unichain policy; found $(length(classes)) closed classes."
    ))
    M = zeros(T, n+1, n+1)
    M[1:n,1:n] = Matrix{T}(I, n, n) - p.P
    M[1:n,n+1] .= one(T)
    M[n+1,1:n] = w
    B = zeros(T, n+1, 2)
    B[1:n,1] = p.h
    B[1:n,2] = p.c
    X, diagnostics = _direct_solve(M, B)
    return AverageEvaluation(model, p, w, X[1:n,1], X[1:n,2], X[n+1,1],
                             X[n+1,2], only(classes), diagnostics)
end

function _finite_charge(::Type{T}, lambda::Real) where {T<:AbstractFloat}
    x = T(lambda)
    isfinite(x) || throw(ArgumentError("The resource charge must be finite."))
    return x
end

"""Discounted charged policy value F + lambda*G, for any finite real charge."""
charged_value(e::DiscountedEvaluation{T}, lambda::Real) where {T} =
    e.F + _finite_charge(T, lambda)*e.G

"""Average charged rate hbar + lambda*cbar (not a statewise bias)."""
charged_rate(e::AverageEvaluation{T}, lambda::Real) where {T} =
    e.hbar + _finite_charge(T, lambda)*e.cbar

"""Normalized charged bias phi + lambda*gamma (not a discounted value)."""
charged_bias(e::AverageEvaluation{T}, lambda::Real) where {T} =
    e.phi + _finite_charge(T, lambda)*e.gamma
