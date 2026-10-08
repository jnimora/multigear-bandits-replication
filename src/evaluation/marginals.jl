"""
Adjacent metrics; row r corresponds to actual state `states[r]` and column a
to the increment a-1 -> a. f is the holding-cost reduction; g is the resource
increase. m is f/g only when g > positivity_tol, and `missing` otherwise.
These are policy-dependent MP metrics, NOT a computed DAI or a PCL certificate.
"""
struct MarginalMetrics{T<:AbstractFloat}
    states::Vector{Int}
    f::Matrix{T}
    g::Matrix{T}
    m::Matrix{Union{Missing,T}}
    positivity_tol::T
end

"""
    policy_qvalues(evaluation, lambda)

One-step charged values with the evaluated policy as common continuation.
For discounting: h + lambda*c + beta*p*(F+lambda*G).
For averaging: h + lambda*c + p*(phi+lambda*gamma).

Returns an N-by-(A+1) matrix using slot a+1 for gear a. Unavailable gears
at uncontrollable states are `missing`. These are NOT optimal Q-values unless
the continuation policy has separately been established to be Bellman optimal.
"""
function policy_qvalues(e::Union{DiscountedEvaluation{T},AverageEvaluation{T}},
                        lambda::Real) where {T}
    l = _finite_charge(T, lambda)
    model = e.model
    v, factor = e isa DiscountedEvaluation ? (charged_value(e, l), e.beta) :
                                              (charged_bias(e, l), one(T))
    Q = Matrix{Union{Missing,T}}(undef, nstates(model), maxgear(model)+1)
    fill!(Q, missing)
    for i in 1:nstates(model), a in admissible_gears(model, i)
        Q[i,a+1] = model.h[i,a+1] + l*model.c[i,a+1] +
                   factor*dot(@view(model.P[i,:,a+1]), v)
    end
    return Q
end

"""
    adjacent_marginals(evaluation; positivity_tol=0)

Recover all admissible adjacent f,g from one fresh full policy evaluation.
Discounted: f=-Delta h-beta*Delta p*F; g=Delta c+beta*Delta p*G.
Average: f=-Delta h-Delta p*phi; g=Delta c+Delta p*gamma.

`positivity_tol` is finite and nonnegative. The default tests the sign of the
computed g against zero, not validated positivity. Nonpositive or unresolved
comparisons are retained (with m=missing), not silently discarded.
"""
function adjacent_marginals(e::Union{DiscountedEvaluation{T},AverageEvaluation{T}};
                            positivity_tol::Real=0) where {T}
    tol = T(positivity_tol)
    isfinite(tol) && tol >= zero(T) || throw(ArgumentError(
        "positivity_tol must be finite and nonnegative."
    ))
    model = e.model
    states = findall(model.controllable)
    nc, A = length(states), maxgear(model)
    f, g = Matrix{T}(undef, nc, A), Matrix{T}(undef, nc, A)
    m = Matrix{Union{Missing,T}}(undef, nc, A)
    fill!(m, missing)
    v, w, factor = e isa DiscountedEvaluation ? (e.F, e.G, e.beta) :
                                                (e.phi, e.gamma, one(T))
    for (r, i) in enumerate(states), a in 1:A
        # Column a is gear a-1; column a+1 is gear a.
        dp = model.P[i,:,a+1] - model.P[i,:,a]
        f[r,a] = model.h[i,a] - model.h[i,a+1] - factor*dot(dp, v)
        g[r,a] = model.c[i,a+1] - model.c[i,a] + factor*dot(dp, w)
        isfinite(f[r,a]) && isfinite(g[r,a]) || error("Nonfinite marginal metric.")
        if g[r,a] > tol
            ratio = f[r,a]/g[r,a]
            isfinite(ratio) || error("Nonfinite MP ratio; use higher precision or rescale.")
            m[r,a] = ratio
        end
    end
    return MarginalMetrics(states, f, g, m, tol)
end
