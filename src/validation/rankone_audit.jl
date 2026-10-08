"""
Numerical audit of one maintained policy against the original direct evaluator.
Errors use max |x-y|/(1+max(|x|,|y|)), elementwise. Average column errors include
both biases and rates; rate errors are also reported separately. Full inverse
residuals and fresh direct solves are validation costs, not R-DS loop work.
"""
struct RankOneAuditRecord{T<:AbstractFloat}
    policy_index::Int
    actions::Vector{Int}
    holding_column_error::T
    resource_column_error::T
    holding_rate_error::Union{Nothing,T}
    resource_rate_error::Union{Nothing,T}
    marginal_f_error::T
    marginal_g_error::T
    marginal_ratio_error::Union{Nothing,T}
    ratios_checked::Int
    inverse_left_residual::T
    inverse_right_residual::T
    inverse_backward_error::T
    max_value_backward_error::T
    passed::Bool
end

function _rankone_scaled_error(x,y)
    return maximum(abs.(x.-y)./(one(eltype(x)).+max.(abs.(x),abs.(y))))
end
function _rankone_close(x,y,atol,rtol)
    return size(x) == size(y) && all(isfinite,x) && all(isfinite,y) &&
        all(abs.(x.-y) .<= atol .+ rtol.*max.(abs.(x),abs.(y)))
end

"""
    audit_rankone(state; policy_index=0, atol=1e-10, rtol=1e-8,
                   inverse_tol=1e-10, backward_tol=1e-10)

Use evaluate_discounted/evaluate_average, NOT an inverse-based reconstruction,
to audit the maintained policy. All adjacent f and g are compared, even outside
the feasible frontier or with g <= 0. Ratios are additionally compared only
where BOTH denominators are safely positive on the stated numerical scale.
The audit does not certify indexability and does not repair a failing state.
"""
function audit_rankone(state::RankOnePolicyState{T};policy_index::Integer=0,
                       atol::Real=1e-10,rtol::Real=1e-8,inverse_tol::Real=1e-10,
                       backward_tol::Real=1e-10) where {T}
    at,rt = T(atol),T(rtol)
    all(isfinite,(at,rt)) && at >= 0 && rt >= 0 || throw(ArgumentError("Invalid audit tolerances."))
    bt,it,_,_ = _rankone_tolerances(T,backward_tol,inverse_tol,0,0)
    policy_index >= 0 || throw(ArgumentError("policy_index must be nonnegative."))
    e = rankone_evaluation(state)
    reference = state.criterion == :discounted ?
        evaluate_discounted(state.model,state.policy.actions;beta=state.beta) :
        evaluate_average(state.model,state.policy.actions;omega=state.omega)
    rate_h,rate_c = nothing,nothing
    if state.criterion == :discounted
        reference_columns = hcat(reference.F,reference.G)
    else
        reference_columns = vcat(hcat(reference.phi,reference.gamma),
                             reshape(T[reference.hbar,reference.cbar],1,2))
        rate_h = abs(e.hbar-reference.hbar)/(one(T)+max(abs(e.hbar),abs(reference.hbar)))
        rate_c = abs(e.cbar-reference.cbar)/(one(T)+max(abs(e.cbar),abs(reference.cbar)))
    end
    # "reference" here is an independent FLOATING solve, not exact arithmetic.
    herror = _rankone_scaled_error(state.values[:,1],reference_columns[:,1])
    cerror = _rankone_scaled_error(state.values[:,2],reference_columns[:,2])
    metrics,refmetrics = adjacent_marginals(e),adjacent_marginals(reference)
    ferror = _rankone_scaled_error(metrics.f,refmetrics.f)
    gerror = _rankone_scaled_error(metrics.g,refmetrics.g)
    ratio_error,ratios_checked = nothing,0
    ratios_ok = true
    for j in eachindex(metrics.g)
        g1,g2 = metrics.g[j],refmetrics.g[j]
        band = T(100)*(at+rt*(one(T)+max(abs(g1),abs(g2))))
        if g1 > band && g2 > band
            r1,r2 = metrics.f[j]/g1,refmetrics.f[j]/g2
            discrepancy = abs(r1-r2)/(one(T)+max(abs(r1),abs(r2)))
            ratio_error = ratio_error === nothing ? discrepancy : max(ratio_error,discrepancy)
            ratios_checked += 1
            ratios_ok &= isfinite(r1) && isfinite(r2) && abs(r1-r2) <= at+rt*max(abs(r1),abs(r2))
        end
    end
    # Assemble the audit matrix from the ORIGINAL evaluator's policy, rather
    # than trusting the workspace's maintained system as the reference.
    p,n = reference.policy,nstates(state.model)
    if state.criterion == :discounted
        M = Matrix{T}(I,n,n)-state.beta*p.P
        B = hcat(p.h,p.c)
    else
        M = zeros(T,n+1,n+1)
        M[1:n,1:n] = Matrix{T}(I,n,n)-p.P
        M[1:n,n+1] .= one(T)
        M[n+1,1:n] = reference.omega
        B = zeros(T,n+1,2); B[1:n,1] = p.h; B[1:n,2] = p.c
    end
    left,right,inverse_error = _rankone_inverse_errors(M,state.inverse)
    diagnostics = _rankone_diagnostics(M,B,state.values)
    value_error = maximum(diagnostics.backward_error)
    passed = _rankone_close(state.values,reference_columns,at,rt) &&
        _rankone_close(metrics.f,refmetrics.f,at,rt) &&
        _rankone_close(metrics.g,refmetrics.g,at,rt) && ratios_ok &&
        _rankone_close(state.system,M,at,rt) && inverse_error <= it && value_error <= bt
    return RankOneAuditRecord(Int(policy_index),copy(state.policy.actions),herror,cerror,
        rate_h,rate_c,ferror,gerror,ratio_error,ratios_checked,left,right,inverse_error,value_error,passed)
end
