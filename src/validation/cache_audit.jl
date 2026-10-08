"""Combined audit using one independent direct policy solve; no cache repair."""
struct MarginalCacheAuditRecord{T<:AbstractFloat}
    policy::RankOneAuditRecord{T}
    frontier::Vector{Tuple{Int,Int}}
    cache_f_error::T
    cache_g_error::T
    cache_ratio_error::Union{Nothing,T}
    ratios_checked::Int
    cache_structure_valid::Bool
    screening_scale_valid::Bool
    passed::Bool
end

# The original rankone_audit.jl is unchanged. The following private calculation
# mirrors its policy audit, but receives the single direct solve also used for
# the cache check. This routine is not exported and performs no extra solve.
function _cache_policy_audit(state::RankOnePolicyState{T},reference;policy_index::Integer=0,
                       atol::Real=1e-10,rtol::Real=1e-8,inverse_tol::Real=1e-10,
                       backward_tol::Real=1e-10) where {T}
    at,rt = T(atol),T(rtol)
    all(isfinite,(at,rt)) && at >= 0 && rt >= 0 || throw(ArgumentError("Invalid audit tolerances."))
    bt,it,_,_ = _rankone_tolerances(T,backward_tol,inverse_tol,0,0)
    policy_index >= 0 || throw(ArgumentError("policy_index must be nonnegative."))
    e = rankone_evaluation(state)
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

"""
    audit_marginal_cache(state, cache; family=UnrestrictedFamily(), ...)

Compare maintained policy values, inverse and all adjacent metrics with a
fresh direct solve; compare the cache to independently reconstructed current
frontier metrics from that SAME solve. Check the stored fixed rows, constants,
policy labels and family frontier as well, not just numerical ratios. Empty
terminal caches are supported. A failed check is not a nonindexability result.
"""
function audit_marginal_cache(state::RankOnePolicyState{T},cache::FrontierMetricCache{T};
                              family::AbstractPolicyFamily=UnrestrictedFamily(),
                              policy_index::Integer=0,atol::Real=1e-10,rtol::Real=1e-8,
                              inverse_tol::Real=1e-10,backward_tol::Real=1e-10) where {T}
    at,rt = T(atol),T(rtol)
    all(isfinite,(at,rt)) && at >= 0 && rt >= 0 || throw(ArgumentError("Invalid audit tolerances."))
    policy_index >= 0 || throw(ArgumentError("policy_index must be nonnegative."))
    reference = state.criterion == :discounted ?
        evaluate_discounted(state.model,state.policy.actions;beta=state.beta) :
        evaluate_average(state.model,state.policy.actions;omega=state.omega)
    pa = _cache_policy_audit(state,reference;policy_index=policy_index,atol=at,rtol=rt,
        inverse_tol=inverse_tol,backward_tol=backward_tol)
    _check_family(state.model,family)
    frontier = downshift_frontier(state.model,state.policy.actions,family)
    n,m = nstates(state.model),length(frontier)
    structure = cache.model === state.model && cache.criterion == state.criterion &&
        cache.beta == state.beta && cache.omega == state.omega &&
        cache.actions == state.policy.actions && cache.frontier == frontier &&
        all(length(x) == m for x in (cache.rows,cache.immediate_f,cache.immediate_g,cache.row_l1,cache.f,cache.g)) &&
        all(length(row) == n for row in cache.rows)
    fe,ge,ratio_error,checked = T(Inf),T(Inf),nothing,0
    scale_ok,values_ok,ratios_ok = false,false,true
    if structure
        factor = state.criterion == :discounted ? state.beta : one(T)
        for (k,(i,a)) in enumerate(frontier)
            expected_row = factor*(state.model.P[i,:,a+1]-state.model.P[i,:,a])
            structure &= cache.rows[k] == expected_row &&
                cache.row_l1[k] == sum(abs,expected_row) &&
                cache.immediate_f[k] == state.model.h[i,a]-state.model.h[i,a+1] &&
                cache.immediate_g[k] == state.model.c[i,a+1]-state.model.c[i,a]
        end
        rf,rg,direct_scales = _frontier_metrics(reference,frontier)
        if m == 0
            fe,ge = zero(T),zero(T)
        else
            fe,ge = _rankone_scaled_error(cache.f,rf),_rankone_scaled_error(cache.g,rg)
        end
        values_ok = _rankone_close(cache.f,rf,at,rt) && _rankone_close(cache.g,rg,at,rt)
        # Evaluate the l1/Inf scale at the REFERENCE values to test the bound
        # independently of value-update error. Actual algorithm scales still
        # use maintained values and remain only numerical diagnostics.
        _,rw = _cache_vectors(reference)
        bound = abs.(cache.immediate_g) .+ cache.row_l1.*norm(rw,Inf)
        scale_ok = all(isfinite,bound) && all(bound .>= zero(T)) &&
            all(direct_scales .<= bound .+ at .+ rt.*max.(abs.(bound),abs.(direct_scales)))
        for j in eachindex(rg)
            band = T(100)*(at+rt*(one(T)+max(abs(rg[j]),abs(cache.g[j]))))
            if rg[j] > band && cache.g[j] > band
                x,y = cache.f[j]/cache.g[j],rf[j]/rg[j]
                discrepancy = abs(x-y)/(one(T)+max(abs(x),abs(y)))
                ratio_error = ratio_error === nothing ? discrepancy : max(ratio_error,discrepancy)
                checked += 1
                ratios_ok &= isfinite(x) && isfinite(y) && abs(x-y) <= at+rt*max(abs(x),abs(y))
            end
        end
    end
    return MarginalCacheAuditRecord(pa,copy(frontier),fe,ge,ratio_error,checked,structure,scale_ok,
        pa.passed && structure && values_ok && ratios_ok && scale_ok && _cache_finite(cache))
end
