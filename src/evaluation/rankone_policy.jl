"""
Full-state rank-one policy-evaluation workspace. Treat all fields as read-only
outside the update routines. There is no marginal-metric cache. The inverse
has dimension N for discounting, N+1 for the augmented average formulation.
Each independent job must own its own workspace.
"""
mutable struct RankOnePolicyState{T<:AbstractFloat}
    model::FiniteMultiGearModel{T}
    criterion::Symbol
    beta::Union{Nothing,T}
    omega::Union{Nothing,Vector{T}}
    policy::PolicyData{T}
    system::Matrix{T}
    inverse::Matrix{T}
    values::Matrix{T}
    recurrent_states::Vector{Int}
    diagnostics::SolveDiagnostics{T}
end

"""One attempted row update. Failed attempts leave the workspace unchanged."""
struct RankOneUpdateReport{T<:AbstractFloat}
    selected::Tuple{Int,Int}
    accepted::Bool
    method::Symbol
    reason::Symbol
    denominator::T
    denominator_scale::T
    rank_one_attempted::Bool
    factorization_attempted::Bool
    max_backward_error::T
    inverse_backward_error::Union{Nothing,T}
    message::String
end

# Parameter validation is shared by the public workspace and R-DS runner.
function _rankone_parameters(model::FiniteMultiGearModel{T},criterion,beta,omega) where {T}
    criterion in (:discounted,:average) || throw(ArgumentError("Unknown criterion."))
    if criterion == :discounted
        beta === nothing && throw(ArgumentError("A discount factor is required."))
        beta isa Real || throw(ArgumentError("beta must be real."))
        b = T(beta)
        isfinite(b) && zero(T) < b < one(T) || throw(ArgumentError("beta must lie in (0,1)."))
        omega === nothing || throw(ArgumentError("omega is only for the average criterion."))
        return b,nothing
    end
    beta === nothing || throw(ArgumentError("Do not supply beta for the average criterion."))
    n = nstates(model)
    if omega === nothing
        w = zeros(T,n); w[1] = one(T)
    else
        omega isa AbstractVector{<:Real} || throw(ArgumentError("omega must be a real vector."))
        Base.require_one_based_indexing(omega)
        length(omega) == n || throw(DimensionMismatch("omega must have one entry per state."))
        w = T.(omega)
        all(isfinite,w) && abs(sum(w)-one(T)) <= model.row_atol ||
            throw(ArgumentError("Invalid bias normalization."))
    end
    return nothing,w
end

function _rankone_system(model::FiniteMultiGearModel{T},actions,criterion,b,w) where {T}
    p = policy_data(model,actions)
    n = nstates(model)
    if criterion == :discounted
        return p,Matrix{T}(I,n,n)-b*p.P,hcat(p.h,p.c),Int[]
    end
    classes = _closed_classes(p.P)
    length(classes) == 1 || throw(ArgumentError(
        "Average rank-one evaluation requires a unichain policy; found $(length(classes)) closed classes."))
    M = zeros(T,n+1,n+1)
    M[1:n,1:n] = Matrix{T}(I,n,n)-p.P
    M[1:n,n+1] .= one(T)
    M[n+1,1:n] = w
    B = zeros(T,n+1,2)
    B[1:n,1] = p.h; B[1:n,2] = p.c
    return p,M,B,only(classes)
end

function _rankone_diagnostics(M::Matrix{T},B::Matrix{T},X::Matrix{T}) where {T}
    all(isfinite,X) || error("Nonfinite rank-one policy values.")
    residuals,errors = zeros(T,2),zeros(T,2)
    mn = opnorm(M,Inf)
    for j in 1:2
        x,b = @view(X[:,j]),@view(B[:,j])
        residuals[j] = norm(M*x-b,Inf)
        denom = mn*norm(x,Inf)+norm(b,Inf)
        isfinite(denom) || error("Rank-one residual scale overflow; rescale or increase precision.")
        errors[j] = iszero(denom) ? (iszero(residuals[j]) ? zero(T) : T(Inf)) : residuals[j]/denom
    end
    all(isfinite,residuals) && all(isfinite,errors) || error("Nonfinite rank-one residual diagnostics.")
    return SolveDiagnostics(residuals,errors)
end

# Full inverse residuals cost O(d^3): used at initialization/refactorization,
# on a requested check, and in audit mode; NOT silently at every normal pivot.
function _rankone_inverse_errors(M::Matrix{T},Z::Matrix{T}) where {T}
    d = size(M,1)
    all(isfinite,Z) || return (T(Inf),T(Inf),T(Inf))
    identity = Matrix{T}(I,d,d)
    left = opnorm(M*Z-identity,Inf)
    right = opnorm(Z*M-identity,Inf)
    scale = one(T)+opnorm(M,Inf)*opnorm(Z,Inf)
    isfinite(scale) && all(isfinite,(left,right)) || return (left,right,T(Inf))
    return left,right,max(left,right)/scale
end

function _rankone_tolerances(::Type{T},backward_tol,inverse_tol,pivot_atol,pivot_rtol) where {T}
    bt,it,pa,pr = T(backward_tol),T(inverse_tol),T(pivot_atol),T(pivot_rtol)
    all(isfinite,(bt,it,pa,pr)) && bt > 0 && it > 0 && pa >= 0 && pr >= 0 ||
        throw(ArgumentError("Invalid rank-one safeguard tolerances."))
    return bt,it,pa,pr
end

function _rankone_factor(M::Matrix{T},B::Matrix{T},bt,it) where {T}
    fac = lu(M;check=true)                     # one factorization, multiple RHS
    Z = fac \ Matrix{T}(I,size(M,1),size(M,1))
    X = fac \ B
    diagnostics = _rankone_diagnostics(M,B,X)
    maximum(diagnostics.backward_error) <= bt || error("Fresh factorization failed the value-residual screen.")
    _,_,inverse_error = _rankone_inverse_errors(M,Z)
    inverse_error <= it || error("Fresh factorization failed the inverse-residual screen.")
    return Z,X,diagnostics,inverse_error
end

"""
    initialize_rankone(model, actions; criterion=:discounted, beta=nothing,
                       omega=nothing, backward_tol=1e-10, inverse_tol=1e-10)

Initialize the full inverse and both value columns with ONE fresh LU. For
average evaluation columns are [phi; hbar] and [gamma; cbar], with fixed
omega normalization. Only this policy is checked for unichainness. No
indexability, resource positivity, or optimality is inferred.
"""
function initialize_rankone(model::FiniteMultiGearModel{T},actions::AbstractVector{<:Integer};
                            criterion::Symbol=:discounted,beta=nothing,omega=nothing,
                            backward_tol::Real=1e-10,inverse_tol::Real=1e-10) where {T}
    b,w = _rankone_parameters(model,criterion,beta,omega)
    bt,it,_,_ = _rankone_tolerances(T,backward_tol,inverse_tol,0,0)
    p,M,B,recurrent = _rankone_system(model,actions,criterion,b,w)
    Z,X,diagnostics,_ = _rankone_factor(M,B,bt,it)
    return RankOnePolicyState(model,criterion,b,w,p,M,Z,X,recurrent,diagnostics)
end

"""Return a standard evaluation snapshot from the maintained values; no solve."""
function rankone_evaluation(state::RankOnePolicyState{T}) where {T}
    p = state.policy
    # The workspace REPLACES its policy and diagnostics on each accepted update;
    # value-vector slices are copies. Older evaluation snapshots remain valid.
    if state.criterion == :discounted
        return DiscountedEvaluation(state.model,p,state.beta,state.values[:,1],
                                    state.values[:,2],state.diagnostics)
    end
    n = nstates(state.model)
    return AverageEvaluation(state.model,p,copy(state.omega),state.values[1:n,1],
        state.values[1:n,2],state.values[n+1,1],state.values[n+1,2],
        copy(state.recurrent_states),state.diagnostics)
end

"""
    update_downshift!(state, i; refactor_on_failure=true, force_refactor=false,
                      check_inverse=false, backward_tol=1e-10,
                      inverse_tol=1e-10, pivot_atol=0, pivot_rtol=1e-12)

Lower the CURRENT gear at state i by one. Forward differences are high minus
low. With d=beta*Delta p (or [Delta p;0] for averaging), y=Z*e_i,
z=d'*Z and eta=1+d'*y, use OLD data throughout:
    Znew = Z-y*z/eta;  xh_new=xh+(f/eta)*y;  xc_new=xc-(g/eta)*y.
No positivity of f or g is required by this policy-evaluation identity.

The successor is assembled from the stored primitives, not by accumulating
row-roundoff. An identically zero RHS is assigned its exact zero solution;
no tolerance-based clipping of values, metrics, or denominators is used. Average successors are checked for unichainness before use.
A small/nonpositive denominator, nonfinite update, or excessive value
backward error triggers a fresh successor factorization when enabled.
check_inverse adds cubic full-inverse residual checks. force_refactor supports
explicitly scheduled rebuilds. The report distinguishes these operations.

The update is transactional for ordinary numerical failures: no workspace
field is changed until the successor passes the requested screens. Screens
are numerical diagnostics, not rigorous forward-error or positivity bounds.
"""
function update_downshift!(state::RankOnePolicyState{T},i::Integer;
                           refactor_on_failure::Bool=true,force_refactor::Bool=false,
                           check_inverse::Bool=false,backward_tol::Real=1e-10,
                           inverse_tol::Real=1e-10,pivot_atol::Real=0,
                           pivot_rtol::Real=1e-12) where {T}
    bt,it,pa,pr = _rankone_tolerances(T,backward_tol,inverse_tol,pivot_atol,pivot_rtol)
    1 <= i <= nstates(state.model) || throw(BoundsError(state.policy.actions,i))
    a = state.policy.actions[i]
    state.model.controllable[i] && a > 0 || throw(ArgumentError("No admissible downshift at state $i."))
    pair = (Int(i),a)
    eta,scale = T(NaN),T(NaN)
    attempted,factored = false,false
    inv_error = nothing
    error_value = T(Inf)
    report(accepted,method,reason,message) = RankOneUpdateReport(pair,accepted,method,reason,
        eta,scale,attempted,factored,error_value,inv_error,message)
    successor = copy(state.policy.actions); successor[i] -= 1
    local p,M,B,recurrent
    try
        p,M,B,recurrent = _rankone_system(state.model,successor,state.criterion,state.beta,state.omega)
    catch err
        if err isa ArgumentError && occursin("unichain",sprint(showerror,err))
            return report(false,:failed,:unsupported_average_policy,sprint(showerror,err))
        end
        rethrow()
    end
    n = nstates(state.model)
    d = zeros(T,size(state.inverse,1))
    factor = state.criterion == :discounted ? state.beta : one(T)
    d[1:n] = factor*(state.model.P[i,:,a+1]-state.model.P[i,:,a])
    y = copy(@view(state.inverse[:,i]))        # MUST be copied before modifying Z
    z = vec(transpose(d)*state.inverse)
    eta = one(T)+dot(d,y)
    scale = one(T)+sum(abs.(d .* y))
    f = state.model.h[i,a]-state.model.h[i,a+1]-dot(d,@view(state.values[:,1]))
    g = state.model.c[i,a+1]-state.model.c[i,a]+dot(d,@view(state.values[:,2]))
    reason = force_refactor ? :scheduled : :none
    local Znew,Xnew,diagnostics
    if !force_refactor
        if !(isfinite(eta) && isfinite(scale) && eta > pa+pr*scale)
            reason = :small_or_nonpositive_denominator
        elseif !(all(isfinite,y) && all(isfinite,z) && isfinite(f) && isfinite(g))
            reason = :nonfinite_update
        else
            attempted = true
            Znew = state.inverse-(y/eta)*transpose(z)
            Xnew = copy(state.values)
            Xnew[:,1] .+= (f/eta).*y
            Xnew[:,2] .-= (g/eta).*y
            # Nonsingularity and an IDENTICALLY zero RHS imply an exactly zero
            # solution. Enforce that identity to remove cancellation remnants
            # (not tolerance-based clipping), notably G=0 at a zero-resource
            # terminal policy. It also avoids a meaningless relative residual
            # test against a roundoff-only vector and zero RHS.
            for column in 1:2
                all(iszero,@view(B[:,column])) && (Xnew[:,column] .= zero(T))
            end
            if !(all(isfinite,Znew) && all(isfinite,Xnew))
                reason = :nonfinite_update
            else
                try
                    diagnostics = _rankone_diagnostics(M,B,Xnew)
                    error_value = maximum(diagnostics.backward_error)
                    error_value > bt && (reason = :value_residual)
                catch err
                    err isa ErrorException || rethrow()
                    reason = :value_residual
                end
                if reason == :none && check_inverse
                    _,_,inv_error = _rankone_inverse_errors(M,Znew)
                    inv_error > it && (reason = :inverse_residual)
                end
            end
        end
    end
    method = :rank_one
    if reason != :none
        if !(refactor_on_failure || force_refactor)
            return report(false,:failed,reason,"Rank-one update rejected; refactorization disabled. State unchanged.")
        end
        factored = true
        try
            Znew,Xnew,diagnostics,inv_error = _rankone_factor(M,B,bt,it)
            error_value = maximum(diagnostics.backward_error)
        catch err
            if err isa SingularException || err isa ErrorException
                return report(false,:failed,:refactorization_failure,sprint(showerror,err))
            end
            rethrow()
        end
        method = :refactorization
    end
    state.policy = p
    state.system = M
    state.inverse = Znew
    state.values = Xnew
    state.recurrent_states = recurrent
    state.diagnostics = diagnostics
    return report(true,method,reason,method == :rank_one ?
        "Accepted full-inverse and two-column rank-one update." :
        "Accepted fresh successor factorization; trigger recorded in reason.")
end
