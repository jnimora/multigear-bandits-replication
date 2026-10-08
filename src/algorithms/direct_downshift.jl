"""One completed C-DS pivot; vectors follow the lexicographically sorted frontier."""
struct DownshiftStep{T<:AbstractFloat}
    number::Int
    frontier::Vector{Tuple{Int,Int}}
    frontier_f::Vector{T}
    frontier_g::Vector{T}
    frontier_m::Vector{T}
    selected::Tuple{Int,Int}
    assignment::T
    exact_assignment::Union{Nothing,ExactRational}
    comparison_evidence::Symbol
    diagnostics::SolveDiagnostics{T}
end

"""A possibly partial MPI computation, never an implicit DAI certificate."""
struct DirectDownshiftResult{T<:AbstractFloat}
    model::FiniteMultiGearModel{T}
    criterion::Symbol
    beta::Union{Nothing,T}
    omega::Union{Nothing,Vector{T}}
    family::AbstractPolicyFamily
    states::Vector{Int}
    mpi::Matrix{Union{Missing,T}}
    steps::Vector{DownshiftStep{T}}
    policies::Vector{Vector{Int}}
    complete::Bool
    status::Symbol
    message::String
    monotonicity_screen::Symbol
    policy_evaluations::Int
    exact_comparison_solves::Int
end

# Protocol screening scales, not rigorous error bounds. The default guard
# triggers exact fallback on the small rational fixtures. No near-tie merging.
_screen_band(s,atol,rtol,guard) = guard*(atol+rtol*(one(s)+s))
function _frontier_metrics(e,frontier)
    T = eltype(e.model.h)
    v,w,factor = e isa DiscountedEvaluation ? (e.F,e.G,e.beta) : (e.phi,e.gamma,one(T))
    f,g,scale = zeros(T,length(frontier)),zeros(T,length(frontier)),zeros(T,length(frontier))
    for (k,(i,a)) in enumerate(frontier)
        dp = e.model.P[i,:,a+1]-e.model.P[i,:,a]
        dc = e.model.c[i,a+1]-e.model.c[i,a]
        f[k] = e.model.h[i,a]-e.model.h[i,a+1]-factor*dot(dp,v)
        g[k] = dc+factor*dot(dp,w)
        scale[k] = abs(dc)+abs(factor)*sum(abs.(dp .* w))
    end
    return f,g,scale
end

# Comparison safeguard only. This uses the same augmented formulation as
# C-DS, with rational LU. The independent oracle instead uses Gauss-Jordan
# and, for averaging, stationarity plus centered Poisson equations.
function _exact_frontier_comparison(q::ExactMultiGearModel,e,frontier)
    n = nstates(q)
    P = zeros(ExactRational,n,n)
    rhs = zeros(ExactRational,n,2)
    for i in 1:n
        slot = e.policy.actions[i]+1
        P[i,:] = q.P[i,:,slot]
        rhs[i,1],rhs[i,2] = q.h[i,slot],q.c[i,slot]
    end
    if e isa DiscountedEvaluation
        factor = _exact_number(e.beta)
        M = Matrix{ExactRational}(I,n,n)-factor*P
        values = lu(M; check=true) \ rhs
    else
        factor = one(ExactRational)
        omega = _exact_normalization(n,e.omega)
        M = zeros(ExactRational,n+1,n+1)
        M[1:n,1:n] = Matrix{ExactRational}(I,n,n)-P
        M[1:n,n+1] .= 1
        M[n+1,1:n] = omega
        B = zeros(ExactRational,n+1,2); B[1:n,:] = rhs
        values = (lu(M; check=true) \ B)[1:n,:]
    end
    f,g = zeros(ExactRational,length(frontier)),zeros(ExactRational,length(frontier))
    for (k,(i,a)) in enumerate(frontier)
        dp = q.P[i,:,a+1]-q.P[i,:,a]
        f[k] = q.h[i,a]-q.h[i,a+1]-factor*dot(dp,values[:,1])
        g[k] = q.c[i,a+1]-q.c[i,a]+factor*dot(dp,values[:,2])
    end
    return f,g
end

function _monotonicity_screen(steps,atol,rtol,guard)
    ambiguous = false
    all_exact = length(steps) > 1
    for k in 2:length(steps)
        left,right = steps[k-1],steps[k]
        if left.exact_assignment !== nothing && right.exact_assignment !== nothing
            right.exact_assignment >= left.exact_assignment || return :exact_violation
        else
            all_exact = false
            difference = right.assignment-left.assignment
            band = _screen_band(abs(left.assignment)+abs(right.assignment),atol,rtol,guard)
            difference < -band && return :numerical_violation
            abs(difference) <= band && (ambiguous = true)
        end
    end
    return ambiguous ? :unresolved : (all_exact ? :exact_pass : :numerical_pass)
end

"""
    direct_downshift(model; criterion=:discounted, beta=nothing, omega=nothing,
                     family=UnrestrictedFamily(), screen_atol=1e-11,
                     screen_rtol=1e-9, guard_factor=1000,
                     exact_fallback=true, exact_max_states=8,
                     max_steps=nothing)

Fresh dense policy evaluation and direct FRONTIER metrics at every iteration.
Start with gear A at each controllable state; select the minimum positive-
denominator MP ratio, breaking exact ties lexicographically; lower that gear
by one. Uncontrollable states retain gear zero.

Clearly negative-work frontier candidates are skipped. Near-zero work signs
must be resolved by the exact safeguard or stop as unresolved; exactly
nonpositive work is excluded. No positive candidate, empty frontier, unsupported
average policy, and unresolved comparisons have distinct partial-output statuses.
MPI monotonicity is checked before each assignment/pivot; a resolved decrease
stops with :nonmonotone and preserves the accepted prefix. This does not by
itself establish nonindexability.

Close comparisons trigger an exact rational policy solve when the small stored
model is exactly stochastic. Otherwise the run returns unresolved. Equality of
floating ratios alone is NOT accepted as a mathematical tie. The production
256/512/1024-bit retry ladder is not implemented in this milestone.

The result records K selection-policy evaluations on a complete K-step run.
The final all-zero policy is recorded but NOT evaluated in the index loop.
Independent terminal-policy evaluation and certificates are postprocessing.
Exact comparison solves are counted separately. This is not a timing harness.
"""
function direct_downshift(model::FiniteMultiGearModel{T};
                           criterion::Symbol=:discounted,beta=nothing,omega=nothing,
                           family::AbstractPolicyFamily=UnrestrictedFamily(),
                           screen_atol::Real=1e-11,screen_rtol::Real=1e-9,
                           guard_factor::Real=1000,exact_fallback::Bool=true,
                           exact_max_states::Integer=8,max_steps=nothing,
                           monotonicity_atol::Real=1e-10,monotonicity_rtol::Real=1e-9) where {T}
    criterion in (:discounted,:average) || throw(ArgumentError("Unknown criterion."))
    atol,rtol,guard = T(screen_atol),T(screen_rtol),T(guard_factor)
    all(isfinite,(atol,rtol,guard)) && atol >= 0 && rtol >= 0 && guard >= 1 ||
        throw(ArgumentError("Invalid comparison-screen settings."))
    exact_max_states >= 1 || throw(ArgumentError("exact_max_states must be positive."))
    ma,mr=T(monotonicity_atol),T(monotonicity_rtol)
    _check_monotonicity_tolerances(ma,mr)
    b = nothing
    w = nothing
    if criterion == :discounted
        beta === nothing && throw(ArgumentError("A discount factor is required."))
        b = T(beta)
        isfinite(b) && 0 < b < 1 || throw(ArgumentError("beta must lie in (0,1)."))
        omega === nothing || throw(ArgumentError("omega is only for the average criterion."))
    else
        beta === nothing || throw(ArgumentError("Do not supply beta for the average criterion."))
        w = omega === nothing ? zeros(T,nstates(model)) : T.(omega)
        omega === nothing && (w[1] = 1)
        length(w) == nstates(model) && all(isfinite,w) && abs(sum(w)-1) <= model.row_atol ||
            throw(ArgumentError("Invalid bias normalization."))
    end
    _check_family(model,family)
    states,A = findall(model.controllable),maxgear(model)
    K = A*length(states)
    limit = max_steps === nothing ? K : max_steps
    limit isa Integer && 0 <= limit <= K || throw(ArgumentError("max_steps must be in 0:K."))
    mpi = Matrix{Union{Missing,T}}(undef,length(states),A); fill!(mpi,missing)
    actions = [model.controllable[i] ? A : 0 for i in 1:nstates(model)]
    policies = [copy(actions)]
    steps = DownshiftStep{T}[]
    evaluations = 0
    exact_solves = 0
    exact_model = nothing
    exact_reason = "Exact fallback disabled or above its state limit."
    if exact_fallback && nstates(model) <= exact_max_states
        try
            exact_model = ExactMultiGearModel(model)
        catch error
            error isa ArgumentError || rethrow()
            exact_reason = sprint(showerror,error)
        end
    end
    finish(status,message) = DirectDownshiftResult(model,criterion,b,w,family,states,mpi,steps,
        policies,status == :complete,status,message,
        (status==:nonmonotone ? :detected_violation : _monotonicity_screen(steps,atol,rtol,guard)),evaluations,exact_solves)
    for k in 1:K
        k > limit && return finish(:step_limit,"Requested step limit reached; MPI is partial.")
        frontier = downshift_frontier(model,actions,family)
        isempty(frontier) && return finish(:empty_frontier,"No feasible downshift before the lower endpoint.")
        evaluations += 1
        local e
        try
            e = criterion == :discounted ? evaluate_discounted(model,actions; beta=b) :
                                          evaluate_average(model,actions; omega=w)
        catch error
            if error isa ArgumentError && occursin("unichain",sprint(showerror,error))
                return finish(:unsupported_average_policy,sprint(showerror,error))
            elseif error isa SingularException || error isa ErrorException
                return finish(:evaluation_failure,sprint(showerror,error))
            end
            rethrow()
        end
        maximum(e.diagnostics.backward_error) <= T(1e-10) ||
            return finish(:unresolved_evaluation,"Policy-evaluation backward error exceeds screening limit.")
        f,g,scales = _frontier_metrics(e,frontier)
        all(isfinite,f) && all(isfinite,g) && all(isfinite,scales) ||
            return finish(:unresolved_comparison,"Nonfinite frontier metrics or scales.")
        bands = [_screen_band(s,atol,rtol,guard) for s in scales]
        selected,ratios,need_exact = _positive_frontier_screen(f,g,bands,atol,rtol,guard)
        selected==0 && !need_exact && return finish(:no_positive_candidate,
            "No feasible downshift has resolved positive marginal work; MPI is partial.")
        exact_assignment = nothing
        evidence = :numerical_screen
        if need_exact
            exact_model === nothing && return finish(:unresolved_comparison,
                "A denominator or ratio comparison is unresolved. "*exact_reason)
            exact_solves += 1
            local ef,eg
            try
                ef,eg = _exact_frontier_comparison(exact_model,e,frontier)
            catch error
                if error isa ArgumentError || error isa SingularException
                    return finish(:unresolved_comparison,"Exact comparison unavailable: "*sprint(showerror,error))
                end
                rethrow()
            end
            selected = _exact_positive_minimum(ef,eg)
            selected==0 && return finish(:no_positive_candidate,
                "No feasible downshift has exactly positive marginal work; MPI is partial.")
            exact_assignment = ef[selected]/eg[selected]
            ratios = [eg[j]>0 ? T(ef[j]/eg[j]) : T(NaN) for j in eachindex(eg)]
            all(j->eg[j]<=0 || isfinite(ratios[j]),eachindex(eg)) ||
                return finish(:unresolved_comparison,"Exact ratios overflow the output precision.")
            evidence = :exact_comparison
        end
        pair = frontier[selected]
        value = ratios[selected]
        if !isempty(steps) && !_mpi_nondecreasing(steps[end].assignment,value,ma,mr,
                steps[end].exact_assignment,exact_assignment)
            return finish(:nonmonotone,
                "MPI would decrease from $(steps[end].assignment) to $value at step $k; the assignment and pivot were not accepted.")
        end
        push!(steps,DownshiftStep(k,copy(frontier),f,g,ratios,pair,value,exact_assignment,evidence,e.diagnostics))
        row = findfirst(==(pair[1]),states)
        mpi[row,pair[2]] = value
        actions[pair[1]] -= 1
        push!(policies,copy(actions))
    end
    return finish(:complete,"Complete MPI assignments; DAI equality requires separate verification.")
end
