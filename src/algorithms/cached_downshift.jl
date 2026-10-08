"""
M-DS result: the full-inverse baseline plus a frontier-only marginal cache.
MPI assignments are distinct from verified DAI values. Cache exposure work,
recurrence work, extra rebuild work, pivot preparation and audits are counted
separately. A completed run includes the all-zero successor evaluation.
"""
struct CachedDownshiftResult{T<:AbstractFloat}
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
    initial_factorizations::Int
    rank_one_updates::Int
    refactorizations::Int
    refactorization_attempts::Int
    exact_comparison_solves::Int
    updates::Vector{RankOneUpdateReport{T}}
    audit_evaluations::Int
    audits::Vector{MarginalCacheAuditRecord{T}}
    last_evaluation::Union{Nothing,DiscountedEvaluation{T},AverageEvaluation{T}}
    cache_records::Vector{MarginalCacheRecord}
    cache_exposures::Int
    cache_direct_evaluations::Int
    cache_recurrence_refreshes::Int
    cache_rebuild_evaluations::Int
    cache_rebuilds::Int
    cache_pivot_preparations::Int
    last_cache::Union{Nothing,FrontierMetricCache{T}}
end

"""
    cached_downshift(model; criterion=:discounted, beta=nothing, omega=nothing,
        family=UnrestrictedFamily(), audit=false, cache_rebuild_every=0, ...)

M-DS uses the existing full-inverse rank-one kernel and the same mathematical
selection and exact-tie rule as C-DS/R-DS. Retained FRONTIER pairs are refreshed;
new/reentering pairs are evaluated directly. All frontier entries are rebuilt
after a policy refactorization; cache_rebuild_every optionally requests
additional cache-only refreshes without a policy factorization.

The denominator screen uses |Delta c|+||beta*Delta p||_1*||G||_inf
(or the average bias counterpart) so retained pairs need no second length-N
dot product solely for screening. This diagnostic bound can be more
conservative than C-DS/R-DS and lead to extra exact comparison solves or an
unresolved result; it is not a rigorous error bound. No near-tie merging.

With audit=true, ONE separate fresh policy solve at the initial policy and
every accepted successor checks values, inverse, all adjacent marginals and
the active cache. These K+1 audit solves are not M-DS algorithm work.
For a complete unreconstructed run cache_exposures is B_F, and the direct
cache count equals B_F. Policy/cache rebuilds are separate extra work. A
partial run can already have prepared its last successor frontier; records
report the work actually done instead of hiding that preparation.

This is a validation baseline, not an optimized timing harness or reduced
FP-DS. A nonfinite cache stops explicitly; there is no silent cache repair.
"""
function cached_downshift(model::FiniteMultiGearModel{T};
                           criterion::Symbol=:discounted,beta=nothing,omega=nothing,
                           family::AbstractPolicyFamily=UnrestrictedFamily(),
                           screen_atol::Real=1e-11,screen_rtol::Real=1e-9,
                           guard_factor::Real=1000,exact_fallback::Bool=true,
                           exact_max_states::Integer=8,max_steps=nothing,
                           monotonicity_atol::Real=1e-10,monotonicity_rtol::Real=1e-9,
                           refactor_on_failure::Bool=true,refactor_every::Integer=0,
                           inverse_check_every::Integer=0,cache_rebuild_every::Integer=0,
                           backward_tol::Real=1e-10,inverse_tol::Real=1e-10,
                           pivot_atol::Real=0,pivot_rtol::Real=1e-12,
                           audit::Bool=false,audit_atol::Real=1e-10,
                           audit_rtol::Real=1e-8) where {T}
    b,w = _rankone_parameters(model,criterion,beta,omega)
    atol,rtol,guard = T(screen_atol),T(screen_rtol),T(guard_factor)
    all(isfinite,(atol,rtol,guard)) && atol >= 0 && rtol >= 0 && guard >= 1 ||
        throw(ArgumentError("Invalid comparison-screen settings."))
    exact_max_states >= 1 || throw(ArgumentError("exact_max_states must be positive."))
    ma,mr=T(monotonicity_atol),T(monotonicity_rtol)
    _check_monotonicity_tolerances(ma,mr)
    refactor_every >= 0 && inverse_check_every >= 0 && cache_rebuild_every >= 0 ||
        throw(ArgumentError("Rebuild and inverse-check intervals must be nonnegative."))
    bt,it,pa,pr = _rankone_tolerances(T,backward_tol,inverse_tol,pivot_atol,pivot_rtol)
    aat,art = T(audit_atol),T(audit_rtol)
    all(isfinite,(aat,art)) && aat >= 0 && art >= 0 || throw(ArgumentError("Invalid audit tolerances."))
    _check_family(model,family)
    states,A = findall(model.controllable),maxgear(model)
    K = A*length(states)
    limit = max_steps === nothing ? K : max_steps
    limit isa Integer && 0 <= limit <= K || throw(ArgumentError("max_steps must be in 0:K."))
    mpi = Matrix{Union{Missing,T}}(undef,length(states),A); fill!(mpi,missing)
    actions = [model.controllable[i] ? A : 0 for i in 1:nstates(model)]
    policies = [copy(actions)]
    steps = DownshiftStep{T}[]
    updates = RankOneUpdateReport{T}[]
    audits = MarginalCacheAuditRecord{T}[]
    cache_records = MarginalCacheRecord[]
    cache_pivot_preparations = 0
    cache = nothing
    initial_factorizations,exact_solves,audit_evaluations = 0,0,0
    state = nothing
    exact_model = nothing
    exact_reason = "Exact fallback disabled or above its state limit."
    if exact_fallback && nstates(model) <= exact_max_states
        try
            exact_model = ExactMultiGearModel(model)
        catch err
            err isa ArgumentError || rethrow()
            exact_reason = sprint(showerror,err)
        end
    end
    finish(status,message) = CachedDownshiftResult(model,criterion,b,w,family,states,mpi,steps,
        policies,status == :complete,status,message,(status==:nonmonotone ? :detected_violation : _monotonicity_screen(steps,atol,rtol,guard)),
        initial_factorizations,count(u -> u.accepted && u.method == :rank_one,updates),
        count(u -> u.accepted && u.method == :refactorization,updates),
        count(u -> u.factorization_attempted,updates),exact_solves,updates,audit_evaluations,audits,
        state === nothing ? nothing : rankone_evaluation(state),cache_records,
        sum((length(r.exposed) for r in cache_records);init=0),
        sum((r.direct_evaluations for r in cache_records);init=0),
        sum((r.recurrence_refreshes for r in cache_records);init=0),
        sum((r.rebuild_evaluations for r in cache_records);init=0),
        count(r -> r.rebuilt,cache_records),cache_pivot_preparations,cache)
    for k in 1:K
        k > limit && return finish(:step_limit,"Requested step limit reached; MPI is partial.")
        frontier = downshift_frontier(model,actions,family)
        isempty(frontier) && return finish(:empty_frontier,"No feasible downshift before the lower endpoint.")
        if state === nothing
            try
                p,M,B,recurrent = _rankone_system(model,actions,criterion,b,w)
                initial_factorizations += 1
                Z,X,diagnostics,_ = _rankone_factor(M,B,bt,it)
                state = RankOnePolicyState(model,criterion,b,w,p,M,Z,X,recurrent,diagnostics)
            catch err
                if err isa ArgumentError && occursin("unichain",sprint(showerror,err))
                    return finish(:unsupported_average_policy,sprint(showerror,err))
                elseif err isa SingularException || err isa ErrorException
                    return finish(:evaluation_failure,sprint(showerror,err))
                end
                rethrow()
            end
            cache,record = initialize_marginal_cache(rankone_evaluation(state);family=family,policy_index=0)
            push!(cache_records,record)
            _cache_finite(cache) || return finish(:unresolved_cache,"Nonfinite initial cache; no DAI is asserted.")
            if audit
                audit_evaluations += 1
                local check
                try
                    check = audit_marginal_cache(state,cache;family=family,policy_index=0,atol=aat,rtol=art,
                                          inverse_tol=it,backward_tol=bt)
                catch err
                    if err isa SingularException || err isa ErrorException
                        return finish(:audit_failure,sprint(showerror,err))
                    end
                    rethrow()
                end
                push!(audits,check)
                check.passed || return finish(:audit_failure,"Initial inverse/value/cache audit failed.")
            end
        end
        e = rankone_evaluation(state)          # no fresh policy solve here
        maximum(e.diagnostics.backward_error) <= bt ||
            return finish(:unresolved_evaluation,"Maintained policy-value residual exceeds screening limit.")
        cache.frontier == frontier || return finish(:unresolved_cache,"Stored frontier is stale.")
        f,g,scales = _cached_frontier_metrics(cache,e)
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
        pivot = prepare_marginal_pivot(state,pair[1])
        cache_pivot_preparations += 1
        report = update_downshift!(state,pair[1];refactor_on_failure=refactor_on_failure,
            force_refactor=(refactor_every > 0 && k % refactor_every == 0),
            check_inverse=(inverse_check_every > 0 && k % inverse_check_every == 0),
            backward_tol=bt,inverse_tol=it,pivot_atol=pa,pivot_rtol=pr)
        push!(updates,report)
        if !report.accepted
            status = report.reason == :unsupported_average_policy ? :unsupported_average_policy : :unresolved_pivot
            return finish(status,report.message)
        end
        push!(steps,DownshiftStep(k,copy(frontier),f,g,ratios,pair,value,exact_assignment,evidence,e.diagnostics))
        row = findfirst(==(pair[1]),states)
        mpi[row,pair[2]] = value
        actions = copy(state.policy.actions)
        push!(policies,copy(actions))
        rebuild = report.method == :refactorization ||
            (cache_rebuild_every > 0 && k % cache_rebuild_every == 0)
        reason = report.method == :refactorization ? :policy_refactorization : :scheduled_cache
        cache,record = refresh_marginal_cache(cache,pivot,rankone_evaluation(state);
            family=family,policy_index=k,rebuild=rebuild,reason=reason)
        push!(cache_records,record)
        _cache_finite(cache) || return finish(:unresolved_cache,"Nonfinite cache after accepted pivot $k; no DAI is asserted.")
        if audit
            audit_evaluations += 1
            local check
            try
                check = audit_marginal_cache(state,cache;family=family,policy_index=k,atol=aat,rtol=art,
                                      inverse_tol=it,backward_tol=bt)
            catch err
                if err isa SingularException || err isa ErrorException
                    return finish(:audit_failure,sprint(showerror,err))
                end
                rethrow()
            end
            push!(audits,check)
            check.passed || return finish(:audit_failure,"Fresh-solve audit failed after pivot $k; no DAI is asserted.")
        end
    end
    return finish(:complete,"Complete cached MPI assignments and terminal policy update; DAI equality requires separate verification.")
end
