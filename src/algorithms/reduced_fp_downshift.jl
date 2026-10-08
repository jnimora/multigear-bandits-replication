# Greedy orchestration uses ORIGINAL state labels, never physical-slot ordering.
@inline _fp_feasible(w,::UnrestrictedFamily,i)=true
@inline function _fp_feasible(w,::OrderedThresholdFamily,i)
    p=w.predecessor[i]
    return p==0 || w.actions[p]<w.actions[i]
end
function _fp_feasible(w,f::ExplicitPolicyFamily,i)
    # No successor-vector or Tuple(actions) construction. This small-family scan
    # deliberately trades generic Set lookup for an allocation-light comparison.
    for p in f.policies
        same=true
        for j in eachindex(w.actions)
            if p[j]!=w.actions[j]-(j==i ? 1 : 0); same=false; break; end
        end
        same && return true
    end
    return false
end

# Feasibility of a second downshift from the proposed successor, without
# mutating the accepted policy. This is generic family logic, not AoII logic.
@inline _fp_successor_feasible(w,::UnrestrictedFamily,state,selected)=true
@inline function _fp_successor_feasible(w,::OrderedThresholdFamily,state,selected)
    p=w.predecessor[state]
    return p==0 || w.actions[p]-(p==selected)<w.actions[state]-(state==selected)
end
function _fp_successor_feasible(w,f::ExplicitPolicyFamily,state,selected)
    for policy in f.policies
        same=true
        for j in eachindex(w.actions)
            if policy[j]!=w.actions[j]-(j==selected)-(j==state)
                same=false;break
            end
        end
        same && return true
    end
    false
end

function _fp_frontier!(w)
    k=0
    for i in w.states
        if w.actions[i]>0 && _fp_feasible(w,w.family,i)
            k+=1; w.frontier_slots[k]=w.slot_of[i]
        end
    end
    w.nfrontier=k
    return k
end

# Ordinary selection is an isbits return. Rational fallback is kept in a
# separate function so the normal Float64 path need not allocate exact objects.
function _fp_select!(w::ReducedFPWorkspace{T},atol::T,rtol::T,guard::T) where {T}
    q=_fp_frontier!(w)
    q>0 || return (0,T(NaN),:empty_frontier)
    best=0; bestvalue=T(Inf); ambiguous=false
    for k in 1:q
        s=w.frontier_slots[k]; f=w.f[s]; g=w.g[s]; sc=w.g_scale[s]
        all(isfinite,(f,g,sc)) || return (0,T(NaN),:unresolved_comparison)
        band=_screen_band(sc,atol,rtol,guard)
        g < -band && continue
        if g<=band; ambiguous=true; continue; end
        r=f/g
        isfinite(r) || return (0,T(NaN),:unresolved_comparison)
        if r<bestvalue; best=s; bestvalue=r; end
    end
    ambiguous && return (0,T(NaN),:unresolved_comparison)
    best==0 && return (0,T(NaN),:no_positive_candidate)
    for k in 1:q
        s=w.frontier_slots[k]
        w.g[s]>_screen_band(w.g_scale[s],atol,rtol,guard) || continue
        if s!=best && abs(w.f[s]/w.g[s]-bestvalue)<=
                _screen_band(abs(w.f[s]/w.g[s])+abs(bestvalue),atol,rtol,guard)
            return (0,T(NaN),:unresolved_comparison)
        end
    end
    return (best,bestvalue,:numerical_screen)
end

# This exceptional path uses the SAME small-model safeguard as C/R/M-DS.
# Its extra float solve and rational solve are both counted, not hidden.
function _fp_exact_select(w::ReducedFPWorkspace{T},max_states) where {T}
    nstates(w.model)<=max_states || return (0,T(NaN),nothing,:unresolved_comparison)
    local q
    try
        q=ExactMultiGearModel(w.model)
    catch err
        err isa ArgumentError || rethrow()
        return (0,T(NaN),nothing,:unresolved_comparison)
    end
    w.counts.fallback_direct_solves+=1
    e=w.criterion==:discounted ? evaluate_discounted(w.model,w.actions;beta=w.beta) :
                               evaluate_average(w.model,w.actions;omega=w.omega)
    frontier=[(w.state_at[w.frontier_slots[k]],w.actions[w.state_at[w.frontier_slots[k]]])
              for k in 1:w.nfrontier]
    w.counts.exact_comparison_solves+=1
    ef,eg=_exact_frontier_comparison(q,e,frontier)
    k=_exact_positive_minimum(ef,eg)
    k==0 && return (0,T(NaN),nothing,:no_positive_candidate)
    exact_value=ef[k]/eg[k]; value=T(exact_value)
    isfinite(value) || return (0,T(NaN),nothing,:unresolved_comparison)
    return (w.frontier_slots[k],value,exact_value,:exact_comparison)
end

"""
    run_reduced_fp!(workspace; ..., audit_records=nothing)

Run the preallocated index loop, returning a STATUS SYMBOL. Ordinary output is
written to preallocated MPI/log arrays; no policy/frontier histories are created.
Use `reduced_fp_downshift` for an owned compact result. Initial setup, returned
result snapshots, exact fallbacks, rebuilds, and requested audits are separate
allocation categories. This is not the paper's timing harness.

Resolved negative-work candidates are skipped; unresolved signs/comparisons STOP
unless the small-model exact fallback resolves them. Exactly nonpositive work
is excluded. Floating near-ties are never merged. A resolved MPI decrease stops
before the offending assignment/pivot, preserving the accepted prefix.
The default average successor graph check is enabled and counted. Finite-value
and denominator screens are not rigorous error bounds; their comparison scale
is accumulated in reduced coordinates, not identical to C/R/M's full-state
residual screen. Do not benchmark them as equal-safeguard implementations yet.
"""
function run_reduced_fp!(w::ReducedFPWorkspace{T};max_steps=nothing,
        screen_atol::Real=1e-11,screen_rtol::Real=1e-9,guard_factor::Real=1000,
        monotonicity_atol::Real=1e-10,monotonicity_rtol::Real=1e-9,
        exact_fallback::Bool=true,exact_max_states::Integer=8,
        pivot_atol::Real=0,pivot_rtol::Real=1e-12,check_unichain::Bool=true,
        rebuild_every::Integer=0,rebuild_on_failure::Bool=true,backward_tol::Real=1e-10,
        audit_records::Union{Nothing,Vector{FPAuditRecord{T}}}=nothing,
        audit_atol::Real=1e-10,audit_rtol::Real=1e-8) where {T}
    aa,rr,gg=T(screen_atol),T(screen_rtol),T(guard_factor)
    ma,mr=T(monotonicity_atol),T(monotonicity_rtol)
    _check_monotonicity_tolerances(ma,mr)
    all(isfinite,(aa,rr,gg)) && aa>=0 && rr>=0 && gg>=1 || throw(ArgumentError("Invalid comparison settings."))
    exact_max_states>=1 && rebuild_every>=0 || throw(ArgumentError("Invalid retry settings."))
    max_steps===nothing || max_steps isa Integer || throw(ArgumentError("max_steps must be an integer."))
    limit=max_steps===nothing ? length(w.log_state) : Int(max_steps)
    0<=limit<=length(w.log_state) || throw(ArgumentError("max_steps outside 0:K."))
    if audit_records!==nothing
        check=audit_reduced_fp(w;atol=audit_atol,rtol=audit_rtol,backward_tol=backward_tol)
        push!(audit_records,check); check.passed || return :audit_failure
    end
    while w.m>0 && w.k<limit
        if rebuild_every>0 && w.k>0 && w.k%rebuild_every==0
            try
                rebuild_reduced_fp!(w;backward_tol=backward_tol)
            catch err
                (err isa ErrorException || err isa SingularException || err isa ArgumentError) || rethrow()
                return :rebuild_failure
            end
        end
        accepted=false
        for attempt in 1:(rebuild_on_failure ? 2 : 1)
            if attempt==2
                try
                    rebuild_reduced_fp!(w;backward_tol=backward_tol)
                catch err
                    (err isa ErrorException || err isa SingularException || err isa ArgumentError) || rethrow()
                    return :rebuild_failure
                end
            end
            slot,value,evidence=_fp_select!(w,aa,rr,gg)
            exact_value=nothing
            if evidence==:unresolved_comparison && exact_fallback
                try
                    slot,value,exact_value,evidence=_fp_exact_select(w,exact_max_states)
                catch err
                    (err isa ArgumentError || err isa SingularException || err isa ErrorException) || rethrow()
                    return :unresolved_comparison
                end
            end
            evidence in (:numerical_screen,:exact_comparison) || return evidence
            if w.k>0 && !_mpi_nondecreasing(w.log_m[w.k],value,ma,mr,
                    w.log_exact[w.k],exact_value)
                return :nonmonotone
            end
            i=w.state_at[slot]; a=w.actions[i]; oldn=w.m; f=w.f[slot]; g=w.g[slot]
            report=pivot_reduced_fp!(w,i;pivot_atol=pivot_atol,pivot_rtol=pivot_rtol,
                                    check_unichain=check_unichain)
            if !report.accepted
                report.reason==:unsupported_average_policy && return :unsupported_average_policy
                if attempt==2 || !rebuild_on_failure; return :unresolved_pivot; end
                continue
            end
            # Commit compact output only for an ACCEPTED pivot.
            k=w.k+1; w.k=k
            w.log_state[k]=i; w.log_gear[k]=a; w.log_f[k]=f; w.log_g[k]=g; w.log_m[k]=value
            w.log_eta[k]=report.denominator; w.log_dimension[k]=oldn
            w.log_evidence[k]=evidence; w.log_exact[k]=exact_value
            w.mpi[w.reference_slot[i],a]=value
            accepted=true; break
        end
        accepted || return :unresolved_pivot
        if audit_records!==nothing
            check=audit_reduced_fp(w;atol=audit_atol,rtol=audit_rtol,backward_tol=backward_tol)
            push!(audit_records,check); check.passed || return :audit_failure
        end
    end
    return w.m==0 ? (w.k==length(w.log_state) ? :complete : :suffix_complete) : :step_limit
end

struct ReducedFPResult{T<:AbstractFloat,W<:ReducedFPWorkspace{T}}
    workspace::W
    states::Vector{Int}
    mpi::Matrix{Union{Missing,T}}
    order::Vector{Tuple{Int,Int}}
    status::Symbol
    complete::Bool
    audits::Vector{FPAuditRecord{T}}
end

"""
    reduced_fp_downshift(model; criterion=:discounted, beta=nothing, omega=nothing,
        family=UnrestrictedFamily(), keep_recovery=false, audit=false, kwargs...)

Owned initialization + index loop + compact result. Completeness means that all
A*Nc MP assignments were produced, NOT that they are verified DAI values. Use
`validate_downshift(result, independent_reference)` for separate verification.
The workspace is retained for inspection; mpi/order fields are stable snapshots.
"""
function reduced_fp_downshift(model::FiniteMultiGearModel{T};criterion::Symbol=:discounted,
        beta=nothing,omega=nothing,family::AbstractPolicyFamily=UnrestrictedFamily(),
        keep_recovery::Bool=false,audit::Bool=false,backward_tol::Real=1e-10,kwargs...) where {T}
    w=initialize_reduced_fp(model;criterion=criterion,beta=beta,omega=omega,family=family,
                           keep_recovery=keep_recovery,backward_tol=backward_tol)
    records=FPAuditRecord{T}[]
    status=run_reduced_fp!(w;backward_tol=backward_tol,audit_records=(audit ? records : nothing),kwargs...)
    order=[(w.log_state[k],w.log_gear[k]) for k in 1:w.k]
    return ReducedFPResult(w,copy(w.states),copy(w.mpi),order,status,status==:complete,records)
end


@inline function _fp_feasible(w,f::RowThresholdFamily,i)
    p=f.predecessor[i]
    p==0 || w.actions[p]<w.actions[i]
end
@inline function _fp_successor_feasible(w,f::RowThresholdFamily,state,selected)
    p=f.predecessor[state]
    p==0 || w.actions[p]-(p==selected)<w.actions[state]-(state==selected)
end
