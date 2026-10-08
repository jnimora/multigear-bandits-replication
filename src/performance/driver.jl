"""Common, preallocated INDEX-ONLY output. One entry per accepted assignment."""
mutable struct PerformanceLog
    states::Vector{Int}
    row_of::Vector{Int}
    mpi::Matrix{Union{Missing,Float64}}
    state::Vector{Int}
    gear::Vector{Int}
    f::Vector{Float64}
    g::Vector{Float64}
    ratio::Vector{Float64}
    dimension::Vector{Int}
    evidence::Vector{Symbol}
    k::Int
end
function PerformanceLog(model)
    states=findall(model.controllable); n=nstates(model); A=maxgear(model); K=length(states)*A
    row=zeros(Int,n)
    for (j,i) in enumerate(states); row[i]=j; end
    mpi=Matrix{Union{Missing,Float64}}(undef,length(states),A); fill!(mpi,missing)
    return PerformanceLog(states,row,mpi,zeros(Int,K),zeros(Int,K),zeros(K),zeros(K),zeros(K),
                          zeros(Int,K),fill(:unassigned,K),0)
end

"""
Index-only performance run. By default, at :complete the last MPI is recorded WITHOUT a
numerical update to the terminal all-zero policy. Thus core.actions/core.values
still represent the LAST SELECTION policy, not a recovered terminal evaluation.
Use performance_snapshot, not the internal workspace, for the public output.
This convention is identical for C/R/M/FP and avoids a gratuitous terminal LU.
Set options.terminal_update=true for the all-K-update endpoint control.
"""
mutable struct PerformanceRun{W}
    core::W
    algorithm::Symbol
    options::PerformanceOptions
    log::PerformanceLog
    status::Symbol
    exact_comparison_solves::Int
    fallback_direct_solves::Int
    terminal_update_skipped::Bool
    residual_comparison_checks::Int
    residual_certified_selections::Int
    residual_inverse_rhs_columns::Int
    residual_comparisons::Vector{Dict{String,Any}}
end

"""
    prepare_performance_run(input, algorithm; criterion=:discounted, beta=nothing,
        omega=nothing, family=UnrestrictedFamily(), options=PerformanceOptions())

Prepare :C, :R, :M, or :FP with the SAME minimal requested output and comparison
policy. All-high evaluation, initial frontier metrics, buffers, and (for FP)
reference data are charged to initialization. Model generation/shared admissible
input checking are outside algorithm initialization and separately reported.
The ordinary loop selects only resolved positive-work candidates, checks MPI
monotonicity before accepting an assignment/update, and uses finite-value,
comparison-separation and update-denominator checks. No automatic rebuild or periodic full
residual solve is hidden in any method. Independent pivotwise audits are separate.
Comparison tolerances are screens, not rigorous error bounds or certificates.
Initialization screens ALL solved columns for all methods. :optimized (default)
and :legacy select FP preparation only; both use the current checked pivot kernel. Optional
preparation_trace is for separate diagnostic runs, not headline timings.
"""
function prepare_performance_run(input::PerformanceInput,algorithm::Symbol;
        criterion::Symbol=:discounted,beta=nothing,omega=nothing,
        family::AbstractPolicyFamily=UnrestrictedFamily(),options=PerformanceOptions(),
        fp_initializer::Symbol=:optimized,preparation_trace::Union{Nothing,PreparationTrace}=nothing,
        reference_block_columns::Integer=64)
    algorithm in (:C,:R,:M,:FP) || throw(ArgumentError("Algorithm must be :C, :R, :M, or :FP."))
    _performance_check_options(options)
    fp_initializer in (:optimized,:legacy) || throw(ArgumentError("Unknown FP initializer."))
    1<=reference_block_columns<=typemax(Int) || throw(ArgumentError("Invalid reference block width."))
    VERSION>=v"1.13" || error("This performance pilot requires Julia 1.13 or later.")
    if criterion==:average && !_performance_average_admitted(input)
        throw(ArgumentError("Average performance requires dense-positive rows or explicitly checked support topology."))
    end
    core=if algorithm==:FP
        w=if fp_initializer==:legacy
            # Retained old source is a diagnostic/control path, not a new pivot algorithm.
            _preparation_stage(preparation_trace,:legacy_FP_initialization) do
                initialize_reduced_fp(input.model;criterion=criterion,beta=beta,omega=omega,
                    family=family,keep_recovery=false)
            end
        else
            initialize_reduced_fp_optimized(input;criterion=criterion,beta=beta,omega=omega,
                family=family,keep_recovery=false,trace=preparation_trace,
                block_columns=reference_block_columns)
        end
        w.frontier_only=options.fp_frontier_cache && !(family isa UnrestrictedFamily)
        _fp_frontier!(w)
        w
    else
        _initialize_full_performance(input,Val(algorithm);criterion=criterion,beta=beta,omega=omega,
            family=family,preparation_trace=preparation_trace)
    end
    log=core isa ReducedFPWorkspace ? PerformanceLog(core.states,core.reference_slot,core.mpi,
        core.log_state,core.log_gear,core.log_f,core.log_g,core.log_m,core.log_dimension,core.log_evidence,0) :
        PerformanceLog(input.model)
    return PerformanceRun(core,algorithm,options,log,:ready,0,0,false,0,0,0,Dict{String,Any}[])
end

@inline _performance_nfrontier(w)=w.nfrontier
@inline _performance_state(w::FullPerformanceWorkspace,j)=w.frontier[j]
@inline _performance_state(w::ReducedFPWorkspace,j)=w.state_at[w.frontier_slots[j]]
@inline _performance_pair_metrics(w::FullPerformanceWorkspace,j)=(w.f[w.frontier[j]],w.g[w.frontier[j]])
@inline _performance_pair_metrics(w::ReducedFPWorkspace,j)=(w.f[w.frontier_slots[j]],w.g[w.frontier_slots[j]])
@inline _performance_dimension(w::FullPerformanceWorkspace)=size(w.values,1)
@inline _performance_dimension(w::ReducedFPWorkspace)=w.m

# All four methods use this selector, in original-label lexicographic order.
# It never merges close floating values or treats a numerical tie as exact.
function _performance_select(w,o::PerformanceOptions)
    q=_performance_nfrontier(w)
    q==0 && return (0,NaN,:empty_frontier)
    best=0; bestvalue=Inf; ambiguous=false
    for j in 1:q
        f,g=_performance_pair_metrics(w,j)
        all(isfinite,(f,g)) || return (0,NaN,:nonfinite_metric)
        band=o.comparison_atol+o.comparison_rtol*max(1.0,abs(g))
        g < -band && continue
        if g<=band; ambiguous=true; continue; end
        value=f/g
        isfinite(value) || return (0,NaN,:nonfinite_metric)
        if value<bestvalue; best=j; bestvalue=value; end
    end
    ambiguous && return (0,NaN,:unresolved_comparison)
    best==0 && return (0,NaN,:no_positive_candidate)
    for j in 1:q
        j==best && continue
        f,g=_performance_pair_metrics(w,j)
        g>o.comparison_atol+o.comparison_rtol*max(1.0,abs(g)) || continue
        value=f/g
        band=o.comparison_atol+o.comparison_rtol*max(1.0,abs(value),abs(bestvalue))
        abs(value-bestvalue)<=band && return (0,NaN,:unresolved_comparison)
    end
    return (best,bestvalue,:numerical_screen)
end

# Exceptional exact safeguard is shared verbatim by all four performance paths.
# Its float and rational solves remain inside loop time and allocation totals.
function _performance_exact_select!(run)
    w=run.core; o=run.options
    nstates(w.model)<=o.exact_max_states || return (0,NaN,:unresolved_comparison)
    local qmodel,e,ef,eg
    try
        qmodel=ExactMultiGearModel(w.model)
        run.fallback_direct_solves+=1
        e=w.criterion==:discounted ? evaluate_discounted(w.model,w.actions;beta=w.beta) :
                                    evaluate_average(w.model,w.actions;omega=w.omega)
        frontier=[(_performance_state(w,j),w.actions[_performance_state(w,j)]) for j in 1:w.nfrontier]
        run.exact_comparison_solves+=1
        ef,eg=_exact_frontier_comparison(qmodel,e,frontier)
    catch err
        (err isa ArgumentError || err isa SingularException || err isa ErrorException) || rethrow()
        return (0,NaN,:unresolved_comparison)
    end
    best=_exact_positive_minimum(ef,eg)
    best==0 && return (0,NaN,:no_positive_candidate)
    value=Float64(ef[best]/eg[best])
    isfinite(value) || return (0,NaN,:unresolved_comparison)
    return (best,value,:exact_comparison)
end

# Exact residual-bound certification is a separate opt-in selection safeguard.
# Every model, policy, and comparator uses the same implementation. No state is
# updated here. Its LU, rational work, and event log stay INSIDE measured loops.
function _performance_residual_select!(run)
    w=run.core
    nstates(w.model)<=run.options.residual_max_states || return (0,NaN,:unresolved_comparison)
    run.residual_comparison_checks+=1
    run.fallback_direct_solves+=1
    frontier=[(_performance_state(w,j),w.actions[_performance_state(w,j)]) for j in 1:w.nfrontier]
    try
        answer=residual_certified_selection(w.model,w.actions,frontier;
            criterion=w.criterion,beta=w.beta,omega=w.omega,bound_method=run.options.residual_bound)
        report=answer.report
        report["selection_step"]=run.log.k+1
        run.residual_inverse_rhs_columns+=report["inverse_rhs_columns"]
        answer.status==:residual_certified && (run.residual_certified_selections+=1)
        push!(run.residual_comparisons,report)
        return (answer.position,answer.value,answer.status)
    catch err
        err isa InterruptException && rethrow()
        (err isa ArgumentError || err isa SingularException || err isa ErrorException) || rethrow()
        push!(run.residual_comparisons,Dict{String,Any}("status"=>"unresolved_comparison",
            "selection_step"=>run.log.k+1,"error"=>sprint(showerror,err)))
        return (0,NaN,:unresolved_comparison)
    end
end

function _performance_update!(w::ReducedFPWorkspace,i,o)
    report=pivot_reduced_fp!(w,i;pivot_atol=o.pivot_atol,pivot_rtol=o.pivot_rtol,
                            check_unichain=false)
    # Every policy was structurally covered by PerformanceInput in average mode.
    report.accepted || return (report.reason,report.denominator)
    _fp_frontier!(w)
    return (:accepted,report.denominator)
end

"""Perform one selection/accepted update. Use only from one owning task."""
step_performance!(run::PerformanceRun) = _step_performance_with_update!(run,_performance_update!)

# Keep admission, safeguards and log commit shared with experimental update kernels.
function _step_performance_with_update!(run::PerformanceRun, update!::F) where {F}
    run.status in (:ready,:running,:step_limit) || return run.status
    w=run.core; log=run.log
    log.k<length(log.state) || (run.status=:complete; return run.status)
    j,value,evidence=_performance_select(w,run.options)
    if evidence==:unresolved_comparison && run.options.exact_fallback
        j,value,evidence=_performance_exact_select!(run)
    end
    if evidence==:unresolved_comparison && run.options.residual_fallback
        j,value,evidence=_performance_residual_select!(run)
    end
    if !(evidence in (:numerical_screen,:exact_comparison,:residual_certified))
        run.status=evidence; return run.status
    end
    i=_performance_state(w,j); a=w.actions[i]; f,g=_performance_pair_metrics(w,j)
    dimension=_performance_dimension(w); k=log.k+1
    if log.k>0 && !_mpi_nondecreasing(log.ratio[log.k],value,
            run.options.monotonicity_atol,run.options.monotonicity_rtol)
        run.status=:nonmonotone
        return run.status
    end
    if k<length(log.state) || run.options.terminal_update
        reason,_=update!(w,i,run.options)
        if reason!=:accepted
            run.status=reason; return run.status
        end
    else
        # Index-only endpoint: nothing downstream consumes terminal policy values.
        run.terminal_update_skipped=true
    end
    log.state[k]=i; log.gear[k]=a; log.f[k]=f; log.g[k]=g
    log.ratio[k]=value; log.dimension[k]=dimension; log.evidence[k]=evidence
    log.mpi[log.row_of[i],a]=value; log.k=k
    w isa ReducedFPWorkspace && !run.terminal_update_skipped && (w.k=k)
    run.status=k==length(log.state) ? :complete : :running
    return run.status
end

function run_performance!(run::PerformanceRun;max_steps::Union{Nothing,Integer}=nothing)
    limit=max_steps===nothing ? length(run.log.state) : Int(max_steps)
    0<=limit<=length(run.log.state) || throw(ArgumentError("max_steps outside 0:K."))
    while run.log.k<limit && run.status in (:ready,:running,:step_limit)
        step_performance!(run)
    end
    if run.log.k<length(run.log.state) && run.status in (:ready,:running,:step_limit)
        run.status=:step_limit
    end
    return run.status
end

function performance_counts(run::PerformanceRun)
    w=run.core
    counts=Dict{String,Int}()
    for name in fieldnames(typeof(w.counts)); counts[string(name)]=getfield(w.counts,name); end
    counts["exact_comparison_solves"]=run.exact_comparison_solves
    counts["fallback_direct_solves"]=run.fallback_direct_solves
    counts["residual_comparison_checks"]=run.residual_comparison_checks
    counts["residual_certified_selections"]=run.residual_certified_selections
    counts["residual_inverse_rhs_columns"]=run.residual_inverse_rhs_columns
    counts["assignments"]=run.log.k
    counts["terminal_numerical_updates_skipped"]=Int(run.terminal_update_skipped)
    return counts
end

"""Owned compact result, allocated separately from the ordinary index loop."""
function performance_snapshot(run::PerformanceRun)
    log=run.log; k=log.k
    return (algorithm=run.algorithm,status=run.status,complete=run.status==:complete,
        selection_contract=_SELECTION_CONTRACT,
        states=copy(log.states),mpi=copy(log.mpi),
        order=[(log.state[j],log.gear[j]) for j in 1:k],
        f=log.f[1:k],g=log.g[1:k],ratio=log.ratio[1:k],dimension=log.dimension[1:k],
        evidence=log.evidence[1:k],counts=performance_counts(run),
        terminal_update_skipped=run.terminal_update_skipped,
        residual_comparisons=deepcopy(run.residual_comparisons))
end

function performance_agreement(a,b;atol=1e-9,rtol=1e-7)
    a.status==b.status && a.order==b.order && a.states==b.states && size(a.mpi)==size(b.mpi) || return false
    for j in eachindex(a.mpi,b.mpi)
        x,y=a.mpi[j],b.mpi[j]
        if ismissing(x) || ismissing(y)
            isequal(x,y) || return false
        else
            isapprox(x,y;atol=atol,rtol=rtol) || return false
        end
    end
    return true
end
