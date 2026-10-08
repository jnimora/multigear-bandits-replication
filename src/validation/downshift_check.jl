struct DownshiftValidation
    execution::Symbol
    stopping::Symbol
    numeric_assignments::Symbol
    monotonicity::Symbol
    frontier_positivity::Symbol
    pathwise_positivity::Symbol
    family_wide_positivity::Symbol
    bellman_path::Symbol
    reference_sign_indexability::Symbol
    mpi_dai_agreement::Symbol
    max_scaled_index_error::Union{Nothing,Float64}
    messages::Vector{String}
end

function _same_exact_model(a,b)
    return a.P == b.P && a.h == b.h && a.c == b.c && a.controllable == b.controllable
end
function _interval_subset(inner::ChargeInterval,outer::ChargeInterval)
    inner.empty && return true
    outer.empty && return false
    lower_ok = outer.lower === nothing || (inner.lower !== nothing && outer.lower <= inner.lower)
    upper_ok = outer.upper === nothing || (inner.upper !== nothing && inner.upper <= outer.upper)
    return lower_ok && upper_ok
end

"""
    validate_downshift(run, reference; atol=1e-11, rtol=1e-9)

Check a C-DS, R-DS or M-DS output against an independently enumerated exact model. Rejects
mismatched primitives, criterion, beta or normalization. It checks exact
positive-work argmin selection (including lexicographic ties), the recorded family frontier,
and MPI assignments. Positivity of the frontier, EVERY adjacent comparison on
EVERY path policy (including the terminal policy), and the whole family are
separate fields. Proposed policy intervals are checked against ALL-ACTION
Bellman intervals, not only family-preserving deviations.

MPI-to-DAI error is reported only when the independent oracle has established
sign-indexability. A finite-precision output remains a numerical approximation
to the exact reference DAI; it is not relabelled an exact numerical value.
Partial executions remain partial; failures are not collapsed into a generic
nonindexable status. No model is claimed PCL-indexable just because it finished.
"""
function validate_downshift(run::Union{DirectDownshiftResult,RankOneDownshiftResult,CachedDownshiftResult},ref::BellmanEnumeration;
                             atol::Real=1e-11,rtol::Real=1e-9)
    isfinite(atol) && isfinite(rtol) && atol >= 0 && rtol >= 0 ||
        throw(ArgumentError("Invalid numerical comparison tolerances."))
    _same_exact_model(ExactMultiGearModel(run.model),ref.model) ||
        throw(ArgumentError("The oracle and downshift run describe DIFFERENT stored primitives."))
    run.criterion == ref.criterion || throw(ArgumentError("Criterion mismatch."))
    if run.criterion == :discounted
        _exact_number(run.beta) == ref.beta || throw(ArgumentError("Discount-factor mismatch."))
    else
        _exact_normalization(nstates(run.model),run.omega) == ref.omega ||
            throw(ArgumentError("Bias-normalization mismatch."))
    end
    messages = String[]
    if !ref.complete || ref.coverage != :exact_pass
        push!(messages,"The independent reference is incomplete or lacks whole-line Bellman coverage.")
        return DownshiftValidation(:unresolved,:unresolved,:unresolved,:unresolved,:unresolved,:unresolved,
            :unresolved,:unresolved,ref.sign_indexability,:unresolved,nothing,messages)
    end
    lookup = Dict(Tuple(p.actions) => p for p in ref.records)
    exact_values = ExactRational[]
    exact_map = Matrix{Union{Missing,ExactRational}}(undef,length(ref.states),maxgear(ref.model))
    fill!(exact_map,missing)
    execution_ok = length(run.policies) == length(run.steps)+1 &&
        run.states == ref.states && size(run.mpi) == size(exact_map)
    frontier_ok = true
    numeric_ok = true
    for (k,step) in enumerate(run.steps)
        execution_ok || break
        step.number == k || (execution_ok = false; break)
        policy = run.policies[k]
        record = get(lookup,Tuple(policy),nothing)
        record === nothing && (execution_ok = false; break)
        frontier = downshift_frontier(ref.model,policy,run.family)
        if frontier != step.frontier || !(step.selected in frontier)
            execution_ok = false
            break
        end
        ratios = ExactRational[]; positive_frontier=Tuple{Int,Int}[]
        for (i,a) in frontier
            row = findfirst(==(i),ref.states)
            g = -record.gap_slope[row,a]
            if g <= 0
                frontier_ok = false
                continue
            end
            push!(ratios,record.gap_intercept[row,a]/g)
            push!(positive_frontier,(i,a))
        end
        isempty(ratios) && (execution_ok=false;break)
        selected = positive_frontier[argmin(ratios)]
        selected == step.selected || (execution_ok = false; break)
        value = minimum(ratios)
        successor = copy(policy); successor[selected[1]] -= 1
        successor == run.policies[k+1] || (execution_ok = false; break)
        row = findfirst(==(selected[1]),ref.states)
        ismissing(exact_map[row,selected[2]]) || (execution_ok = false; break)
        exact_map[row,selected[2]] = value
        push!(exact_values,value)
        # Compare against the mathematical rational value, not just a cast.
        for x in (step.assignment,run.mpi[row,selected[2]])
            if ismissing(x) || !isfinite(x)
                numeric_ok = false
            else
                tolerance = _exact_number(atol)+_exact_number(rtol)*max(abs(value),abs(_exact_number(x)))
                numeric_ok &= abs(_exact_number(x)-value) <= tolerance
            end
        end
    end
    high = [ref.model.controllable[i] ? maxgear(ref.model) : 0 for i in 1:nstates(ref.model)]
    execution_ok &= !isempty(run.policies) && run.policies[1] == high
    if run.complete
        execution_ok &= length(run.steps) == length(ref.states)*maxgear(ref.model)
        execution_ok &= run.policies[end] == zeros(Int,nstates(ref.model))
        execution_ok &= all(x -> !ismissing(x),exact_map)
    end
    if !execution_ok
        push!(messages,"The stored path fails an exact adaptive-greedy selection or transition check.")
    end
    !frontier_ok && push!(messages,"Some frontier work marginals are nonpositive; positive-work selection is checked separately.")
    execution = execution_ok ? (run.complete ? :exact_pass : :partial_path_checked) : :exact_violation
    stopping = :unresolved
    frontier_evidence = frontier_ok ? (run.complete ? :exact_pass : :partial_path_checked) : :exact_violation
    if execution_ok && !isempty(run.policies)
        last_policy = run.policies[end]
        last_record = lookup[Tuple(last_policy)]
        last_frontier = downshift_frontier(ref.model,last_policy,run.family)
        any(last_record.gap_slope[findfirst(==(i),ref.states),a]>=0 for (i,a) in last_frontier) &&
            (frontier_evidence=:exact_violation)
        if run.complete
            stopping = isempty(last_frontier) && all(iszero,last_policy) ? :exact_pass : :exact_violation
        elseif run.status == :empty_frontier
            stopping = isempty(last_frontier) && !all(iszero,last_policy) ? :exact_pass : :exact_violation
        elseif run.status == :nonpositive_denominator
            # Validate historical strict-frontier runs without changing their meaning.
            bad = any(last_record.gap_slope[findfirst(==(i),ref.states),a] >= 0 for (i,a) in last_frontier)
            stopping = bad ? :exact_pass : :exact_violation
            frontier_evidence = bad ? :exact_violation : :partial_path_checked
        elseif run.status == :no_positive_candidate
            no_positive=!isempty(last_frontier) && all(
                last_record.gap_slope[findfirst(==(i),ref.states),a]>=0 for (i,a) in last_frontier)
            stopping=no_positive ? :exact_pass : :exact_violation
        elseif run.status == :nonmonotone
            next_values=ExactRational[]
            for (i,a) in last_frontier
                row=findfirst(==(i),ref.states);g=-last_record.gap_slope[row,a]
                g>0 && push!(next_values,last_record.gap_intercept[row,a]/g)
            end
            violation=!isempty(exact_values) && !isempty(next_values) && minimum(next_values)<exact_values[end]
            stopping=violation ? :exact_pass : :exact_violation
        elseif run.status == :step_limit
            stopping = :requested_limit
        end
    end
    if execution == :exact_violation
        numeric = numeric_ok ? :unresolved : :numerical_violation
    else
        numeric = numeric_ok ? :numerical_pass : :numerical_violation
    end
    mono = length(exact_values) == length(run.steps) ?
        (issorted(exact_values) ? :exact_pass : :exact_violation) : :unresolved
    if !run.complete && mono == :exact_pass; mono = :partial_path_checked; end
    pathwise = :exact_pass
    for p in run.policies
        record = get(lookup,Tuple(p),nothing)
        if record === nothing
            pathwise = :unresolved; break
        elseif any(record.gap_slope .>= 0)
            pathwise = :exact_violation; break
        end
    end
    !run.complete && pathwise == :exact_pass && (pathwise = :partial_path_checked)
    family_positive = all(all(r.gap_slope .< 0) for r in ref.records
                          if policy_in_family(run.family,r.actions)) ? :exact_pass : :exact_violation
    bellman = :unresolved
    if execution == :exact_pass && mono == :exact_pass
        bellman_ok = true
        for k in eachindex(run.policies)
            lower = k == 1 ? nothing : exact_values[k-1]
            upper = k > length(exact_values) ? nothing : exact_values[k]
            record = lookup[Tuple(run.policies[k])]
            bellman_ok &= _interval_subset(ChargeInterval(lower,upper,false),record.interval)
        end
        bellman = bellman_ok ? :exact_pass : :exact_violation
    end
    agreement = :unresolved
    error_value = nothing
    if ref.sign_indexability == :exact_violation
        agreement = :not_applicable
        push!(messages,"Sign-indexability is disproved for this exact model; no reference DAI or index error exists.")
    elseif execution == :exact_pass && ref.dai !== nothing
        same = all(!ismissing(exact_map[r,a]) && exact_map[r,a] == ref.dai[r,a]
                   for r in axes(exact_map,1), a in axes(exact_map,2))
        agreement = same && numeric_ok ? :numerical_pass : :violation
        if all(x -> !ismissing(x) && isfinite(x),run.mpi)
            err = maximum(abs(_exact_number(run.mpi[r,a])-ref.dai[r,a])/(1+abs(ref.dai[r,a]))
                          for r in axes(ref.dai,1), a in axes(ref.dai,2))
            error_value = Float64(err)
        end
        same || push!(messages,"The exact MP assignments on the recorded path differ from the oracle DAI.")
    end
    !run.complete && push!(messages,"Only the completed prefix was checked; no complete MPI is asserted.")
    return DownshiftValidation(execution,stopping,numeric,mono,
        frontier_evidence,
        pathwise,family_positive,bellman,ref.sign_indexability,agreement,error_value,messages)
end
