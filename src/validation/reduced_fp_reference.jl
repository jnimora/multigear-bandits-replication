"""
Expand a compact FP log ONLY for existing independent-reference postprocessing.
Frontier vectors below are recomputed by direct evaluation and explicitly are
NOT recorded FP-cache snapshots. Runtime-cache agreement is checked separately
by audit_reduced_fp. This adapter does not change the selected pairs or MPI map.
It performs one fresh evaluation per recorded step OUTSIDE the index loop.
"""
function _fp_reference_adapter(run::ReducedFPResult{T}) where {T}
    w=run.workspace; actions=[w.model.controllable[i] ? maxgear(w.model) : 0 for i in 1:nstates(w.model)]
    policies=[copy(actions)]; steps=DownshiftStep{T}[]
    for k in 1:w.k
        frontier=downshift_frontier(w.model,actions,w.family)
        e=w.criterion==:discounted ? evaluate_discounted(w.model,actions;beta=w.beta) :
                                   evaluate_average(w.model,actions;omega=w.omega)
        f,g,_=_frontier_metrics(e,frontier)
        r=[g[j]>0 ? f[j]/g[j] : T(NaN) for j in eachindex(g)]
        push!(steps,DownshiftStep(k,frontier,f,g,r,(w.log_state[k],w.log_gear[k]),
            w.log_m[k],w.log_exact[k],w.log_evidence[k],e.diagnostics))
        actions[w.log_state[k]]-=1; push!(policies,copy(actions))
    end
    mono=_monotonicity_screen(steps,T(1e-11),T(1e-9),T(1000))
    return DirectDownshiftResult(w.model,w.criterion,w.beta,w.omega,w.family,copy(run.states),
        copy(run.mpi),steps,policies,run.complete,run.status,
        "Compact reduced-FP output expanded for independent reference validation only.",mono,
        length(steps),w.counts.exact_comparison_solves)
end

function validate_downshift(run::ReducedFPResult,ref::BellmanEnumeration;
                            atol::Real=1e-11,rtol::Real=1e-9)
    return validate_downshift(_fp_reference_adapter(run),ref;atol=atol,rtol=rtol)
end
