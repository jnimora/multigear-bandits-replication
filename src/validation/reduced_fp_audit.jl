"""One fresh-solve audit. Numerical discrepancies are diagnostics, not certificates."""
struct FPAuditRecord{T<:AbstractFloat}
    policy_index::Int
    dimension::Int
    passed::Bool
    maps_passed::Bool
    tableau_error::T
    aggregate_f_error::T
    aggregate_g_error::T
    cached_f_error::T
    cached_g_error::T
    recovery_error::Union{Nothing,T}
end

"""Explicitly allocating, stable diagnostic snapshot; never called in the index loop."""
function reduced_fp_snapshot(w::ReducedFPWorkspace)
    return (actions=copy(w.actions),labels=copy(w.state_at[1:w.m]),slot_of=copy(w.slot_of),
        dimension=w.m,tableau=copy(w.X[1:w.m,1:w.m]),phi=copy(w.phi[1:w.m]),
        psi=copy(w.psi[1:w.m]),f=copy(w.f[1:w.m]),g=copy(w.g[1:w.m]))
end

"""
Return newly allocated full value columns from retained reference data, with no
policy solve. Requires `keep_recovery=true` at initialization. Columns are F,G
under discounting; under averaging rows 1:N are normalized biases and row N+1
contains the two rates. Recovery is an EXTRA output, not an index-loop operation.
"""
function recover_reduced_values(w::ReducedFPWorkspace)
    w.recovery_W!==nothing && w.recovery_values!==nothing || throw(ArgumentError(
        "Initialize with keep_recovery=true to request full value/bias-rate recovery."))
    values=copy(w.recovery_values)
    for s in 1:w.m
        c=w.reference_slot[w.state_at[s]]
        for i in axes(values,1)
            values[i,1]-=w.recovery_W[i,c]*w.phi[s]
            values[i,2]+=w.recovery_W[i,c]*w.psi[s]
        end
    end
    return values
end

@inline function _fp_compare(x,y,atol,rtol)
    if !isfinite(x) || !isfinite(y); return false,oftype(x,Inf); end
    err=abs(x-y); mag=max(abs(x),abs(y))
    return err<=atol+rtol*mag,err/(one(x)+mag)
end

"""
    audit_reduced_fp(workspace; atol=1e-10, rtol=1e-8, backward_tol=1e-10)

Use ONE fresh full-system factorization to compare the active tableau, aggregate
metrics, all maintained current adjacent metrics, and optional
full bias/rate recovery. Check original-label maps as well. All N Markov states
remain in the independent system even when the reduced dimension has shrunk.
In frontier_only mode, only feasible cached comparisons are maintained; the
tableau and aggregate metrics are still checked at every active coordinate.
The audit allocates deliberately and increments a separate audit counter.
"""
function audit_reduced_fp(w::ReducedFPWorkspace{T};atol::Real=1e-10,rtol::Real=1e-8,
                          backward_tol::Real=1e-10) where {T}
    aa,rr=T(atol),T(rtol)
    isfinite(aa) && isfinite(rr) && aa>=0 && rr>=0 || throw(ArgumentError("Invalid audit tolerances."))
    labels=w.state_at[1:w.m]
    maps=sort(labels)==[i for i in w.states if w.actions[i]>0]
    for s in 1:w.m; maps &= w.slot_of[labels[s]]==s; end
    for i in eachindex(w.actions); maps &= (w.actions[i]>0)==(w.slot_of[i]>0); end
    w.counts.audit_solves+=1
    X,ph,ps,_,values,_=_fp_fresh(w.model,w.actions,w.criterion,w.beta,w.omega,labels;
                                backward_tol=backward_tol)
    passed=maps; xe=zero(T); pe=zero(T); qe=zero(T); fe=zero(T); ge=zero(T)
    for c in 1:w.m, s in 1:w.m
        ok,e=_fp_compare(w.X[s,c],X[s,c],aa,rr); passed &= ok; xe=max(xe,e)
    end
    factor=w.criterion==:discounted ? w.beta : one(T)
    for s in 1:w.m
        ok,e=_fp_compare(w.phi[s],ph[s],aa,rr); passed &= ok; pe=max(pe,e)
        ok,e=_fp_compare(w.psi[s],ps[s],aa,rr); passed &= ok; qe=max(qe,e)
        i=labels[s]; a=w.actions[i]
        w.frontier_only && !_fp_feasible(w,w.family,i) && continue
        ff=w.model.h[i,a]-w.model.h[i,a+1]; gg=w.model.c[i,a+1]-w.model.c[i,a]
        for j in 1:nstates(w.model)
            dp=factor*(w.model.P[i,j,a+1]-w.model.P[i,j,a])
            ff-=dp*values[j,1]; gg+=dp*values[j,2]
        end
        ok,e=_fp_compare(w.f[s],ff,aa,rr); passed &= ok; fe=max(fe,e)
        ok,e=_fp_compare(w.g[s],gg,aa,rr); passed &= ok; ge=max(ge,e)
    end
    re=nothing
    if w.recovery_W!==nothing
        recovered=recover_reduced_values(w); re=zero(T)
        for j in axes(values,2), i in axes(values,1)
            ok,e=_fp_compare(recovered[i,j],values[i,j],aa,rr); passed &= ok; re=max(re,e)
        end
    end
    return FPAuditRecord(w.k,w.m,passed,maps,xe,pe,qe,fe,ge,re)
end
