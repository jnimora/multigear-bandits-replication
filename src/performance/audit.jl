# These fresh-solve checks are intentionally OUTSIDE every timed measurement.
# Never use a monotone output sequence alone as an indexability certificate.
function _performance_scaled_error(A,B)
    size(A)==size(B) || return Inf
    e=0.0
    for j in eachindex(A,B)
        all(isfinite,(A[j],B[j])) || return Inf
        e=max(e,abs(A[j]-B[j])/max(1.0,abs(A[j]),abs(B[j])))
    end
    return e
end

"""Independent current-policy audit, including the inverse or reduced tableau."""
function audit_performance_state(run::PerformanceRun;tol=1e-7)
    tol>0 && isfinite(tol) || throw(ArgumentError("Invalid audit tolerance."))
    w=run.core
    e=w.criterion==:discounted ? evaluate_discounted(w.model,w.actions;beta=w.beta) :
                                evaluate_average(w.model,w.actions;omega=w.omega)
    allm=adjacent_marginals(e); row=zeros(Int,nstates(w.model))
    for (r,i) in enumerate(allm.states); row[i]=r; end
    fgerror=0.0
    for j in 1:w.nfrontier
        i=_performance_state(w,j); a=w.actions[i]; f,g=_performance_pair_metrics(w,j)
        fgerror=max(fgerror,abs(f-allm.f[row[i],a])/max(1.0,abs(f),abs(allm.f[row[i],a])),
                            abs(g-allm.g[row[i],a])/max(1.0,abs(g),abs(allm.g[row[i],a])))
    end
    valueerror=0.0; inverseerror=0.0; reduced_pass=true
    if w isa FullPerformanceWorkspace
        V=e isa DiscountedEvaluation ? hcat(e.F,e.G) :
              vcat(hcat(e.phi,e.gamma),reshape([e.hbar,e.cbar],1,2))
        valueerror=_performance_scaled_error(w.values,V)
        if _performance_method(w)!=:C
            _,M,_,_= _rankone_system(w.model,w.actions,w.criterion,w.beta,w.omega)
            d=size(M,1); Id=Matrix{Float64}(I,d,d)
            Z=lu(M)\Id
            inverseerror=_performance_scaled_error(w.inverse,Z)
        end
    else
        reduced_pass=audit_reduced_fp(w;atol=tol/10,rtol=tol).passed
    end
    passed=all(isfinite,(fgerror,valueerror,inverseerror)) &&
        max(fgerror,valueerror,inverseerror)<=tol && reduced_pass
    return (passed=passed,frontier_error=fgerror,value_error=valueerror,inverse_error=inverseerror,
            reduced_pass=reduced_pass)
end

"""
Reconstruct an independent workspace and audit EVERY selection policy before
its step. Includes incomplete runs: an unresolved/negative case is retained as
such, not converted to a success. No terminal value is requested or compared.
This verifies numerical execution, not Bellman optimality or indexability.
"""
function audit_performance_run(input,algorithm;criterion=:discounted,beta=nothing,omega=nothing,
        family=UnrestrictedFamily(),options=PerformanceOptions(),tol=1e-7,
        fp_initializer::Symbol=:optimized,reference_block_columns::Integer=64)
    run=prepare_performance_run(input,algorithm;criterion=criterion,beta=beta,omega=omega,
                                family=family,options=options,fp_initializer=fp_initializer,
                                reference_block_columns=reference_block_columns)
    checks=0; maxerror=0.0; passed=true
    while run.status in (:ready,:running)
        a=audit_performance_state(run;tol=tol); checks+=1
        maxerror=max(maxerror,a.frontier_error,a.value_error,a.inverse_error)
        passed &= a.passed
        if !a.passed
            run.status=:audit_failure; break
        end
        step_performance!(run)
    end
    return (passed=passed,selection_policies_checked=checks,max_scaled_error=maxerror,
            snapshot=performance_snapshot(run))
end
