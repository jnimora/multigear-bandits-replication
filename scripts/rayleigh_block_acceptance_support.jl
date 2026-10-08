# Included INSIDE RayleighBlock. None of these reference routines is called by
# numerical_downshift. Acceptance happens after autonomous live selection.
scaled_error(x,y)=abs(Float64(x)-Float64(y))/(1+max(abs(Float64(x)),abs(Float64(y))))
function require_accept(ok,msg)
    ok || error("Rayleigh block acceptance failed: "*msg)
    nothing
end
function check_live_against_exact(m,run;tol=1e-9)
    require_accept(run.complete,"live path incomplete: $(run.status): $(run.message)")
    exact=exact_downshift(m;method=:scalar)
    require_accept(run.order==exact.order,"independently selected path differs from exact scalar path")
    err=0.0
    for k in eachindex(run.order)
        for (v,q) in ((run.assignments[k],exact.assignments[k]),(run.f[k],exact.f[k]),(run.g[k],exact.g[k]))
            err=max(err,scaled_error(v,q))
        end
        b=run.bounds[k]
        for (lo,hi,q) in ((b[1],b[2],exact.assignments[k]),(b[3],b[4],exact.f[k]),(b[5],b[6],exact.g[k]))
            require_accept(Q(lo)<=q<=Q(hi),"returned enclosure misses its exact selected metric")
        end
    end
    require_accept(err<=tol,"selected metric error exceeds independent audit tolerance")
    exact,err
end
function check_frontier_enclosures(m,actions;tol=1e-9)
    w=NumericBlockWorkspace(m;actions=actions)
    e=numerical_frontier!(w);ref=excursion(m,actions)
    require_accept(contains_exact(e.hbar,ref.hbar) && contains_exact(e.cbar,ref.cbar),"rate enclosure")
    rows=RayleighAoII.scalar_frontier(m,actions,ref)
    require_accept(length(rows)==length(e.frontier),"frontier size")
    err=max(scaled_error(e.hbar.value,ref.hbar),scaled_error(e.cbar.value,ref.cbar))
    for (x,y) in zip(e.frontier,rows)
        require_accept((x.h,x.a)==(y[1],y[2]),"frontier label")
        for (v,q) in ((x.f,y[3]),(x.g,y[4]),(x.index,y[5]))
            require_accept(contains_exact(v,q),"frontier enclosure misses exact metric")
            err=max(err,scaled_error(v.value,q))
        end
    end
    # Raw candidates can lose relative accuracy through cancellation. They are
    # not accepted unless the live selector's returned enclosure is sufficiently
    # narrow. Here check containment; report raw error rather than relaxing the
    # selector's output tolerance or demanding accurate rejected candidates.
    err
end
function generic_topology_checks(m,exact;tol=1e-7)
    input=topology_performance_input(m);fm=input.model;qm=ExactMultiGearModel(fm)
    em=dense_model(m)
    require_accept(qm.P==em.P && qm.h==em.h && qm.c==em.c,
        "generic reference conversion is not the same exact model; this check requires dyadic fixtures")
    family=OrderedThresholdFamily(collect(2:m.H+1))
    opts=PerformanceOptions(exact_fallback=true,exact_max_states=8,terminal_update=false,
        residual_fallback=true,residual_max_states=256)
    comparisons=Dict{String,Any}[]
    expected=[(h+1,a) for (h,a) in exact.order]
    for (name,method,init) in (("C",:C,:optimized),("R",:R,:optimized),("M",:M,:optimized),
                              ("FP",:FP,:optimized),("FP_legacy",:FP,:legacy))
        audit=audit_performance_run(input,method;criterion=:average,family=family,options=opts,
            fp_initializer=init,tol=tol)
        s=audit.snapshot
        require_accept(audit.passed && s.complete && s.order==expected,"topology-admitted $name execution/path/audit")
        error=maximum(scaled_error(s.ratio[k],exact.assignments[k]) for k in eachindex(s.ratio))
        require_accept(error<=tol,"topology-admitted $name assigned ratios")
        push!(comparisons,Dict{String,Any}("algorithm"=>name,"passed"=>true,"complete"=>s.complete,
            "path_matches_exact"=>true,"maximum_scaled_index_error"=>error,
            "selection_policy_checks"=>audit.selection_policies_checked,"maximum_policy_audit_error"=>audit.max_scaled_error,
            "work_counts"=>Dict(string(k)=>v for (k,v) in s.counts),
            "all_entries_positive"=>input.all_policies_positive,"average_topology"=>string(input.average_topology)))
    end
    comparisons
end
function numerical_checkpoint(id,m,run)
    Dict{String,Any}("id"=>id,"phase"=>"numerical execution before independent acceptance",
        "passed"=>false,"acceptance_pending"=>true,"complete"=>run.complete,"status"=>string(run.status),
        "message"=>run.message,"dai_verified"=>false,"certified_prefix"=>run.certified_prefix,
        "H"=>m.H,"A"=>length(m.p)-1,"wR"=>string(m.wR),"p_exact"=>string.(m.p),
        "c_exact"=>string.(m.c),"counts"=>run.counts,
        "shifts"=>[Dict("age"=>h,"upper_gear"=>a,"value"=>run.assignments[k],"f"=>run.f[k],"g"=>run.g[k],
            "bounds_index_f_g"=>run.bounds[k],"selection_mode"=>string(run.modes[k]))
            for (k,(h,a)) in enumerate(run.order)])
end
function accept_numerical_case(id::AbstractString,m::AoIIModel{Q};verify_dai::Bool=false,check_generic::Bool=false,
                               options=BlockOptions(),audit_tol::Float64=1e-9,checkpoint=nothing)
    # Live run FIRST, with no expected order, reference array, or exact DAI input.
    run=numerical_downshift(m;options=options)
    saved=numerical_checkpoint(id,m,run)
    checkpoint===nothing || checkpoint(saved)
    if !run.complete
        saved["acceptance_pending"]=false
        saved["error"]="Live numerical path did not complete; the valid prefix is retained."
        return saved
    end
    exact,error=check_live_against_exact(m,run;tol=audit_tol)
    cert=Dict{String,Any}("passed"=>false,"scope"=>"not requested: numerical stress fixture only")
    oracle_policies=0;dai=false
    if verify_dai
        cert=exact_certificate(m)
        require_accept(cert["passed"],"full threshold-family certificate")
        oracle=enumerate_bellman(dense_model(m);criterion=:average,max_states=8,
            max_policies=1000,time_limit_s=180)
        require_accept(oracle.complete && oracle.coverage==:exact_pass && oracle.sign_indexability==:exact_pass &&
            oracle.dai==exact.mpi,"independent all-charge Bellman DAI verification")
        oracle_policies=length(oracle.records);dai=true
    end
    # This loop audits freshly initialized arbitrary continuation policies and
    # agrees with the live path; it is outside the numerical algorithm.
    actions=fill(length(m.p)-1,m.H);frontier_error=0.0;frontier_refusals=0
    for (h,a) in run.order
        try
            frontier_error=max(frontier_error,check_frontier_enclosures(m,actions;tol=audit_tol))
        catch err
            err isa NumericalIssue || rethrow()
            frontier_refusals+=1
        end
        actions[h]-=1
    end
    comparisons=check_generic ? generic_topology_checks(m,exact) : Dict{String,Any}[]
    Dict{String,Any}("id"=>id,"passed"=>true,"H"=>m.H,"A"=>length(m.p)-1,"criterion"=>"average",
        "complete"=>run.complete,"status"=>string(run.status),"autonomous_path_matches_exact"=>true,
        "selection_certification"=>"outward-enclosed comparisons; exact algebraic equality or counted current-policy rational fallback",
        "dai_verified"=>dai,"certificate"=>cert,"all_policy_oracle_policies"=>oracle_policies,
        "maximum_scaled_selected_metric_error"=>error,"maximum_scaled_fresh_frontier_error"=>frontier_error,
        "fresh_frontier_refusals"=>frontier_refusals,
        "exact_selected_metrics_enclosed"=>true,"generic_checks"=>comparisons,"counts"=>run.counts,
        "wR"=>string(m.wR),"source_states"=>m.nsrc,"success_exact"=>string.(m.success),
        "p_exact"=>string.(m.p),"c_exact"=>string.(m.c),"kappa"=>string(m.kappa),"epsilon"=>string(m.epsilon),
        "index_by_age"=>[Float64.(run.mpi[h,:]) for h in 1:m.H],
        "exact_index_by_age"=>[string.(exact.mpi[h,:]) for h in 1:m.H],
        "shifts"=>[Dict("age"=>h,"upper_gear"=>a,"value"=>run.assignments[k],"f"=>run.f[k],"g"=>run.g[k],
            "bounds_index_f_g"=>run.bounds[k],"selection_mode"=>string(run.modes[k]),
            "exact_value"=>string(exact.assignments[k])) for (k,(h,a)) in enumerate(run.order)],
        "resource_scope"=>"constructed synchronization-gain resource; not calibrated battery energy",
        "scope"=>(verify_dai ? "small exact DAI fixture; no timing or allocation claim" :
                             "numerical path stress only; no indexability claim"))
end
function stress_cases()
    # Deterministic numerical tests, not random draws searched for favorable paths.
    [
      ("near_unit_survival",from_success(16;wR=1-1//BigInt(2)^32,success=[0//1,1//4,3//4],epsilon=0)),
      ("near_zero_survival",from_success(16;wR=1-1//BigInt(2)^32,
          success=[0//1,1//2,1-1//BigInt(2)^28],epsilon=0)),
      ("near_gear_ties",from_success(6;epsilon=1//BigInt(2)^48)),
      ("non_dyadic_primitives",from_success(8;wR=4//5,success=[0//1,1//5,4//5],epsilon=1//32)),
      ("long_live_path",from_success(128;success=[0//1,1//4,3//4],epsilon=0))
    ]
end
function topology_rejection_checks()
    m=from_success(4);fm=dense_model(m;exact=false);unchanged=copy(fm.P)
    ordinary=PerformanceInput(fm);accepted=PerformanceInput(fm;average_topology=:reset_or_increment)
    require_accept(!ordinary.all_policies_positive && !accepted.all_policies_positive,"truthful positivity flag")
    require_accept(fm.P==unchanged,"admission modified a transition")
    function rejects(fn)
        try;fn();return false;catch err;err isa ArgumentError || rethrow();return true;end
    end
    require_accept(rejects(()->prepare_performance_run(ordinary,:C;criterion=:average)),"default sparse guard bypassed")
    require_accept(rejects(()->PerformanceInput(fm;average_topology=:unverified)),"unknown admission accepted")
    rejected=0
    for kind in (:missing_reset,:missing_progression,:unexpected_edge,:zero_cannot_leave)
        P=copy(fm.P)
        if kind==:missing_reset;P[2,3,1]+=P[2,1,1];P[2,1,1]=0.0
        elseif kind==:missing_progression;P[2,1,1]+=P[2,3,1];P[2,3,1]=0.0
        elseif kind==:unexpected_edge;P[2,2,1]=P[2,3,1]/2;P[2,3,1]/=2
        else;P[1,1,1]=1.0;P[1,2,1]=0.0
        end
        altered=FiniteMultiGearModel(P,fm.h,fm.c;controllable=fm.controllable)
        require_accept(rejects(()->PerformanceInput(altered;average_topology=:reset_or_increment)),"invalid support admitted: $kind")
        rejected+=1
    end
    # Extra uncontrollable positive state also invalidates this proof's domain.
    mask=copy(fm.controllable);mask[end]=false
    altered=FiniteMultiGearModel(fm.P,fm.h,fm.c;controllable=mask)
    require_accept(rejects(()->PerformanceInput(altered;average_topology=:reset_or_increment)),"unsupported controllability mask admitted")
    P=copy(fm.P);P[2,:,1].*=0.5
    loose=FiniteMultiGearModel(P,fm.h,fm.c;controllable=fm.controllable,row_atol=0.6)
    require_accept(rejects(()->PerformanceInput(loose;average_topology=:reset_or_increment)),"loose row-sum tolerance bypassed stochastic admission")
    Dict{String,Any}("passed"=>true,"invalid_topologies_rejected"=>rejected+2,"default_sparse_guard_retained"=>true,
        "transition_arrays_unchanged"=>true,"unknown_admission_rejected"=>true)
end
function evaluator_stress_checks()
    # Long-block evaluation deliberately follows manufactured policies, NOT a
    # claimed index path. The live H=128 fixture is tested separately above.
    m=from_success(1600;success=[0//1,1//4,3//4],epsilon=0)
    w=NumericBlockWorkspace(m);A=length(m.p)-1
    require_accept(w.stats[end].power.value==0,"long block did not exercise Float64 underflow")
    actions=fill(A,m.H);remaining=[1599,1000,800,760,700,50,1];points=Dict{String,Any}[]
    shifted=0;maxerr=0.0
    for target in remaining
        while w.lengths[end]>target
            numerical_shift!(w,shifted+1,A);actions[shifted+1]-=1;shifted+=1
        end
        e=numerical_frontier!(w);ref=excursion(m,actions);rows=RayleighAoII.scalar_frontier(m,actions,ref)
        require_accept(contains_exact(e.hbar,ref.hbar) && contains_exact(e.cbar,ref.cbar),"long-block rate enclosure")
        for (x,y) in zip(e.frontier,rows)
            require_accept((x.h,x.a)==(y[1],y[2]),"long-block frontier label")
            for (v,q) in ((x.f,y[3]),(x.g,y[4]),(x.index,y[5]))
                require_accept(contains_exact(v,q),"long-block interval lost exact metric")
                maxerr=max(maxerr,scaled_error(v.value,q))
            end
        end
        if target<=700;require_accept(w.stats[end].power.value>0,"survival power did not recover after shrinking");end
        push!(points,Dict("remaining_high_block"=>target,"power"=>w.stats[end].power.value))
    end
    require_accept(maxerr<=1e-9 && w.counts["underflow_stat_rebuilds"]>0,"long-block stability/rebuild evidence")
    # Insufficient comparison resources MUST yield an incomplete prefix, not a
    # tolerance tie or a fabricated DAI. This fixture succeeds with fallback.
    close=from_success(4;epsilon=1//BigInt(2)^50)
    denied=numerical_downshift(close;options=BlockOptions(exact_fallback=false,value_atol=1e-30,value_rtol=1e-30))
    require_accept(!denied.complete && denied.status==:unresolved_comparison && !denied.dai_verified,
        "unresolved numerical output was silently accepted")
    allowed=numerical_downshift(close)
    check_live_against_exact(close,allowed)
    require_accept(allowed.counts["exact_policy_evaluations"]>0,"near-tie test never exercised exact fallback")
    # Retain an intentionally excessive-curvature stress model. The first shift
    # is valid, then a nonpositive marginal resource is confirmed exactly. This
    # tests rejection, not a positive-curvature indexability assertion.
    bad=from_success(16;wR=1-1//BigInt(2)^32,success=[0//1,1//2,1-1//BigInt(2)^28],epsilon=1//8)
    rejected=numerical_downshift(bad)
    require_accept(!rejected.complete && rejected.status==:nonpositive_resource &&
        length(rejected.order)==1 && !rejected.dai_verified,"negative-resource case was not rejected with its prefix")
    badacts=fill(2,16);badacts[1]=1;be=excursion(bad,badacts)
    negative=false
    for h in 1:16
        a=badacts[h]
        if a>0 && (h==1 || badacts[h-1]<a)
            negative |= bad.c[a+1]-bad.c[a]-(bad.p[a+1]-bad.p[a])*be.gamma[min(h+1,16)]<=0
        end
    end
    require_accept(negative,"reference did not confirm nonpositive frontier resource")
    # Float64 collapsing a positive edge is a rejected representation, not a
    # topology repair. No epsilon transitions or clipped probabilities are used.
    collapsed=from_success(2;wR=1-1//BigInt(2)^100,success=[0//1,1-1//BigInt(2)^100])
    representation_rejected=false
    try;NumericBlockWorkspace(collapsed)
    catch err;err isa ArgumentError || rethrow();representation_rejected=true;end
    require_accept(representation_rejected,"collapsed float edge was not rejected")
    Dict{String,Any}("passed"=>true,"long_block_H"=>1600,"long_block_checkpoints"=>points,
        "maximum_scaled_long_block_metric_error"=>maxerr,"long_block_counts"=>w.counts,
        "unresolved_without_fallback"=>string(denied.status),"preserved_prefix_length"=>length(denied.order),
        "near_tie_fallback_count"=>allowed.counts["exact_policy_evaluations"],
        "nonpositive_resource_case"=>Dict("status"=>string(rejected.status),"prefix_length"=>length(rejected.order),
            "exact_confirmation"=>negative,"diagnostic"=>rejected.message),
        "collapsed_float_edge_rejected"=>representation_rejected,
        "scope"=>"numerical evaluator, representation and failure-path tests; no large-H DAI certificate")
end
