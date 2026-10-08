# Correctness tests only. Allocation acceptance is a SEPARATE executable check.

@testset "Reduced FP / C-DS / R-DS / M-DS and exact references" begin
    for model in (one_state_model(),two_state_model()), criterion in (:discounted,:average)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        c=direct_downshift(model;kw...); r=rankone_downshift(model;kw...); m=cached_downshift(model;kw...)
        fp=reduced_fp_downshift(model;kw...,audit=true,keep_recovery=true)
        @test fp.complete && c.complete && r.complete && m.complete
        @test fp.order==[s.selected for s in c.steps]==[s.selected for s in r.steps]==[s.selected for s in m.steps]
        @test isapprox(Float64.(fp.mpi),Float64.(c.mpi);atol=1e-10,rtol=1e-9)
        @test isapprox(Float64.(fp.mpi),Float64.(m.mpi);atol=1e-10,rtol=1e-9)
        w=fp.workspace; C=ncontrollable(model); K=C*maxgear(model)
        @test w.counts.policy_factorizations==1 && w.counts.reference_factorizations==1
        @test w.counts.final_pivots==C && w.counts.nonfinal_pivots==K-C
        @test w.k==K && w.m==0 && all(iszero,w.actions)
        @test w.counts.rebuilds==0 && w.counts.audit_solves==K+1
        @test length(fp.audits)==K+1 && all(x->x.passed,fp.audits)
        @test size(w.X)==size(w.trial_X)==(C,C)
        ref=enumerate_bellman(model;kw...)
        check=validate_downshift(fp,ref)
        @test check.execution==:exact_pass
        @test check.bellman_path==:exact_pass
        @test check.mpi_dai_agreement==:numerical_pass
        plain=reduced_fp_downshift(model;kw...)
        @test plain.order==fp.order && isequal(plain.mpi,fp.mpi)
        @test plain.workspace.recovery_W===nothing && isempty(plain.audits)
        @test plain.workspace.counts.audit_solves==0
    end
end

@testset "Arbitrary pivots, packed permutations, unchanged reference data" begin
    for seed in 20260916:20260921, criterion in (:discounted,:average)
        model=dyadic_validation_model(seed;n=4,A=3)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.875) :
                                  (;criterion=criterion,omega=fill(0.25,4))
        w=initialize_reduced_fp(model;kw...,keep_recovery=true)
        refs=copy(w.ell); first=w.X; second=w.trial_X
        @test audit_reduced_fp(w).passed
        # Nontrivial deletion order, including a non-last slot and then a state
        # that has moved physically. Keep ORIGINAL labels in the sequence.
        sequence=isodd(seed) ? [2,4,1,3] : [3,1,4,2]
        for a in 3:-1:1, i in sequence
            oldm=w.m; report=pivot_reduced_fp!(w,i)
            @test report.accepted
            @test w.m==oldm-(a==1 ? 1 : 0)
            @test w.X===first || w.X===second
            @test w.trial_X===first || w.trial_X===second
            @test w.X!==w.trial_X
            @test size(w.X)==(4,4)
            @test w.ell==refs
            @test audit_reduced_fp(w;rtol=1e-7).passed
        end
        @test w.m==0 && all(iszero,w.slot_of)
        @test w.counts.final_pivots==4 && w.counts.nonfinal_pivots==8
    end
end

@testset "Binary no-reference path and retained Markov states" begin
    for model in (transient_model(),periodic_model()), criterion in (:discounted,:average)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.75) : (;criterion=criterion)
        w=initialize_reduced_fp(model;kw...)
        @test w.counts.reference_factorizations==0 && isempty(w.ell)
        @test isempty(w.fhat) && isempty(w.ghat)
        @test w.recovery_W===nothing && w.recovery_values===nothing
        @test_throws ArgumentError recover_reduced_values(w)
        for i in reverse(w.states)
            report=pivot_reduced_fp!(w,i)
            @test report.accepted && audit_reduced_fp(w).passed
            @test nstates(w.model)==nstates(model)
        end
        @test w.counts.retained_refreshes==w.counts.new_comparisons==0
        @test w.m==0
    end
    model,_=random_test_model(20260917;n=5,A=1)
    w=initialize_reduced_fp(model;beta=0.9)
    @test size(w.X)==(4,4) && nstates(w.model)==5
    for i in (4,2,5,3)
        @test pivot_reduced_fp!(w,i).accepted
        @test audit_reduced_fp(w).passed
    end
end

@testset "Frontiers and all-current cache, even when infeasible" begin
    model=dyadic_validation_model(120;n=4,A=3)
    family=OrderedThresholdFamily([1,2,3,4])
    for criterion in (:discounted,:average)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        w=initialize_reduced_fp(model;kw...,family=family)
        for k in 1:12
            expected=downshift_frontier(model,w.actions,family)
            q=MultiGearBandits._fp_frontier!(w)
            @test [(w.state_at[w.frontier_slots[t]],w.actions[w.state_at[w.frontier_slots[t]]]) for t in 1:q]==expected
            i=first(expected)[1]
            @test pivot_reduced_fp!(w,i).accepted
            @test audit_reduced_fp(w).passed # audits ALL current comparisons
        end
        @test w.m==0
    end
    for criterion in (:discounted,:average)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        model,family=cached_reentry_fixture()
        c=direct_downshift(model;kw...,family=family)
        fp=reduced_fp_downshift(model;kw...,family=family,audit=true)
        @test fp.complete && fp.order==[s.selected for s in c.steps]
        @test all(x->x.passed,fp.audits)
        @test isapprox(Float64.(fp.mpi),Float64.(c.mpi);atol=1e-9,rtol=1e-8)
    end
end

@testset "Transactional failure, corruption audit and explicit rebuild" begin
    for criterion in (:discounted,:average)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        w=initialize_reduced_fp(two_state_model();kw...,keep_recovery=true)
        before=reduced_fp_snapshot(w); old=w.X; other=w.trial_X
        report=pivot_reduced_fp!(w,1;pivot_atol=1e6)
        @test !report.accepted && report.reason==:unsafe_denominator
        @test isequal(reduced_fp_snapshot(w),before)
        @test w.X===old && w.trial_X===other
        @test pivot_reduced_fp!(w,1).accepted
        saved=reduced_fp_snapshot(w)
        w.X[1,1]+=0.1
        @test !audit_reduced_fp(w).passed
        rebuild_reduced_fp!(w)
        @test audit_reduced_fp(w).passed && w.counts.rebuilds==1
        @test w.actions==saved.actions && w.state_at[1:w.m]==saved.labels
        w.g[1]+=0.2
        @test !audit_reduced_fp(w).passed
        rebuild_reduced_fp!(w)
        @test audit_reduced_fp(w).passed
        run=reduced_fp_downshift(two_state_model();kw...,audit=true,rebuild_every=1)
        base=direct_downshift(two_state_model();kw...)
        @test run.complete && run.order==[s.selected for s in base.steps]
        @test run.workspace.counts.rebuilds==3 && all(x->x.passed,run.audits)
        failed=reduced_fp_downshift(two_state_model();kw...,pivot_atol=1e6,rebuild_on_failure=false)
        @test failed.status==:unresolved_pivot && isempty(failed.order) && all(ismissing,failed.mpi)
    end
    @test_throws ArgumentError initialize_reduced_fp(multichain_model();criterion=:average)
    # Both endpoints are unichain, but [1,0] is multichain. The proposed
    # downshift must be rejected without modifying the accepted workspace.
    P0=[0.0 1.0;0.0 1.0]; P1=[1.0 0.0;1.0 0.0]
    model=FiniteMultiGearModel(cat(P0,P1;dims=3),[2.0 1.0;2.0 1.0],[0.0 1.0;0.0 1.0])
    w=initialize_reduced_fp(model;criterion=:average)
    before=reduced_fp_snapshot(w)
    report=pivot_reduced_fp!(w,2)
    @test !report.accepted && report.reason==:unsupported_average_policy
    @test isequal(reduced_fp_snapshot(w),before)
end

@testset "Exact ties, close strict inequalities, stopping and suffixes" begin
    for criterion in (:discounted,:average)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        model=downshift_case("all_tied")
        fp=reduced_fp_downshift(model;kw...,audit=true)
        c=direct_downshift(model;kw...)
        @test fp.complete && fp.order==[s.selected for s in c.steps]
        @test fp.workspace.counts.exact_comparison_solves>0
        @test validate_downshift(fp,enumerate_bellman(model;kw...)).execution==:exact_pass
        unresolved=reduced_fp_downshift(model;kw...,exact_fallback=false)
        @test unresolved.status==:unresolved_comparison && isempty(unresolved.order)
    end
    fp=reduced_fp_downshift(close_unequal_model();beta=0.5,audit=true)
    @test fp.complete && first(fp.order)==(2,1)
    @test fp.workspace.log_evidence[1]==:exact_comparison
    for (model,expected,k) in ((nonpositive_marginal_model(),:no_positive_candidate,1),
                               (downshift_case("zero_denominator"),:complete,2))
        fp=reduced_fp_downshift(model;beta=.5)
        @test fp.status==expected && length(fp.order)==k
    end
    model=two_state_model(); family=ExplicitPolicyFamily([[2,2],[0,0]])
    fp=reduced_fp_downshift(model;beta=0.5,family=family)
    @test fp.status==:empty_frontier && !fp.complete
    prefix=reduced_fp_downshift(model;beta=0.5,max_steps=1,audit=true)
    @test prefix.status==:step_limit && length(prefix.order)==1 && length(prefix.audits)==2
    @test validate_downshift(prefix,enumerate_bellman(model;beta=0.5)).execution==:partial_path_checked
    w=initialize_reduced_fp(model;beta=0.5,actions=[1,1])
    @test run_reduced_fp!(w)==:suffix_complete
    # Each call must own its bindings as well as its workspace. `model` and
    # `fp` already exist in the enclosing testset: without explicit `local`,
    # an inner function would update those shared outer bindings.
    function fp_job(seed, criterion, interleave)
        local model=dyadic_validation_model(seed;n=3,A=2)
        interleave && yield() # Exercise overlap before the model is consumed.
        local fp=criterion==:discounted ?
            reduced_fp_downshift(model;beta=0.5,audit=true) :
            reduced_fp_downshift(model;criterion=:average,audit=true)
        interleave && yield() # Exercise overlap before the result is consumed.
        @assert all(a->a.passed,fp.audits)
        return (seed=seed,criterion=criterion,status=fp.status,order=fp.order,mpi=fp.mpi)
    end
    outer_model=model; outer_fp=fp
    # No captured testset variables may be stored in the job closure.
    @test fieldcount(typeof(fp_job))==0
    seeds=collect(701:708)
    for criterion in (:discounted,:average)
        serial=[fp_job(seed,criterion,false) for seed in seeds]
        for repetition in 1:10
            parallel=similar(serial)
            Threads.@threads for k in eachindex(seeds)
                parallel[k]=fp_job(seeds[k],criterion,true)
            end
            # Keep EXACT comparison, including partial runs and missing entries.
            @test isequal(serial,parallel)
            @testset "FP job criterion=$criterion repetition=$repetition seed=$(seeds[k])" for k in eachindex(seeds)
                @test isequal(serial[k],parallel[k])
            end
        end
    end
    @test model===outer_model
    @test fp===outer_fp
end

@testset "Nonmonotone cases and average normalization" begin
    for name in ("nonconcave_one_state","negative_off_frontier","bellman_not_sign")
        model=downshift_case(name)
        c=direct_downshift(model;beta=0.75)
        fp=reduced_fp_downshift(model;beta=0.75,audit=true)
        @test fp.status==c.status && fp.order==[s.selected for s in c.steps]
        ref=enumerate_bellman(model;beta=0.75)
        cv=validate_downshift(c,ref); fv=validate_downshift(fp,ref)
        @test fv.execution==cv.execution && fv.bellman_path==cv.bellman_path
        @test fv.reference_sign_indexability==cv.reference_sign_indexability
    end
    model=two_state_model()
    ordinary=reduced_fp_downshift(model;criterion=:average,audit=true,keep_recovery=true)
    shifted=reduced_fp_downshift(model;criterion=:average,omega=[2.0,-1.0],audit=true,keep_recovery=true)
    @test shifted.complete && shifted.order==ordinary.order
    @test all(a->a.passed,shifted.audits)
    @test isapprox(Float64.(shifted.mpi),Float64.(ordinary.mpi);atol=1e-9,rtol=1e-8)
end
