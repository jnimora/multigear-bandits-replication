@testset "Positive-work selection and monotonicity before commit" begin
    for criterion in (:discounted,:average)
        kw=criterion==:discounted ? (;criterion,beta=0.5) : (;criterion,)
        model=downshift_case("nonconcave_one_state")
        for alg in (:C,:R,:M,:FP)
            run=prepare_performance_run(PerformanceInput(model),alg;kw...)
            @test step_performance!(run)==:running
            before=copy(run.core.actions); counts=performance_counts(run)
            @test step_performance!(run)==:nonmonotone
            @test run.core.actions==before==[1]
            @test run.log.k==1 && run.log.ratio[1]==3
            @test ismissing(run.log.mpi[1,1]) && run.log.mpi[1,2]==3
            @test performance_counts(run)==counts
            @test run_performance!(run)==:nonmonotone
            @test audit_performance_state(run).passed
            @test performance_snapshot(run).selection_contract=="positive_work_monotone_v1"
        end
        for fn in (direct_downshift,rankone_downshift,cached_downshift,reduced_fp_downshift)
            run=fn(model;kw...)
            @test run.status==:nonmonotone && !run.complete
            @test ismissing(run.mpi[1,1]) && run.mpi[1,2]==3
            check=validate_downshift(run,enumerate_bellman(model;kw...))
            @test check.execution==:partial_path_checked && check.stopping==:exact_pass
            @test check.monotonicity==:partial_path_checked
        end
    end
    # Genuine model: first frontier contains a negative-work pair and a
    # positive one. The positive move is accepted, then no candidate remains.
    negative=nonpositive_marginal_model()
    for alg in (:C,:R,:M,:FP)
        run=prepare_performance_run(PerformanceInput(negative),alg;beta=0.5)
        @test step_performance!(run)==:running
        @test run.log.state[1]==2 && run.log.g[1]>0
        before=copy(run.core.actions)
        @test step_performance!(run)==:no_positive_candidate
        @test run.core.actions==before && run.log.k==1
    end
    # An exactly zero candidate is excluded by exact comparison; its work
    # becomes positive after the other state's accepted move.
    zero=downshift_case("zero_denominator")
    for fn in (direct_downshift,rankone_downshift,cached_downshift,reduced_fp_downshift)
        run=fn(zero;beta=0.5)
        @test run.complete && run.mpi[:,1]==[2,1]
        check=validate_downshift(run,enumerate_bellman(zero;beta=0.5))
        @test check.execution==:exact_pass
        @test check.frontier_positivity==:exact_violation
    end
    for alg in (:C,:R,:M,:FP)
        run=prepare_performance_run(PerformanceInput(zero),alg;beta=0.5)
        @test run_performance!(run)==:complete
        @test run.log.state==[2,1] && run.exact_comparison_solves>=1
        unresolved=prepare_performance_run(PerformanceInput(zero),alg;beta=0.5,
            options=PerformanceOptions(exact_fallback=false))
        @test run_performance!(unresolved)==:unresolved_comparison && unresolved.log.k==0
        # Exercise the residual fallback on a real mixed-sign frontier. The
        # certified position must retain the original (unfiltered) numbering.
        fallback=prepare_performance_run(PerformanceInput(negative),alg;beta=0.5,
            options=PerformanceOptions(comparison_atol=100.0,exact_fallback=false,residual_fallback=true))
        @test step_performance!(fallback)==:running
        @test fallback.log.state[1]==2 && fallback.log.evidence[1]==:residual_certified
    end
    report=residual_certified_selection(negative,[1,1],[(1,1),(2,1)];beta=0.5)
    @test report.status==:residual_certified && report.position==2
    @test report.report["excluded_nonpositive_candidates"]==1
    none=residual_certified_selection(negative,[1,0],[(1,1)];beta=0.5)
    @test none.status==:no_positive_candidate
    @test MultiGearBandits._mpi_nondecreasing(2.0,2.0,1e-10,1e-9)
    @test MultiGearBandits._mpi_nondecreasing(2.0,2.0-1e-12,1e-10,1e-9)
    @test !MultiGearBandits._mpi_nondecreasing(2.0,1.0,1e-10,1e-9)
    @test !MultiGearBandits._mpi_nondecreasing(2.0,2.0,1e-10,1e-9,2//1,1999999//1000000)
    @test_throws ArgumentError prepare_performance_run(PerformanceInput(zero),:FP;beta=.5,
        options=PerformanceOptions(monotonicity_atol=-1))
    for fn in (direct_downshift,rankone_downshift,cached_downshift,reduced_fp_downshift)
        @test_throws ArgumentError fn(zero;beta=.5,monotonicity_rtol=-1)
    end
end
