# Helpers at top level: NO accidental capture of enclosing @testset bindings.
function performance_test_job(seed,criterion,algorithm)
    model,_=random_test_model(seed;n=4,A=2)
    input=PerformanceInput(model)
    kw=criterion==:discounted ? (;criterion=criterion,beta=0.75) : (;criterion=criterion)
    run=prepare_performance_run(input,algorithm;kw...)
    yield()
    run_performance!(run)
    s=performance_snapshot(run)
    yield()
    return (seed=seed,criterion=criterion,algorithm=algorithm,status=s.status,order=s.order,mpi=s.mpi)
end

@testset "Minimal output and the independently checked reference algorithms" begin
    for model in (one_state_model(),two_state_model()), criterion in (:discounted,:average)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        input=PerformanceInput(model); reference=direct_downshift(model;kw...)
        @test reference.complete
        K=ncontrollable(model)*maxgear(model)
        for algorithm in (:C,:R,:M,:FP), terminal in (false,true)
            options=PerformanceOptions(terminal_update=terminal)
            run=prepare_performance_run(input,algorithm;kw...,options=options)
            @test audit_performance_state(run).passed
            @test run_performance!(run)==:complete
            s=performance_snapshot(run)
            @test s.order==[x.selected for x in reference.steps]
            @test isapprox(Float64.(s.mpi),Float64.(reference.mpi);atol=1e-10,rtol=1e-9)
            @test s.terminal_update_skipped==!terminal
            @test s.counts["assignments"]==K
            @test sum(run.core.actions)==(terminal ? 0 : 1)
            if algorithm==:C
                @test s.counts["policy_factorizations"]==K+Int(terminal)
                @test isempty(run.core.inverse)
            elseif algorithm in (:R,:M)
                @test s.counts["policy_factorizations"]==1
                @test s.counts["rankone_updates"]==K-1+Int(terminal)
                @test run.core.lu_workspace===nothing && isempty(run.core.system)
            else
                @test s.counts["nonfinal_pivots"]+s.counts["final_pivots"]==K-1+Int(terminal)
                @test s.counts["reference_factorizations"]==1
                @test run.core.recovery_W===nothing
            end
            @test audit_performance_state(run).passed
            a=audit_performance_run(input,algorithm;kw...,options=options)
            @test a.passed && a.selection_policies_checked==K
            @test performance_agreement(s,a.snapshot)
            before=copy(s.mpi)
            run.log.mpi[1,1]=12345.0
            @test isequal(s.mpi,before) # owned snapshot is not a workspace alias
        end
    end
end

@testset "Every arbitrary update, stable buffers, no stale frontier rows" begin
    for seed in 810:813, criterion in (:discounted,:average), alg in (:C,:R,:M)
        model=dyadic_validation_model(seed;n=4,A=3)
        input=PerformanceInput(model)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.875) :
                                  (;criterion=criterion,omega=fill(0.25,4))
        run=prepare_performance_run(input,alg;kw...); w=run.core
        first=w.inverse; second=w.trial_inverse; rows=w.cache_rows
        factors=alg==:C ? w.lu_workspace.factors : nothing
        pivots=alg==:C ? w.lu_workspace.ipiv : nothing
        @test audit_performance_state(run).passed
        for a in 3:-1:1, i in (3,1,4,2)
            reason,_=MultiGearBandits._performance_update!(w,i,run.options)
            @test reason==:accepted
            @test w.actions[i]==a-1
            @test audit_performance_state(run).passed
            @test w.cache_rows===rows
            if alg==:C
                @test w.lu_workspace.factors===factors
                @test w.lu_workspace.ipiv===pivots
            else
                @test w.inverse===first || w.inverse===second
                @test w.trial_inverse===first || w.trial_inverse===second
                @test w.inverse!==w.trial_inverse
            end
        end
        @test all(iszero,w.actions)
    end
    for criterion in (:discounted,:average), family in
        (OrderedThresholdFamily([1,2]),ExplicitPolicyFamily([[2,2],[1,2],[1,1],[0,1],[0,0]]))
        input=PerformanceInput(two_state_model())
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        base=audit_performance_run(input,:C;kw...,family=family)
        reference=direct_downshift(input.model;kw...,family=family)
        @test base.passed && base.snapshot.status==reference.status
        @test base.snapshot.order==[s.selected for s in reference.steps]
        for alg in (:R,:M,:FP)
            a=audit_performance_run(input,alg;kw...,family=family)
            @test a.passed
            @test performance_agreement(base.snapshot,a.snapshot)
        end
    end
end

@testset "Identical comparison contract, ties, failed and partial paths" begin
    tied=PerformanceInput(downshift_case("all_tied"))
    for criterion in (:discounted,:average), alg in (:C,:R,:M,:FP)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        run=prepare_performance_run(tied,alg;kw...)
        @test run_performance!(run)==:complete
        s=performance_snapshot(run)
        @test s.order==[(1,2),(1,1),(2,2),(2,1)]
        @test s.counts["exact_comparison_solves"]>0
        @test all(x->x==2.0,s.mpi)
        unresolved=prepare_performance_run(tied,alg;kw...,options=PerformanceOptions(exact_fallback=false))
        @test run_performance!(unresolved)==:unresolved_comparison
        @test unresolved.log.k==0
        stopped=prepare_performance_run(PerformanceInput(two_state_model()),alg;kw...)
        @test run_performance!(stopped;max_steps=1)==:step_limit
        @test stopped.log.k==1
        @test run_performance!(stopped)==:complete
        @test stopped.log.k==4
    end
    for alg in (:C,:R,:M,:FP)
        input=PerformanceInput(nonpositive_marginal_model())
        run=prepare_performance_run(input,alg;beta=0.5)
        @test run_performance!(run)==:no_positive_candidate
        @test run.log.k==1 && run.log.state[1]==2
        @test_throws ArgumentError prepare_performance_run(input,alg;criterion=:average)
        family=ExplicitPolicyFamily([[2,2],[0,0]])
        empty=prepare_performance_run(PerformanceInput(two_state_model()),alg;beta=0.5,family=family)
        @test run_performance!(empty)==:empty_frontier
        close=prepare_performance_run(PerformanceInput(close_unequal_model()),alg;beta=0.5)
        @test run_performance!(close)==:complete
        @test (close.log.state[1],close.log.gear[1])==(2,1)
    end
end

@testset "Generator, uncontrollable states, normalization and concurrency" begin
    p=performance_primitives(8,3,1)
    p2=performance_primitives(8,3,1)
    @test p.q==p2.q && p.theta==p2.theta && p.weights==p2.weights
    @test all(x->x>0,p.q)
    @test all(issorted(p.theta[i,:];rev=true) for i in 1:8)
    m0=performance_model(p,0.0); m1=performance_model(p,0.1)
    @test m0.h==m1.h && m0.c==m1.c
    @test m0.P[:,:,1]==m0.P[:,:,2]==m1.P[:,:,1]
    for criterion in (:discounted,:average), eta in (0.0,0.1), alg in (:C,:R,:M,:FP)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.99) : (;criterion=criterion)
        a=audit_performance_run(PerformanceInput(performance_model(p,eta)),alg;kw...)
        @test a.passed
        if eta==0
            @test a.snapshot.complete
            @test isapprox(Float64.(a.snapshot.mpi),p.theta;atol=1e-9,rtol=1e-8)
        end
    end
    for alg in (:C,:R,:M,:FP)
        input=PerformanceInput(two_state_model())
        a=prepare_performance_run(input,alg;criterion=:average,omega=[1.0,0.0])
        b=prepare_performance_run(input,alg;criterion=:average,omega=[0.25,0.75])
        run_performance!(a); run_performance!(b)
        @test performance_agreement(performance_snapshot(a),performance_snapshot(b))
    end
    seeds=collect(901:906)
    for alg in (:C,:R,:M,:FP), criterion in (:discounted,:average)
        serial=[performance_test_job(seed,criterion,alg) for seed in seeds]
        for _ in 1:3
            parallel=similar(serial)
            Threads.@threads for j in eachindex(seeds)
                parallel[j]=performance_test_job(seeds[j],criterion,alg)
            end
            @test isequal(serial,parallel)
        end
    end
end

@testset "Paired summaries never turn failed or unmatched runs into speedups" begin
    include(joinpath(@__DIR__,"..","scripts","performance_support.jl"))
    rows=Dict{String,Any}[
        Dict("case_id"=>"x","algorithm"=>"C","phase"=>"index_loop","repetition"=>1,
             "executed"=>true,"eligible"=>true,"seconds"=>2.0,"allocated_bytes"=>0,"gc_seconds"=>0.0),
        Dict("case_id"=>"x","algorithm"=>"FP","phase"=>"index_loop","repetition"=>1,
             "executed"=>true,"eligible"=>true,"seconds"=>1.0,"allocated_bytes"=>0,"gc_seconds"=>0.0),
        Dict("case_id"=>"x","algorithm"=>"C","phase"=>"index_loop","repetition"=>2,
             "executed"=>true,"eligible"=>false,"seconds"=>0.2),
        Dict("case_id"=>"x","algorithm"=>"FP","phase"=>"index_loop","repetition"=>2,
             "executed"=>true,"eligible"=>true,"seconds"=>1.1,"allocated_bytes"=>0,"gc_seconds"=>0.0)]
    summary,ratios=pilot_summaries(rows)
    @test length(ratios)==2
    @test ratios[1]["baseline_over_FP"]==2.0
    @test !haskey(ratios[2],"baseline_over_FP") && !ratios[2]["eligible_pair"]
    c=only([r for r in summary if r["algorithm"]=="C"])
    @test c["attempts"]==2 && c["eligible"]==1 && c["median_seconds"]==2.0
    mktempdir() do dir
        path=joinpath(dir,"summary.csv")
        pilot_csv(path,summary,["case_id","algorithm","eligible"])
        @test isfile(path)
        p=performance_primitives(4,2,1)
        hash=pilot_archive_primitives(dir,p)
        @test length(hash)==64
        record=TOML.parsefile(joinpath(dir,"primitives_N4_A2_draw1.toml"))
        @test reshape(record["q"],4,2)==p.q
    end
end
