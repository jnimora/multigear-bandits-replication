# Frozen pivot plus independent policy solves cover every active coordinate,
# including coordinates temporarily outside the feasible frontier.
Base.include(MultiGearBandits,joinpath(@__DIR__,"fixtures","fp_pivot_before_optimization.jl"))
@testset "Optimized and frozen pivots agree at every gear and deletion" begin
    for A in (1,2,4),criterion in (:discounted,:average),seed in (812,921)
        model=dyadic_validation_model(seed;n=5,A=A)
        kw=criterion==:average ? (;criterion,omega=[.1,.2,.3,.1,.3]) : (;criterion,beta=.875)
        old=initialize_reduced_fp(model;kw...);new=deepcopy(old)
        # Interleave gears and final deletions, replacing first, middle and
        # last physical slots while previously moved states remain active.
        for i in (1,4,2,5,3), _ in 1:A
            a=MultiGearBandits._pivot_reduced_fp_before_optimization!(old,i)
            b=pivot_reduced_fp!(new,i)
            @test a.accepted && b.accepted
            @test a.reason==b.reason && a.new_dimension==b.new_dimension
            @test isapprox(a.denominator,b.denominator;atol=1e-11,rtol=1e-10)
            @test old.actions==new.actions && old.state_at==new.state_at && old.slot_of==new.slot_of
            m=new.m
            @test isapprox(old.X[1:m,1:m],new.X[1:m,1:m];atol=1e-10,rtol=1e-9)
            for name in (:phi,:psi,:psi_scale,:f,:g,:g_scale)
                @test isapprox(getfield(old,name)[1:m],getfield(new,name)[1:m];atol=1e-10,rtol=1e-9)
            end
            @test audit_reduced_fp(new).passed
            for name in fieldnames(FPWorkCounts)
                name in (:audit_solves,:reference_dot_elements,:materialization_elements) && continue
                @test getfield(old.counts,name)==getfield(new.counts,name)
            end
        end
    end
end

@testset "Fused finite checks preserve rejected state" begin
    for A in (1,3),bad in (Inf,NaN)
        w=initialize_reduced_fp(fp_memory_model(8,A);beta=.75)
        w.X[1,1]=bad
        saved=deepcopy(w)
        r=pivot_reduced_fp!(w,2)
        @test !r.accepted && r.reason==:nonfinite_trial
        @test isequal(w.X,saved.X) && w.actions==saved.actions && w.state_at==saved.state_at
        for name in (:phi,:psi,:f,:g,:psi_scale,:g_scale)
            @test isequal(getfield(w,name),getfield(saved,name))
        end
    end
    for alg in (:R,:M),bad in (Inf,NaN)
        r=prepare_performance_run(PerformanceInput(fp_memory_model(8,3)),alg;beta=.75)
        w=r.core;w.inverse[1,1]=bad;saved=deepcopy(w)
        status,_=MultiGearBandits._performance_update!(w,2,r.options)
        @test status==:nonfinite_update
        @test isequal(w.inverse,saved.inverse) && w.actions==saved.actions
        @test isequal(w.values,saved.values) && isequal(w.f,saved.f) && isequal(w.g,saved.g)
    end
end

@testset "Column traversal reproduces direct frontier metrics" begin
    for criterion in (:discounted,:average),alg in (:C,:R),A in (1,3)
        model=fp_memory_model(64,A);input=PerformanceInput(model)
        kw=criterion==:average ? (;criterion,) : (;criterion,beta=.99)
        run=prepare_performance_run(input,alg;kw...)
        for step in 0:10
            w=run.core
            for k in 1:w.nfrontier
                i=w.frontier[k]
                f,g=MultiGearBandits._performance_metrics(model,i,w.actions[i],w.factor,w.values)
                @test w.f[i]==f && w.g[i]==g
            end
            @test audit_performance_state(run).passed
            step==10 || @test step_performance!(run) in (:running,:complete)
        end
    end
end
