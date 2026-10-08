using Test, LinearAlgebra
@testset "Residual and inverse bounds" begin
    Q=Rational{BigInt}
    A=Q.([2 1; 0 4]);B=reshape(Q.([1,2]),2,1)
    X=reshape([0.249999,0.500002],2,1)
    Z=[0.5 -0.125;0 0.25]
    inv=MultiGearBandits._residual_inverse_bound(A,Z)
    @test inv!==nothing
    @test inv.contraction==0
    b=MultiGearBandits._residual_value_bounds(A,B,X,inv.bound)
    exact=reshape(Q[1//4,1//2],2,1)
    @test all(abs(exact[i]-b.x[i])<=b.error[1] for i in eachindex(exact))
    @test MultiGearBandits._residual_inverse_bound(A,zeros(2,2))===nothing
    @test MultiGearBandits._residual_unique_minimum([(Q(1),Q(2)),(Q(3),Q(4))])==1
    @test MultiGearBandits._residual_unique_minimum([(Q(1),Q(2)),(Q(2),Q(3))])==0
    @test MultiGearBandits._residual_ratio_interval(Q(-2),Q(-1),Q(1),Q(2))==(Q(-2),Q(-1,2))
    @test_throws ArgumentError MultiGearBandits._residual_ratio_interval(Q(1),Q(2),Q(0),Q(1))
end

# All probabilities are dyadic and the two closest ratios differ by 2^-35.
# Disabling the existing small-state exact-LU route reproduces the large-N issue.
function residual_test_model(;tied=false)
    P=fill(0.25,4,4,2);h=zeros(4,2);c=hcat(zeros(4),ones(4))
    h[:,1]=[1.0,1.0+(tied ? 0.0 : 2.0^-35),2.0,3.0]
    return FiniteMultiGearModel(P,h,c)
end
@testset "Strict close minima and unresolved exact ties" begin
    for criterion in (:discounted,:average)
        model=residual_test_model();kw=criterion==:discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        q=residual_certified_selection(model,ones(Int,4),[(i,1) for i in 1:4];kw...)
        @test q.status==:residual_certified
        @test q.position==1
        @test q.report["separation_lower_approx"]>0
        t=residual_certified_selection(residual_test_model(tied=true),ones(Int,4),[(i,1) for i in 1:4];kw...)
        @test t.status==:unresolved_comparison
        for alg in (:C,:R,:M,:FP), initializer in (alg==:FP ? (:legacy,:optimized) : (:optimized,))
            input=PerformanceInput(model)
            ordinary=prepare_performance_run(input,alg;kw...,fp_initializer=initializer,
                options=PerformanceOptions(exact_max_states=1))
            run_performance!(ordinary)
            @test ordinary.status==:unresolved_comparison
            run=prepare_performance_run(input,alg;kw...,fp_initializer=initializer,
                options=PerformanceOptions(exact_max_states=1,residual_fallback=true))
            while run.status in (:ready,:running)
                @test audit_performance_state(run).passed
                step_performance!(run)
            end
            s=performance_snapshot(run)
            @test s.complete
            @test s.order==[(1,1),(2,1),(3,1),(4,1)]
            @test s.counts["residual_certified_selections"]==1
            @test s.counts["residual_comparison_checks"]==1
            @test s.counts["exact_comparison_solves"]==0
            @test s.counts["fallback_direct_solves"]==1
            @test s.evidence[1]==:residual_certified
            @test length(s.residual_comparisons)==1
            @test s.residual_comparisons[1]["selected_pair"]==[1,1]
            @test s.counts["residual_inverse_rhs_columns"]==(criterion==:average ? 5 : 0)
        end
    end
end

function residual_thread_job(seed)
    local model=residual_test_model()
    local criterion=iseven(seed) ? :average : :discounted
    local kw=criterion==:average ? (;criterion=criterion) : (;criterion=criterion,beta=0.5)
    local run=prepare_performance_run(PerformanceInput(model),:FP;kw...,
        options=PerformanceOptions(exact_max_states=1,residual_fallback=true))
    run_performance!(run)
    local s=performance_snapshot(run)
    return (s.status,s.order,s.ratio,s.counts,s.residual_comparisons)
end
@testset "Thread-owned exact residual computations" begin
    serial=[residual_thread_job(i) for i in 1:8]
    parallel=Vector{Any}(undef,8)
    Threads.@threads for i in 1:8
        parallel[i]=residual_thread_job(i)
    end
    @test isequal(serial,parallel)
end
