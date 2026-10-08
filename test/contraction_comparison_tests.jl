using Test,LinearAlgebra,Random,TOML
@testset "Contraction bounds enclose independent exact dense policy solves" begin
    Q=Rational{BigInt};parseq(s)=parse(BigInt,split(s,"//")[1])//parse(BigInt,split(s,"//")[2])
    for seed in 1:12,criterion in (:discounted,:average),actions in ([1,1,1],[2,1,2],[0,1,2])
        m=dyadic_validation_model(seed);n=3;fac=criterion==:average ? Q(1) : Q(0.99)
        P=Q[m.P[i,j,actions[i]+1] for i in 1:n,j in 1:n]
        B=Q[isodd(t) ? m.h[i,actions[i]+1] : m.c[i,actions[i]+1] for i in 1:n,t in 1:2]
        if criterion==:discounted
            X=(Matrix{Q}(I,n,n)-fac*P)\B
        else
            M=zeros(Q,n+1,n+1);M[1:n,1:n]=Matrix{Q}(I,n,n)-P
            M[1:n,n+1].=1;M[n+1,1]=1
            X=(M\vcat(B,zeros(Q,1,2)))[1:n,:]
        end
        frontier=[(i,actions[i]) for i in 1:n if actions[i]>0]
        # A signed normalization stresses the normalization-invariant span bound.
        r=residual_certified_selection(m,actions,frontier;criterion,beta=.99,
            omega=[-1.0,1.0,1.0],bound_method=:contraction)
        @test r.report["inverse_rhs_columns"]==0
        exactratios=Tuple{Int,Q}[]
        for (k,(i,a)) in enumerate(frontier)
            delta=fac.*(Q.(m.P[i,:,a+1])-Q.(m.P[i,:,a]))
            f=Q(m.h[i,a])-Q(m.h[i,a+1])-dot(delta,X[:,1])
            g=Q(m.c[i,a+1])-Q(m.c[i,a])+dot(delta,X[:,2])
            row=r.report["frontier_bounds"][k]
            @test parseq(row["f_lower_exact"])<=f<=parseq(row["f_upper_exact"])
            @test parseq(row["g_lower_exact"])<=g<=parseq(row["g_upper_exact"])
            if g>0
                push!(exactratios,(k,f/g))
                if haskey(row,"ratio_lower_exact")
                    @test parseq(row["ratio_lower_exact"])<=f/g<=parseq(row["ratio_upper_exact"])
                end
            end
        end
        if r.status==:residual_certified
            @test !isempty(exactratios)
            @test r.position==exactratios[argmin(last.(exactratios))][1]
        end
    end
end

@testset "Contraction selector retains refusal and acceptance contracts" begin
    for criterion in (:discounted,:average)
        kw=criterion==:discounted ? (;criterion,beta=.5) : (;criterion,)
        for tied in (false,true)
            m=residual_test_model(;tied)
            r=residual_certified_selection(m,ones(Int,4),[(i,1) for i in 1:4];kw...,bound_method=:contraction)
            @test r.status==(tied ? :unresolved_comparison : :residual_certified)
            for alg in (:C,:R,:M,:FP)
                run=prepare_performance_run(PerformanceInput(m),alg;kw...,
                    options=PerformanceOptions(exact_fallback=false,residual_fallback=true,residual_bound=:contraction))
                before=copy(run.core.actions);run_performance!(run)
                @test run.status==(tied ? :unresolved_comparison : :complete)
                if tied
                    @test run.log.k==0 && run.core.actions==before
                else
                    @test run.log.state==[1,2,3,4]
                    @test run.residual_certified_selections==1 && run.residual_inverse_rhs_columns==0
                    @test audit_performance_state(run).passed
                end
            end
        end
        m=downshift_case("nonconcave_one_state")
        for alg in (:C,:R,:M,:FP)
            run=prepare_performance_run(PerformanceInput(m),alg;kw...,
                options=PerformanceOptions(comparison_atol=100.,exact_fallback=false,residual_fallback=true,residual_bound=:contraction))
            @test step_performance!(run)==:running
            before=copy(run.core.actions)
            @test step_performance!(run)==:nonmonotone
            @test run.log.k==1 && run.core.actions==before
        end
    end
    m=nonpositive_marginal_model()
    r=residual_certified_selection(m,[1,1],[(1,1),(2,1)];beta=.5,bound_method=:contraction)
    @test r.status==:residual_certified && r.position==2
    @test r.report["excluded_nonpositive_candidates"]==1
    r=residual_certified_selection(m,[1,0],[(1,1)];beta=.5,bound_method=:contraction)
    @test r.status==:no_positive_candidate
    # Exactly stochastic periodic rows have no one-step common mass.
    P=cat([0. 1.;1. 0.],[0. 1.;1. 0.];dims=3)
    m=FiniteMultiGearModel(P,[1. 0.;2. 0.],[0. 1.;0. 1.])
    r=residual_certified_selection(m,[1,1],[(1,1),(2,1)];criterion=:average,bound_method=:contraction)
    @test r.status==:unresolved_comparison && r.report["factorizations"]==0
    P=fill(.5,2,2,2);P[1,1,2]=nextfloat(.5)
    m=FiniteMultiGearModel(P,[1. 0.;2. 0.],[0. 1.;0. 1.])
    r=residual_certified_selection(m,[1,1],[(1,1),(2,1)];criterion=:average,bound_method=:contraction)
    @test r.status==:unresolved_comparison && occursin("exactly stochastic",r.report["reason"])
    @test_throws ArgumentError prepare_performance_run(PerformanceInput(m),:R;beta=.5,
        options=PerformanceOptions(residual_bound=:unknown))
end
