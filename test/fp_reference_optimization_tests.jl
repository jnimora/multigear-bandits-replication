@testset "Reference compression preserves exact stored differences and general costs" begin
    # Rounded subtraction equality alone must not authorize sharing.
    M=MultiGearBandits
    for (x,y) in ((1.0,2.0^-55),(.25,.125),(nextfloat(.5),.5),(0.0,nextfloat(0.0)))
        hi,lo=M._preparation_difference(x,y)
        @test Rational{BigInt}(hi)+Rational{BigInt}(lo)==Rational{BigInt}(x)-Rational{BigInt}(y)
    end
    @test 1.0-2.0^-55==1.0
    @test M._preparation_difference(1.0,2.0^-55)!=M._preparation_difference(1.0,0.0)
    for A in (2,4,8),criterion in (:discounted,:average),masked in (false,true)
        n=8;P=fill(1.0/n,n,n,A+1);h=zeros(n,A+1);c=zeros(n,A+1)
        for a in 0:A,i in 1:n
            P[i,i,a+1]+=a/256;P[i,mod1(i+1,n),a+1]-=a/256
            h[i,a+1]=(i+1)*a+a*a/7;c[i,a+1]=i*a+a*a/3
        end
        model=FiniteMultiGearModel(P,h,c;controllable=masked ? [true,false,true,true,false,true,true,true] : trues(n))
        input=PerformanceInput(model);kw=criterion==:average ? (;criterion) : (;criterion,beta=.875)
        w=initialize_reduced_fp_optimized(input;kw...)
        @test size(w.ell,2)==ncontrollable(model)
        @test length(w.reference_columns)==ncontrollable(model)*(A-1)
        @test audit_fp_initialization(input;kw...,keep_recovery=true).passed
        @test audit_fp_initialization(input;kw...,actions=zeros(Int,n)).passed
    end
end

@testset "M threshold cache uses one column per feasible gear boundary" begin
    M=MultiGearBandits
    for criterion in (:discounted,:average),A in (1,3),n in (8,16)
        input=PerformanceInput(fp_memory_model(n,A))
        family=OrderedThresholdFamily(reverse(collect(1:n)))
        kw=criterion==:average ? (;criterion) : (;criterion,beta=.875)
        run=prepare_performance_run(input,:M;kw...,family)
        @test size(run.core.cache_rows)==(n+(criterion==:average),A)
        w=run.core
        for a in A:-1:1,i in reverse(1:n)
            @test M._performance_feasible(w,family,i)
            @test audit_performance_state(run).passed
            status,_=M._performance_update!(w,i,run.options)
            @test status==:accepted
            columns=[M._performance_cache_column(w,family,j,w.actions[j]) for j in w.states
                     if w.actions[j]>0 && M._performance_feasible(w,family,j)]
            @test length(unique(columns))==length(columns)
        end
        @test audit_performance_state(run).passed
    end
end

@testset "Frontier cache reentry, packed deletions, full-tableau audit and rollback" begin
    M=MultiGearBandits
    for criterion in (:discounted,:average),family_kind in (:threshold,:explicit),seed in (71,83)
        n=5;A=4;model=dyadic_validation_model(seed;n,A)
        order=[(i,a) for a in A:-1:1 for i in 1:n]
        policies=[fill(A,n)];act=fill(A,n)
        for (i,a) in order;act=copy(act);act[i]-=1;push!(policies,act);end
        family=family_kind==:threshold ? OrderedThresholdFamily(collect(1:n)) : ExplicitPolicyFamily(policies)
        kw=criterion==:average ? (;criterion) : (;criterion,beta=.875)
        full=initialize_reduced_fp(model;kw...,family)
        opt=initialize_reduced_fp_optimized(PerformanceInput(model);kw...,family)
        opt.frontier_only=true
        for (i,a) in order
            @test M._fp_feasible(opt,family,i)
            @test isapprox(full.f[full.slot_of[i]],opt.f[opt.slot_of[i]];atol=1e-9,rtol=1e-8)
            @test isapprox(full.g[full.slot_of[i]],opt.g[opt.slot_of[i]];atol=1e-9,rtol=1e-8)
            @test pivot_reduced_fp!(full,i).accepted
            @test pivot_reduced_fp!(opt,i).accepted
            @test full.actions==opt.actions && full.state_at==opt.state_at
            @test audit_reduced_fp(opt).passed
            for s in 1:opt.m
                j=opt.state_at[s]
                if M._fp_feasible(opt,family,j)
                    @test isfinite(opt.f[s]) && isfinite(opt.g[s])
                    @test isapprox(full.f[s],opt.f[s];atol=1e-9,rtol=1e-8)
                    @test isapprox(full.g[s],opt.g[s];atol=1e-9,rtol=1e-8)
                else
                    @test isnan(opt.f[s]) && isnan(opt.g[s])
                end
            end
        end
    end
    for bad in (Inf,NaN)
        w=initialize_reduced_fp_optimized(PerformanceInput(fp_memory_model(8,3));beta=.875,
            family=OrderedThresholdFamily(collect(1:8)))
        w.frontier_only=true;w.X[1,1]=bad;saved=deepcopy(w)
        r=pivot_reduced_fp!(w,1)
        @test !r.accepted
        for field in (:X,:phi,:psi,:f,:g,:actions,:state_at,:slot_of)
            @test isequal(getfield(w,field),getfield(saved,field))
        end
    end
end

@testset "Public checked runs retain independent acceptance checks" begin
    for criterion in (:discounted,:average),A in (1,2,3),scope in (false,true)
        input=PerformanceInput(fp_memory_model(8,A));family=OrderedThresholdFamily(collect(1:8))
        kw=criterion==:average ? (;criterion) : (;criterion,beta=.875)
        options=PerformanceOptions(fp_frontier_cache=scope)
        a=audit_performance_run(input,:FP;kw...,family,options)
        b=audit_performance_run(input,:M;kw...,family,options)
        @test a.passed && b.passed
        @test performance_agreement(a.snapshot,b.snapshot)
        @test a.snapshot.complete
    end
end
