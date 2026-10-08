@testset "Input model and policy conventions" begin
    model = two_state_model()
    @test nstates(model) == 2
    @test maxgear(model) == 2
    @test ncontrollable(model) == 2
    @test admissible_gears(model,1) == 0:2
    p = policy_data(model,[1,2])
    @test p.P == [7/8 1/8;3/4 1/4]
    @test p.h == [1.5,3.0]
    @test p.c == [1.0,3.0]
    @test p.actions == [1,2]
    # Caller-owned input arrays and action vectors are copied.
    P, h, c = copy(model.P), copy(model.h), copy(model.c)
    copied = FiniteMultiGearModel(P,h,c)
    P[1,1,1], h[1,1], c[1,1] = -7.0, -7.0, -7.0
    @test copied.P == model.P
    @test copied.h == model.h
    @test copied.c == model.c
    actions = [0,1]
    pdata = policy_data(model,actions)
    actions[1] = 2
    @test pdata.actions == [0,1]
    @test_throws DimensionMismatch policy_data(model,[0])
    @test_throws ArgumentError policy_data(model,[-1,0])
    @test_throws ArgumentError policy_data(model,[0,3])
    @test_throws BoundsError admissible_gears(model,0)
    @test_throws DimensionMismatch FiniteMultiGearModel(model.P,model.h,zeros(2,2))
    @test_throws ArgumentError FiniteMultiGearModel(model.P,model.h,model.c;
                                                   controllable=falses(2))
    @test_throws DimensionMismatch FiniteMultiGearModel(model.P,model.h,model.c;
                                                       controllable=trues(1))
    for bad in (-1.0,NaN,Inf)
        pp = copy(model.P); pp[1,1,1] = bad
        @test_throws ArgumentError FiniteMultiGearModel(pp,model.h,model.c)
    end
    pp = copy(model.P); pp[1,1,1] += 0.01
    @test_throws ArgumentError FiniteMultiGearModel(pp,model.h,model.c)
    cc = copy(model.c); cc[1,2] = cc[1,1]
    @test_throws ArgumentError FiniteMultiGearModel(model.P,model.h,cc)
    cc = copy(model.c); cc[1,1] = -0.01
    @test_throws ArgumentError FiniteMultiGearModel(model.P,model.h,cc)
    hh = copy(model.h); hh[1,1] = NaN
    @test_throws ArgumentError FiniteMultiGearModel(model.P,hh,model.c)
    @test_throws ArgumentError FiniteMultiGearModel(model.P,model.h,model.c;row_atol=-1)
    # Valid small row-sum discrepancies are retained, not normalized.
    pp = copy(model.P); pp[1,1,1] += 1e-13
    unaltered = FiniteMultiGearModel(pp,model.h,model.c)
    @test unaltered.P[1,:,1] == pp[1,:,1]
    @test sum(unaltered.P[1,:,1]) != 1.0
    tm = transient_model()
    @test admissible_gears(tm,1) == 0:0
    @test tm.h[1,2] == tm.h[1,1] == 7.0
    @test tm.c[1,2] == tm.c[1,1] == 0.0
    @test_throws ArgumentError policy_data(tm,[1,0,0])
end

@testset "Analytic one-state discounted and average values" begin
    model = one_state_model()
    for a in 0:2
        d = evaluate_discounted(model,[a];beta=0.5)
        av = evaluate_average(model,[a])
        @test d.F == 2 .* [model.h[1,a+1]]
        @test d.G == 2 .* [model.c[1,a+1]]
        @test av.hbar == model.h[1,a+1]
        @test av.cbar == model.c[1,a+1]
        @test av.phi == av.gamma == [0.0]
        @test av.recurrent_states == [1]
        for e in (d,av)
            m = adjacent_marginals(e)
            @test m.states == [1]
            @test m.f == reshape([3.0,2.0],1,2)
            @test m.g == reshape([1.0,2.0],1,2)
            @test m.m == reshape([3.0,1.0],1,2)
            @test maximum(e.diagnostics.backward_error) <= 1e-14
        end
    end
    # Admissible zero resource at the lowest gear gives a zero RHS, not NaN.
    d = evaluate_discounted(model,[0];beta=0.5)
    @test d.diagnostics.residual_inf[2] == 0
    @test d.diagnostics.backward_error[2] == 0
    for beta in (-0.1,0.0,1.0,1.1,NaN,Inf)
        @test_throws ArgumentError evaluate_discounted(model,[0];beta=beta)
    end
    @test_throws ArgumentError charged_value(d,Inf)
    @test_throws ArgumentError adjacent_marginals(d;positivity_tol=-1)
end

@testset "Two-state analytic policy and charged equations" begin
    model, actions = two_state_model(), [1,2]
    # Exact values derived independently from rational two-by-two equations.
    d = evaluate_discounted(model,actions;beta=0.5)
    @test d.F ≈ [16/5,24/5] atol=1e-13 rtol=1e-13
    @test d.G ≈ [34/15,22/5] atol=1e-13 rtol=1e-13
    av = evaluate_average(model,actions)
    @test av.hbar ≈ 12/7 atol=1e-13
    @test av.cbar ≈ 9/7 atol=1e-13
    @test av.phi ≈ [0.0,12/7] atol=1e-13
    @test av.gamma ≈ [0.0,16/7] atol=1e-13
    nu = [6/7,1/7]
    @test transpose(av.policy.P)*nu ≈ nu atol=1e-13
    @test dot(nu,av.policy.h) ≈ av.hbar atol=1e-13
    @test dot(nu,av.policy.c) ≈ av.cbar atol=1e-13
    for lambda in (-2.0,0.0,0.75,4.0)
        v = charged_value(d,lambda)
        b, rate = charged_bias(av,lambda), charged_rate(av,lambda)
        @test v ≈ d.policy.h + lambda*d.policy.c + d.beta*d.policy.P*v atol=1e-12
        @test b .+ rate ≈ av.policy.h + lambda*av.policy.c + av.policy.P*b atol=1e-12
        for e in (d,av)
            Q, m = policy_qvalues(e,lambda), adjacent_marginals(e)
            for i in 1:nstates(model)
                expected = e isa DiscountedEvaluation ? v[i] : rate+b[i]
                @test Q[i,actions[i]+1] ≈ expected atol=1e-12
            end
            for (r,i) in enumerate(m.states), a in 1:maxgear(model)
                @test Q[i,a]-Q[i,a+1] ≈ m.f[r,a]-lambda*m.g[r,a] atol=1e-12
            end
        end
    end
    for omega in ([0.0,1.0],[0.25,0.75],[-1.0,2.0])
        alt = evaluate_average(model,actions;omega=omega)
        @test alt.hbar ≈ av.hbar atol=1e-12
        @test alt.cbar ≈ av.cbar atol=1e-12
        @test alt.phi ≈ av.phi .- dot(omega,av.phi) atol=1e-12
        @test alt.gamma ≈ av.gamma .- dot(omega,av.gamma) atol=1e-12
        @test dot(omega,alt.phi) ≈ 0.0 atol=1e-12
        @test dot(omega,alt.gamma) ≈ 0.0 atol=1e-12
        @test adjacent_marginals(alt).f ≈ adjacent_marginals(av).f atol=1e-12
        @test adjacent_marginals(alt).g ≈ adjacent_marginals(av).g atol=1e-12
    end
    @test_throws ArgumentError evaluate_average(model,actions;omega=[0.0,0.0])
    @test_throws DimensionMismatch evaluate_average(model,actions;omega=[1.0])
    @test_throws ArgumentError evaluate_average(model,actions;omega=[NaN,0.0])
end

@testset "Transient, periodic, and multichain policies" begin
    tm = transient_model()
    e = evaluate_average(tm,[0,1,0])
    @test e.recurrent_states == [2,3]
    @test e.hbar ≈ 3.0 atol=1e-13
    @test e.cbar ≈ 1/3 atol=1e-13
    @test e.phi ≈ [0.0,-4.0,0.0] atol=1e-13
    @test e.gamma ≈ [0.0,1/3,-1.0] atol=1e-13
    @test ismissing(policy_qvalues(e,0.5)[1,2])
    @test adjacent_marginals(e).states == [2,3]
    @test size(adjacent_marginals(e).f) == (2,1)
    d = evaluate_discounted(tm,[0,1,0];beta=0.9)
    @test ismissing(policy_qvalues(d,0.5)[1,2])
    per = evaluate_average(periodic_model(),[1,0])
    @test per.hbar ≈ 3.0 atol=1e-13
    @test per.cbar ≈ 0.5 atol=1e-13
    @test per.phi ≈ [0.0,2.0] atol=1e-13
    @test per.gamma ≈ [0.0,-0.5] atol=1e-13
    @test per.recurrent_states == [1,2]
    mc = multichain_model()
    @test_throws ArgumentError evaluate_average(mc,[1,0])
    @test evaluate_discounted(mc,[1,0];beta=0.5).F == [2.0,10.0]
end

@testset "Zero and negative marginal resource are not hidden" begin
    model = nonpositive_marginal_model()
    zero_g = adjacent_marginals(evaluate_discounted(model,[0,0];beta=0.5))
    neg_g = adjacent_marginals(evaluate_discounted(model,[0,0];beta=0.75))
    avg_g = adjacent_marginals(evaluate_average(model,[0,0]))
    @test zero_g.g[1,1] == 0.0
    @test ismissing(zero_g.m[1,1])
    @test neg_g.g[1,1] == -0.5
    @test ismissing(neg_g.m[1,1])
    @test avg_g.g[1,1] == -1.0
    @test ismissing(avg_g.m[1,1])
    m = adjacent_marginals(evaluate_discounted(one_state_model(),[0];beta=0.5);
                           positivity_tol=1.5)
    @test ismissing(m.m[1,1])
    @test m.m[1,2] == 1.0
end

# An independent graph oracle: Boolean transitive closure, not SCC DFS.
function oracle_closed_classes(P)
    n = size(P,1)
    reach = P .> 0
    for i in 1:n; reach[i,i] = true; end
    for k in 1:n, i in 1:n, j in 1:n
        reach[i,j] = reach[i,j] || (reach[i,k] && reach[k,j])
    end
    groups = Set{Tuple}()
    for i in 1:n
        group = [j for j in 1:n if reach[i,j] && reach[j,i]]
        if all(P[u,v] == 0 for u in group for v in setdiff(1:n,group))
            push!(groups,Tuple(group))
        end
    end
    return groups
end

@testset "Closed-class detector against independent graph oracle" begin
    # All directed support patterns with three nonempty rows: 7^3 patterns.
    for masks in Iterators.product(1:7,1:7,1:7)
        P = zeros(3,3)
        for i in 1:3
            for j in 1:3; P[i,j] = (masks[i] >> (j-1)) & 1; end
            P[i,:] ./= sum(@view P[i,:])
        end
        got = Set(Tuple.(MultiGearBandits._closed_classes(P)))
        @test got == oracle_closed_classes(P)
    end
    delta = 2.0^-40
    P = [1-delta delta;delta 1-delta]
    @test MultiGearBandits._closed_classes(P) == [[1,2]]
end

@testset "Random small models, independent solves, and Bellman identities" begin
    for seed in 1001:1012
        model, actions = random_test_model(seed;n=2+mod(seed,4),A=1+mod(seed,3))
        d = evaluate_discounted(model,actions;beta=0.8)
        av = evaluate_average(model,actions)
        n = nstates(model)
        # QR is independent of the LU solve used in the evaluator.
        M = Matrix{Float64}(I,n,n)-0.8*d.policy.P
        expected = qr(M) \ hcat(d.policy.h,d.policy.c)
        @test d.F ≈ expected[:,1] atol=1e-11 rtol=1e-11
        @test d.G ≈ expected[:,2] atol=1e-11 rtol=1e-11
        # A separate stationary system independently checks the average rates.
        K = Matrix{Float64}(I,n,n)-transpose(av.policy.P)
        K[end,:] .= 1.0
        rhs = zeros(n); rhs[end] = 1.0
        nu = qr(K) \ rhs
        @test minimum(nu) >= -1e-12
        @test transpose(av.policy.P)*nu ≈ nu atol=1e-12
        @test dot(nu,av.policy.h) ≈ av.hbar atol=1e-11
        @test dot(nu,av.policy.c) ≈ av.cbar atol=1e-11
        @test maximum(d.diagnostics.backward_error) <= 1e-12
        @test maximum(av.diagnostics.backward_error) <= 1e-12
        for e in (d,av), lambda in (-1.0,0.0,2.0)
            Q, marg = policy_qvalues(e,lambda), adjacent_marginals(e)
            for (r,i) in enumerate(marg.states), a in 1:maxgear(model)
                @test Q[i,a]-Q[i,a+1] ≈ marg.f[r,a]-lambda*marg.g[r,a] atol=1e-10 rtol=1e-10
            end
        end
        # Independent finite-horizon discounted summation from zero terminal value.
        partial = zeros(n,2)
        for _ in 1:180
            partial = hcat(d.policy.h,d.policy.c) + 0.8*d.policy.P*partial
        end
        @test partial[:,1] ≈ d.F atol=1e-10 rtol=1e-10
        @test partial[:,2] ≈ d.G atol=1e-10 rtol=1e-10
    end
end

@testset "High precision and near-one discount" begin
    # High precision is deliberately confined to this serial test.
    setprecision(BigFloat,256) do
        model = two_state_model(;float_type=BigFloat,row_atol=big"1e-60")
        d = evaluate_discounted(model,[1,2];beta=big"0.5")
        av = evaluate_average(model,[1,2])
        @test eltype(d.F) == BigFloat
        @test d.F ≈ BigFloat[16//5,24//5] atol=big"1e-65" rtol=big"1e-65"
        @test av.hbar ≈ BigFloat(12//7) atol=big"1e-65"
        @test av.cbar ≈ BigFloat(9//7) atol=big"1e-65"
        @test maximum(av.diagnostics.backward_error) < big"1e-65"
    end
    beta = 1.0-2.0^-30
    d = evaluate_discounted(one_state_model(),[1];beta=beta)
    @test (1-beta)*d.F[1] == 3.0
    @test (1-beta)*d.G[1] == 1.0
end

function independent_reference_job(seed)
    model, actions = random_test_model(seed)
    d = evaluate_discounted(model,actions;beta=0.9)
    a = evaluate_average(model,actions)
    return vcat(d.F,d.G,a.phi,a.gamma,[a.hbar,a.cbar])
end

@testset "Independent job agreement with Julia threading" begin
    seeds = collect(2001:2024)
    serial = [independent_reference_job(seed) for seed in seeds]
    parallel = Vector{Vector{Float64}}(undef,length(seeds))
    Threads.@threads for k in eachindex(seeds)
        parallel[k] = independent_reference_job(seeds[k])
    end
    # Each job owns its model, RNG, evaluation arrays and output slot.
    @test serial == parallel
end
