# Synthetic correctness tests only. No timing results or experimental claims.
# Existing reference/downshift fixtures have already been included by runtests.

@testset "Analytic examples and terminal update accounting" begin
    for model in (one_state_model(),two_state_model()), criterion in (:discounted,:average)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        c = direct_downshift(model;kw...)
        r = rankone_downshift(model;kw...,audit=true)
        K = ncontrollable(model)*maxgear(model)
        @test c.complete && r.complete
        @test r.status == :complete
        @test r.policies == c.policies
        @test [s.selected for s in r.steps] == [s.selected for s in c.steps]
        @test isapprox(Float64.(r.mpi),Float64.(c.mpi);atol=1e-12,rtol=1e-11)
        @test r.initial_factorizations == 1
        @test r.rank_one_updates == K
        @test r.refactorizations == r.refactorization_attempts == 0
        @test length(r.updates) == K
        @test all(u -> u.accepted && u.method == :rank_one && u.denominator > 0,r.updates)
        @test r.audit_evaluations == K+1
        @test length(r.audits) == K+1 && all(a -> a.passed,r.audits)
        @test [a.policy_index for a in r.audits] == collect(0:K)
        @test [a.actions for a in r.audits] == r.policies
        @test r.last_evaluation.policy.actions == zeros(Int,nstates(model))
        @test c.policy_evaluations == K # C-DS omits its separate terminal solve
        ref = enumerate_bellman(model;kw...)
        check = validate_downshift(r,ref)
        @test check.execution == :exact_pass
        @test check.bellman_path == :exact_pass
        @test check.mpi_dai_agreement == :numerical_pass
        @test check.max_scaled_index_error < 1e-11
        for k in eachindex(r.steps)
            @test r.steps[k].frontier == c.steps[k].frontier
            @test isapprox(r.steps[k].frontier_f,c.steps[k].frontier_f;atol=1e-10,rtol=1e-9)
            @test isapprox(r.steps[k].frontier_g,c.steps[k].frontier_g;atol=1e-10,rtol=1e-9)
        end
        unaudited = rankone_downshift(model;kw...)
        @test unaudited.audit_evaluations == 0 && isempty(unaudited.audits)
        @test unaudited.policies == r.policies
        @test isequal(unaudited.mpi,r.mpi) # audit must not modify the algorithm state
    end
end

@testset "Kernel identities at every arbitrary downshift, independent of greedy selection" begin
    for seed in 20260916:20260921, criterion in (:discounted,:average)
        m = dyadic_validation_model(seed;n=4,A=3)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.875) :
             (;criterion=criterion,omega=[0.25,0.25,0.25,0.25])
        state = initialize_rankone(m,fill(3,4);kw...)
        @test audit_rankone(state).passed
        order = isodd(seed) ? [1,2,3,4] : [4,2,1,3]
        for a in 3:-1:1, i in order
            previous = rankone_evaluation(state)
            old_actions = copy(previous.policy.actions)
            old_values = copy(state.values)
            old_inverse = copy(state.inverse)
            old_system = copy(state.system)
            report = update_downshift!(state,i;check_inverse=true)
            @test report.accepted && report.method == :rank_one
            @test report.selected == (i,a)
            @test report.denominator > 0
            # Independent inverse solve at every pivot (test-only cubic work).
            d = size(state.system,1)
            independent_inverse = lu(state.system) \ Matrix{Float64}(I,d,d)
            @test isapprox(state.inverse,independent_inverse;atol=1e-9,rtol=1e-8)
            @test isapprox(report.denominator,det(state.system)/det(old_system);atol=1e-10,rtol=1e-9)
            @test audit_rankone(state).passed
            # Snapshot vectors/policies must not alias the mutable update.
            @test previous.policy.actions == old_actions
            if criterion == :discounted
                @test previous.F == old_values[:,1]
                @test previous.G == old_values[:,2]
            else
                @test previous.phi == old_values[1:4,1]
                @test previous.gamma == old_values[1:4,2]
            end
            @test size(state.inverse) == size(old_inverse) # no reduced tableau yet
        end
        @test all(iszero,state.policy.actions)
    end
end

@testset "Exact ties and close but unequal comparisons" begin
    for criterion in (:discounted,:average)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        m = downshift_case("all_tied")
        c = direct_downshift(m;kw...)
        r = rankone_downshift(m;kw...,audit=true)
        @test r.complete && r.policies == c.policies
        @test r.exact_comparison_solves >= 2
        @test r.steps[1].comparison_evidence == :exact_comparison
        @test all(a -> a.passed,r.audits)
        @test validate_downshift(r,enumerate_bellman(m;kw...)).mpi_dai_agreement == :numerical_pass
        stopped = rankone_downshift(m;kw...,exact_fallback=false)
        @test stopped.status == :unresolved_comparison && isempty(stopped.steps)
        @test isempty(stopped.updates) && all(ismissing,stopped.mpi)
    end
    m = close_unequal_model()
    r = rankone_downshift(m;beta=0.5,audit=true)
    @test r.complete && r.steps[1].selected == (2,1)
    @test r.steps[1].comparison_evidence == :exact_comparison
    @test validate_downshift(r,enumerate_bellman(m;beta=0.5)).execution == :exact_pass
end

@testset "Families, partial outputs and failed sufficient conditions" begin
    m = two_state_model()
    for criterion in (:discounted,:average)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        family = OrderedThresholdFamily([1,2])
        c = direct_downshift(m;kw...,family=family)
        r = rankone_downshift(m;kw...,family=family,audit=true)
        @test r.complete && r.policies == c.policies
        @test validate_downshift(r,enumerate_bellman(m;kw...)).execution == :exact_pass
    end
    disconnected = ExplicitPolicyFamily([[2,2],[0,0]])
    r = rankone_downshift(m;beta=0.5,family=disconnected,audit=true)
    @test r.status == :empty_frontier && r.initial_factorizations == 0
    @test r.audit_evaluations == 0 && r.last_evaluation === nothing
    r = rankone_downshift(m;beta=0.5,max_steps=0,audit=true)
    @test r.status == :step_limit && r.initial_factorizations == 0
    @test isempty(r.steps) && all(ismissing,r.mpi)
    r = rankone_downshift(m;beta=0.5,max_steps=1,audit=true)
    @test r.status == :step_limit && length(r.steps) == 1
    @test r.rank_one_updates == 1 && r.audit_evaluations == 2
    @test count(ismissing,r.mpi) == 3
    @test validate_downshift(r,enumerate_bellman(m;beta=0.5)).execution == :partial_path_checked
    for (m,expected,k) in ((nonpositive_marginal_model(),:no_positive_candidate,1),
                           (downshift_case("zero_denominator"),:complete,2))
        r=rankone_downshift(m;beta=.5,audit=true)
        @test r.status==expected && r.complete==(expected==:complete)
        @test length(r.steps)==length(r.updates)==r.rank_one_updates==k
        @test r.audit_evaluations==k+1 && all(a->a.passed,r.audits)
    end
    for name in ("nonconcave_one_state","negative_off_frontier","bellman_not_sign")
        m = downshift_case(name)
        c = direct_downshift(m;beta=0.75)
        r = rankone_downshift(m;beta=0.75,audit=true)
        @test r.status == c.status && r.policies == c.policies
        ref = enumerate_bellman(m;beta=0.75)
        cv,rv = validate_downshift(c,ref),validate_downshift(r,ref)
        @test rv.execution == cv.execution
        @test rv.bellman_path == cv.bellman_path
        @test rv.reference_sign_indexability == cv.reference_sign_indexability
        @test rv.pathwise_positivity == cv.pathwise_positivity
    end
end

@testset "Uncontrollable, transient, periodic and multichain policies" begin
    for m in (transient_model(),periodic_model(),downshift_case("uncontrollable"))
        for criterion in (:discounted,:average)
            kw = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
            c = direct_downshift(m;kw...)
            r = rankone_downshift(m;kw...,audit=true)
            @test r.status == c.status && r.policies == c.policies
            @test all(a -> a.passed,r.audits)
            @test all(all(p[.!m.controllable] .== 0) for p in r.policies)
        end
    end
    @test rankone_downshift(multichain_model();criterion=:average).status == :unsupported_average_policy
    @test rankone_downshift(multichain_model();criterion=:average).initial_factorizations == 0
    @test rankone_downshift(multichain_model();beta=0.5,audit=true).complete
    # A transient state can create an additional recurrent class on a downshift.
    P0 = [1.0 0.0;0.0 1.0]; P1 = [1.0 0.0;1.0 0.0]
    model = FiniteMultiGearModel(cat(P0,P1;dims=3),[2.0 1.0;0.0 1.0],[0.0 2.0;0.0 1.0])
    state = initialize_rankone(model,[1,1];criterion=:average)
    old = deepcopy(state)
    report = update_downshift!(state,2)
    @test !report.accepted && report.reason == :unsupported_average_policy
    @test state.policy.actions == old.policy.actions
    @test state.values == old.values && state.inverse == old.inverse
    r = rankone_downshift(model;criterion=:average,audit=true)
    @test r.status == :unsupported_average_policy && isempty(r.steps)
    @test length(r.updates) == 1 && !r.updates[1].accepted
end

@testset "Bias normalization and higher discount factors" begin
    m = two_state_model()
    runs = [rankone_downshift(m;criterion=:average,omega=w,audit=true)
            for w in ([1.0,0.0],[0.0,1.0],[0.25,0.75],[2.0,-1.0])]
    @test all(r -> r.complete,runs)
    for r in runs
        @test r.policies == runs[1].policies
        @test isapprox(Float64.(r.mpi),Float64.(runs[1].mpi);atol=1e-10,rtol=1e-9)
        @test all(a -> a.passed,r.audits)
        @test abs(dot(r.omega,r.last_evaluation.phi)) < 1e-10
        @test abs(dot(r.omega,r.last_evaluation.gamma)) < 1e-10
    end
    for b in (0.125,0.9,0.99,0.999)
        r = rankone_downshift(m;beta=b,audit=true)
        c = direct_downshift(m;beta=b)
        @test r.complete && r.policies == c.policies
        @test all(a -> a.passed,r.audits)
        @test isapprox(Float64.(r.mpi),Float64.(c.mpi);atol=1e-8,rtol=1e-8)
    end
end

@testset "Logged rebuilds, rejected updates and audit failure detection" begin
    m = two_state_model()
    for criterion in (:discounted,:average)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        r = rankone_downshift(m;kw...,audit=true,refactor_every=2)
        c = direct_downshift(m;kw...)
        @test r.complete && r.policies == c.policies
        @test r.initial_factorizations == 1 && r.rank_one_updates == 2
        @test r.refactorizations == r.refactorization_attempts == 2
        @test count(u -> u.reason == :scheduled,r.updates) == 2
        @test r.audit_evaluations == 5
        # Deliberately strict denominator screen forces the documented fallback.
        rebuilt = rankone_downshift(m;kw...,audit=true,pivot_atol=1e6)
        @test rebuilt.complete && rebuilt.rank_one_updates == 0
        @test rebuilt.refactorizations == 4
        @test all(u -> u.reason == :small_or_nonpositive_denominator,rebuilt.updates)
        rejected = rankone_downshift(m;kw...,pivot_atol=1e6,refactor_on_failure=false)
        @test rejected.status == :unresolved_pivot && isempty(rejected.steps)
        @test length(rejected.updates) == 1 && !rejected.updates[1].accepted
        @test rejected.refactorization_attempts == 0 && all(ismissing,rejected.mpi)
        state = initialize_rankone(m,[2,2];kw...)
        old = deepcopy(state)
        report = update_downshift!(state,1;pivot_atol=1e6,refactor_on_failure=false)
        @test !report.accepted
        @test state.inverse == old.inverse && state.values == old.values
        @test state.policy.actions == old.policy.actions
        # Deliberate corruption is ONLY a test of validation/safeguards.
        state.values[1,1] += 1.0
        @test !audit_rankone(state).passed
        report = update_downshift!(state,1;force_refactor=true)
        @test report.accepted && report.method == :refactorization
        @test audit_rankone(state).passed
        state = initialize_rankone(m,[2,2];kw...)
        state.inverse[1,1] += 0.25
        @test !audit_rankone(state).passed
        report = update_downshift!(state,1;check_inverse=true)
        @test report.accepted && report.method == :refactorization
        @test audit_rankone(state).passed
    end
    # A genuinely small positive denominator (not an artificially strict screen).
    P0 = [1.0 0.0;0.0 1.0]; P1 = [0.0 1.0;0.0 1.0]
    near = FiniteMultiGearModel(cat(P0,P1;dims=3),[2.0 1.0;5.0 4.0],[0.0 1.0;0.0 2.0])
    state = initialize_rankone(near,[1,1];beta=1.0-2.0^(-40))
    report = update_downshift!(state,1)
    @test report.accepted && report.method == :refactorization
    @test report.reason == :small_or_nonpositive_denominator
    @test report.denominator > 0
    @test audit_rankone(state).passed
    # Algebraic update identities do not require a positive marginal resource.
    state = initialize_rankone(nonpositive_marginal_model(),[1,1];beta=0.5)
    @test adjacent_marginals(rankone_evaluation(state)).g[1,1] <= 0
    @test update_downshift!(state,1;check_inverse=true).accepted
    @test audit_rankone(state).passed
end

@testset "Random dyadic C-DS comparison and job-local threading" begin
    models = [dyadic_validation_model(seed) for seed in 20260916:20260921]
    for m in models, criterion in (:discounted,:average)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.75) : (;criterion=criterion)
        c = direct_downshift(m;kw...)
        r = rankone_downshift(m;kw...,audit=true)
        @test r.status == c.status && r.policies == c.policies
        @test all(a -> a.passed,r.audits)
        for j in eachindex(r.mpi)
            @test ismissing(r.mpi[j]) == ismissing(c.mpi[j])
            if !ismissing(r.mpi[j])
                @test isapprox(r.mpi[j],c.mpi[j];atol=1e-9,rtol=1e-8)
            end
        end
        ref = enumerate_bellman(m;kw...)
        @test validate_downshift(r,ref).execution == validate_downshift(c,ref).execution
    end
    serial = [rankone_downshift(m;beta=0.75,audit=true) for m in models]
    parallel = Vector{typeof(serial[1])}(undef,length(models))
    Threads.@threads for j in eachindex(models)
        parallel[j] = rankone_downshift(models[j];beta=0.75,audit=true)
    end
    for j in eachindex(models)
        @test parallel[j].status == serial[j].status
        @test parallel[j].policies == serial[j].policies
        @test isequal(parallel[j].mpi,serial[j].mpi)
        @test all(a -> a.passed,parallel[j].audits)
    end
end

@testset "Argument validation and BigFloat smoke check" begin
    m = two_state_model()
    @test_throws ArgumentError rankone_downshift(m)
    @test_throws ArgumentError rankone_downshift(m;beta=1)
    @test_throws ArgumentError rankone_downshift(m;criterion=:average,beta=0.5)
    @test_throws ArgumentError rankone_downshift(m;beta=0.5,omega=[1,0])
    @test_throws ArgumentError rankone_downshift(m;beta=0.5,refactor_every=-1)
    @test_throws ArgumentError rankone_downshift(m;beta=0.5,pivot_rtol=-1)
    @test_throws ArgumentError rankone_downshift(m;beta=0.5,inverse_tol=0)
    @test_throws ArgumentError rankone_downshift(m;beta=0.5,audit_atol=-1)
    @test_throws ArgumentError rankone_downshift(m;beta=0.5,max_steps=5)
    state = initialize_rankone(m,[0,0];beta=0.5)
    @test_throws ArgumentError update_downshift!(state,1)
    @test_throws BoundsError update_downshift!(state,3)
    setprecision(BigFloat,128) do
        m = one_state_model(float_type=BigFloat)
        for criterion in (:discounted,:average)
            kw = criterion == :discounted ? (;criterion=criterion,beta=BigFloat(0.5)) : (;criterion=criterion)
            r = rankone_downshift(m;kw...,audit=true)
            @test r.complete && eltype(r.model.h) == BigFloat
            @test all(a -> a.passed,r.audits)
            @test Float64.(r.mpi) == [3.0 1.0]
        end
    end
end
