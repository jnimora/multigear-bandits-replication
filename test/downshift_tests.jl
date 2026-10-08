@testset "One-state analytic MPI and DAI" begin
    for criterion in (:discounted,:average)
        m = one_state_model()
        kwargs = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        run = direct_downshift(m;kwargs...)
        ref = enumerate_bellman(m;kwargs...)
        check = validate_downshift(run,ref)
        @test run.complete
        @test [s.selected for s in run.steps] == [(1,2),(1,1)]
        @test Float64.(run.mpi) == [3.0 1.0]
        @test run.policy_evaluations == 2
        @test length(run.policies) == 3
        @test ref.complete && length(ref.records) == 3
        @test ref.coverage == :exact_pass
        @test ref.sign_indexability == :exact_pass
        @test ref.dai == [3//1 1//1]
        @test Set(Tuple.(optimal_policies(ref,1))) == Set([(1,),(2,)])
        @test Set(Tuple.(optimal_policies(ref,3))) == Set([(0,),(1,)])
        @test optimal_policies(ref,-10) == [[2]]
        @test optimal_policies(ref,10) == [[0]]
        @test check.execution == :exact_pass
        @test check.stopping == :exact_pass
        @test check.monotonicity == :exact_pass
        @test check.frontier_positivity == :exact_pass
        @test check.pathwise_positivity == :exact_pass
        @test check.family_wide_positivity == :exact_pass
        @test check.bellman_path == :exact_pass
        @test check.mpi_dai_agreement == :numerical_pass
        @test check.max_scaled_index_error == 0.0
    end
end

@testset "Existing two-state example, both criteria" begin
    m = two_state_model()
    expected = Dict(:discounted => [9//13 33//56;1//1 1//1],
                    :average => [1//1 17//24;5//3 5//3])
    for criterion in (:discounted,:average)
        kwargs = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        run = direct_downshift(m;kwargs...)
        ref = enumerate_bellman(m;kwargs...)
        check = validate_downshift(run,ref)
        @test run.complete && run.status == :complete
        @test length(run.steps) == 4 && length(run.policies) == 5
        @test run.policy_evaluations == 4 # terminal solve is NOT part of selection
        @test [s.selected for s in run.steps] == [(1,2),(1,1),(2,2),(2,1)]
        @test ref.total_policies == 9 && length(ref.records) == 9
        @test ref.coverage == :exact_pass && ref.sign_indexability == :exact_pass
        @test ref.dai == expected[criterion]
        @test Float64.(run.mpi) ≈ Float64.(expected[criterion])
        @test check.execution == :exact_pass && check.stopping == :exact_pass
        @test check.monotonicity == :exact_pass
        @test check.bellman_path == :exact_pass
        @test check.mpi_dai_agreement == :numerical_pass
        @test check.max_scaled_index_error < 1e-12
        # A threshold restriction that contains this realized path preserves it.
        threshold = direct_downshift(m;kwargs...,family=OrderedThresholdFamily([1,2]))
        @test threshold.policies == run.policies
        @test validate_downshift(threshold,ref).bellman_path == :exact_pass
        # Corrupting a numerical output must not be accepted just because the
        # recorded policy sequence is correct.
        corrupted = deepcopy(run)
        corrupted.mpi[1,1] = Float64(corrupted.mpi[1,1])+0.1
        bad = validate_downshift(corrupted,ref)
        @test bad.numeric_assignments == :numerical_violation
        @test bad.mpi_dai_agreement == :violation
    end
end

@testset "Exact ties, singleton intervals and close unequal ratios" begin
    m = downshift_case("all_tied")
    for criterion in (:discounted,:average)
        kwargs = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        run = direct_downshift(m;kwargs...)
        ref = enumerate_bellman(m;kwargs...)
        @test run.complete
        @test run.exact_comparison_solves >= 2
        @test [s.selected for s in run.steps] == [(1,2),(1,1),(2,2),(2,1)]
        @test run.steps[1].comparison_evidence == :exact_comparison
        @test ref.sign_indexability == :exact_pass && ref.dai == fill(2//1,2,2)
        @test length(optimal_policies(ref,2)) == 9
        @test count(r -> r.interval.lower == 2 && r.interval.upper == 2,ref.records) == 7
        @test validate_downshift(run,ref).mpi_dai_agreement == :numerical_pass
        stopped = direct_downshift(m;kwargs...,exact_fallback=false)
        @test !stopped.complete && stopped.status == :unresolved_comparison
        @test isempty(stopped.steps) && all(ismissing,stopped.mpi)
    end
    near = close_unequal_model()
    run = direct_downshift(near;beta=0.5)
    @test run.complete
    @test run.steps[1].selected == (2,1) # NOT a tolerance-based tie with state 1
    @test run.steps[1].comparison_evidence == :exact_comparison
    ref = enumerate_bellman(near;beta=0.5)
    @test ref.dai[2,1] < ref.dai[1,1]
    @test validate_downshift(run,ref).execution == :exact_pass
end

@testset "Monotonicity failure stops before an invalid assignment" begin
    m = downshift_case("nonconcave_one_state")
    for criterion in (:discounted,:average)
        kwargs = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        run = direct_downshift(m;kwargs...)
        ref = enumerate_bellman(m;kwargs...)
        check = validate_downshift(run,ref)
        @test !run.complete && run.status==:nonmonotone
        @test [s.assignment for s in run.steps] == [3.0]
        @test ismissing(run.mpi[1,1]) && run.policies[end]==[1]
        @test ref.coverage == :exact_pass
        @test ref.sign_indexability == :exact_violation
        @test ref.witness.reason == :unordered_critical_charges
        @test ref.dai === nothing
        @test check.execution == :partial_path_checked
        @test check.stopping == :exact_pass
        @test check.monotonicity == :partial_path_checked
        @test check.family_wide_positivity == :exact_pass
        @test check.mpi_dai_agreement == :not_applicable
        @test check.max_scaled_index_error === nothing
        @test Set(Tuple.(optimal_policies(ref,2))) == Set([(0,),(2,)])
        @test only(r for r in ref.records if r.actions == [1]).interval.empty
    end
end

@testset "Positive-work filtering and partial outputs" begin
    m=nonpositive_marginal_model()
    run=direct_downshift(m;beta=0.5)
    @test !run.complete && run.status==:no_positive_candidate
    @test length(run.steps)==1 && run.steps[1].selected==(2,1)
    @test run.policies==[[1,1],[1,0]] && ismissing(run.mpi[1,1])
    @test run.policy_evaluations==2
    check=validate_downshift(run,enumerate_bellman(m;beta=0.5))
    @test check.execution==:partial_path_checked && check.stopping==:exact_pass
    @test check.frontier_positivity==:exact_violation
    zero=direct_downshift(downshift_case("zero_denominator");beta=0.5)
    @test zero.complete && zero.mpi[:,1]==[2,1]
    @test zero.exact_comparison_solves==1
    m = two_state_model()
    run = direct_downshift(m;beta=0.5,max_steps=1)
    ref = enumerate_bellman(m;beta=0.5)
    check = validate_downshift(run,ref)
    @test !run.complete && run.status == :step_limit
    @test count(ismissing,run.mpi) == 3
    @test check.execution == :partial_path_checked && check.stopping == :requested_limit
    @test check.mpi_dai_agreement == :unresolved
    @test check.max_scaled_index_error === nothing
end

@testset "A failed sufficient positivity condition can coexist with a valid DAI" begin
    m = downshift_case("negative_off_frontier")
    run = direct_downshift(m;beta=0.75)
    ref = enumerate_bellman(m;beta=0.75)
    check = validate_downshift(run,ref)
    @test run.complete
    @test ref.sign_indexability == :exact_pass
    @test ref.dai == [11//2 -220//137;20//29 -389//5]
    @test check.execution == :exact_pass && check.monotonicity == :exact_pass
    @test check.frontier_positivity == :exact_pass
    @test check.pathwise_positivity == :exact_violation
    @test check.family_wide_positivity == :exact_violation
    @test check.bellman_path == :exact_pass
    @test check.mpi_dai_agreement == :numerical_pass
    path_records = [r for r in ref.records if r.actions in run.policies]
    @test minimum(-v for r in path_records for v in r.gap_slope) == -1//23
end

@testset "Bellman-optimal path coverage is not sign-indexability" begin
    m = downshift_case("bellman_not_sign")
    run = direct_downshift(m;beta=0.75)
    ref = enumerate_bellman(m;beta=0.75)
    check = validate_downshift(run,ref)
    @test run.complete
    @test check.execution == :exact_pass && check.monotonicity == :exact_pass
    @test check.bellman_path == :exact_pass
    @test ref.coverage == :exact_pass
    @test ref.sign_indexability == :exact_violation && ref.dai === nothing
    @test ref.witness.reason == :nonunique_finite_crossing
    @test ref.witness.pair == (2,1)
    @test ref.witness.roots == [-53//5,55//41]
    @test check.mpi_dai_agreement == :not_applicable
    @test check.max_scaled_index_error === nothing
end

@testset "Family feasibility and uncontrollable states" begin
    binary = FiniteMultiGearModel(cat([0.5 0.5;0.5 0.5],[0.5 0.5;0.5 0.5];dims=3),
                                 [3.0 1.0;4.0 1.0],[0.0 1.0;0.0 1.0])
    disconnected = ExplicitPolicyFamily([[1,1],[0,0]])
    stopped = direct_downshift(binary;beta=0.5,family=disconnected)
    @test stopped.status == :empty_frontier && stopped.policy_evaluations == 0
    @test validate_downshift(stopped,enumerate_bellman(binary;beta=0.5)).stopping == :exact_pass
    @test_throws ArgumentError direct_downshift(binary;beta=0.5,family=ExplicitPolicyFamily([[1,1]]))
    @test_throws ArgumentError direct_downshift(binary;beta=0.5,family=OrderedThresholdFamily([1]))
    @test_throws ArgumentError OrderedThresholdFamily([1,1])
    @test downshift_frontier(two_state_model(),[2,2],OrderedThresholdFamily([1,2])) == [(1,2)]
    @test downshift_frontier(two_state_model(),[1,2],OrderedThresholdFamily([1,2])) == [(1,1),(2,2)]
    m = downshift_case("uncontrollable")
    for criterion in (:discounted,:average)
        kwargs = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        run = direct_downshift(m;kwargs...)
        ref = enumerate_bellman(m;kwargs...)
        @test run.states == [1,2] && size(run.mpi) == (2,2)
        @test length(run.steps) == 4 && ref.total_policies == 9
        @test all(p[3] == 0 for p in run.policies)
        @test ref.dai == [2//1 2//1;3//1 1//1]
        @test validate_downshift(run,ref).mpi_dai_agreement == :numerical_pass
    end
end

@testset "Average rates alone do not verify Bellman optimality" begin
    ref = enumerate_bellman(downshift_case("transient_bellman");criterion=:average)
    @test ref.complete && ref.coverage == :exact_pass
    bad = only(r for r in ref.records if r.actions == [0,0])
    good = only(r for r in ref.records if r.actions == [1,0])
    @test bad.cost_rate == good.cost_rate == 0
    @test !(0 in bad.interval) && 0 in good.interval
    @test bad.residual_intercept[1,2] < 0
    @test !([0,0] in optimal_policies(ref,0))
    ref2 = enumerate_bellman(two_state_model();criterion=:average,omega=[0,1])
    ref1 = enumerate_bellman(two_state_model();criterion=:average)
    @test ref2.dai == ref1.dai
    @test [r.interval for r in ref2.records] == [r.interval for r in ref1.records]
    run = direct_downshift(two_state_model();criterion=:average,omega=[0,1])
    @test validate_downshift(run,ref2).mpi_dai_agreement == :numerical_pass
    @test_throws ArgumentError validate_downshift(run,ref1)
    unsupported = enumerate_bellman(multichain_model();criterion=:average)
    @test !unsupported.complete && unsupported.status == :unsupported_average_model
    @test unsupported.dai === nothing
    @test direct_downshift(multichain_model();criterion=:average).status == :unsupported_average_policy
end

@testset "Exact primitive meaning, limits and invalid inputs" begin
    # Integer/rational inputs retain their exact meaning, including thirds.
    P0 = fill(1//3,3,3)
    raw = ExactMultiGearModel(cat(P0,P0;dims=3),[3 1;3 1;3 1],[0 1;0 1;0 1])
    ref = enumerate_bellman(raw;beta=9//10)
    @test ref.beta == 9//10
    @test ref.sign_indexability == :exact_pass && ref.dai == fill(2//1,3,1)
    approximate = FiniteMultiGearModel(Float64.(raw.P),Float64.(raw.h),Float64.(raw.c))
    @test_throws ArgumentError ExactMultiGearModel(approximate)
    @test_throws ArgumentError enumerate_bellman(approximate;beta=0.9)
    stopped = direct_downshift(approximate;beta=0.5)
    @test !stopped.complete && stopped.status == :unresolved_comparison
    m = two_state_model()
    limited = enumerate_bellman(m;beta=0.5,max_policies=2)
    @test !limited.complete && limited.status == :policy_limit
    @test limited.dai === nothing && limited.sign_indexability == :unresolved
    @test validate_downshift(direct_downshift(m;beta=0.5),limited).execution == :unresolved
    @test_throws ArgumentError optimal_policies(limited,0)
    @test enumerate_bellman(m;beta=0.5,time_limit_s=0).status == :time_limit
    @test enumerate_bellman(m;beta=0.5,max_states=1).status == :state_limit
    @test_throws ArgumentError direct_downshift(m)
    @test_throws ArgumentError direct_downshift(m;beta=1)
    @test_throws ArgumentError direct_downshift(m;criterion=:average,beta=0.9)
    @test_throws ArgumentError direct_downshift(m;beta=0.5,omega=[1,0])
    @test_throws ArgumentError direct_downshift(m;beta=0.5,screen_atol=-1)
    @test_throws ArgumentError direct_downshift(m;beta=0.5,max_steps=5)
    @test_throws ArgumentError enumerate_bellman(m;beta=0.5,max_policies=0)
    @test_throws ArgumentError validate_downshift(direct_downshift(m;beta=0.5),enumerate_bellman(m;beta=0.75))
end

@testset "Independent exact checks on dyadic models and threaded execution" begin
    models = [dyadic_validation_model(seed) for seed in 20260915:20260920]
    serial = [direct_downshift(m;beta=0.75) for m in models]
    parallel = Vector{typeof(serial[1])}(undef,length(models))
    Threads.@threads for j in eachindex(models)
        parallel[j] = direct_downshift(models[j];beta=0.75)
    end
    for j in eachindex(models)
        @test parallel[j].status == serial[j].status
        @test parallel[j].policies == serial[j].policies
        @test isequal(parallel[j].mpi,serial[j].mpi)
        for criterion in (:discounted,:average)
            kwargs = criterion == :discounted ? (;criterion=criterion,beta=0.75) : (;criterion=criterion)
            run = criterion == :discounted ? serial[j] : direct_downshift(models[j];kwargs...)
            ref = enumerate_bellman(models[j];kwargs...)
            @test ref.complete && ref.coverage == :exact_pass
            check = validate_downshift(run,ref)
            if run.complete
                @test check.execution == :exact_pass
                @test check.numeric_assignments == :numerical_pass
                @test check.stopping == :exact_pass
            elseif run.status in (:no_positive_candidate,:nonmonotone)
                @test check.stopping == :exact_pass
            else
                @test run.status == :unresolved_comparison
            end
            @test (ref.dai !== nothing) == (ref.sign_indexability == :exact_pass)
        end
    end
end
