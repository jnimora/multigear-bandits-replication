# Synthetic correctness tests. No timings or production evidence.
# Uses the preceding reference and downshift fixtures, already included.

function cached_reentry_fixture()
    n = 4
    P0 = fill(0.25,n,n)
    P1 = copy(P0)
    for i in 1:n
        P1[i,i] += 0.125
        P1[i,mod1(i+1,n)] -= 0.125
    end
    model = FiniteMultiGearModel(cat(P0,P1;dims=3),
        hcat([1.0,3.0,2.0,4.0],zeros(n)),hcat(zeros(n),ones(n)))
    # (2,1) is feasible initially, disappears after shifting state 1,
    # and re-enters after shifting state 3. Never use its stale old metrics.
    family = ExplicitPolicyFamily([[1,1,1,1],[0,1,1,1],[1,0,1,1],
                                  [0,1,0,1],[0,0,0,1],[0,0,0,0]])
    return model,family
end

@testset "Analytic C-DS/R-DS/M-DS agreement and cache work" begin
    for model in (one_state_model(),two_state_model()), criterion in (:discounted,:average)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        c = direct_downshift(model;kw...)
        r = rankone_downshift(model;kw...)
        m = cached_downshift(model;kw...,audit=true)
        K = ncontrollable(model)*maxgear(model)
        @test c.complete && r.complete && m.complete
        @test m.policies == r.policies == c.policies
        @test [s.selected for s in m.steps] == [s.selected for s in c.steps]
        @test isapprox(Float64.(m.mpi),Float64.(c.mpi);atol=1e-11,rtol=1e-10)
        @test m.initial_factorizations == 1 && m.rank_one_updates == K
        @test m.refactorizations == m.refactorization_attempts == 0
        @test m.cache_exposures == m.cache_direct_evaluations == K
        @test m.cache_rebuild_evaluations == m.cache_rebuilds == 0
        @test m.cache_pivot_preparations == K
        total_frontier = sum(length(s.frontier) for s in m.steps)
        @test m.cache_direct_evaluations+m.cache_recurrence_refreshes == total_frontier
        @test length(m.cache_records) == length(m.policies) == K+1
        @test m.audit_evaluations == length(m.audits) == K+1
        @test all(a -> a.passed,m.audits)
        @test all(a -> a.policy.passed && a.cache_structure_valid && a.screening_scale_valid,m.audits)
        @test isempty(m.last_cache.frontier) && isempty(m.last_cache.f) && isempty(m.last_cache.g)
        @test all(iszero,m.last_cache.actions)
        @test all(iszero,m.last_evaluation.policy.actions)
        @test m.audits[end].cache_f_error == m.audits[end].cache_g_error == 0
        for k in eachindex(m.steps)
            @test m.steps[k].frontier == c.steps[k].frontier
            @test isapprox(m.steps[k].frontier_f,c.steps[k].frontier_f;atol=1e-10,rtol=1e-9)
            @test isapprox(m.steps[k].frontier_g,c.steps[k].frontier_g;atol=1e-10,rtol=1e-9)
        end
        for k in eachindex(m.cache_records)
            record = m.cache_records[k]
            @test record.policy_index == k-1
            @test record.frontier == downshift_frontier(model,m.policies[k],m.family)
            old = k == 1 ? Tuple{Int,Int}[] : m.cache_records[k-1].frontier
            @test Set(record.exposed) == setdiff(Set(record.frontier),Set(old))
            @test Set(record.retained) == intersect(Set(record.frontier),Set(old))
            @test Set(record.discarded) == setdiff(Set(old),Set(record.frontier))
        end
        ref = enumerate_bellman(model;kw...)
        check = validate_downshift(m,ref)
        @test check.execution == :exact_pass
        @test check.bellman_path == :exact_pass
        @test check.mpi_dai_agreement == :numerical_pass
        plain = cached_downshift(model;kw...)
        @test plain.audit_evaluations == 0 && isempty(plain.audits)
        @test plain.policies == m.policies && isequal(plain.mpi,m.mpi)
        @test plain.cache_direct_evaluations == m.cache_direct_evaluations
    end
    for criterion in (:discounted,:average)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        m = cached_downshift(two_state_model();kw...)
        @test m.cache_direct_evaluations == 4
        @test m.cache_recurrence_refreshes == 2
        @test sum(length(s.frontier) for s in m.steps) == 6
    end
end

@testset "Cache identities under arbitrary downshifts" begin
    for seed in 20260916:20260921, criterion in (:discounted,:average)
        model = dyadic_validation_model(seed;n=4,A=3)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.875) :
             (;criterion=criterion,omega=[0.25,0.25,0.25,0.25])
        state = initialize_rankone(model,fill(3,4);kw...)
        cache,record = initialize_marginal_cache(rankone_evaluation(state))
        @test record.direct_evaluations == 4
        @test audit_marginal_cache(state,cache).passed
        sequence = isodd(seed) ? [1,2,3,4] : [4,2,1,3]
        k = 0
        for a in 3:-1:1, i in sequence
            k += 1
            old_f,old_g,old_actions = copy(cache.f),copy(cache.g),copy(cache.actions)
            old_rows = deepcopy(cache.rows)
            pivot = prepare_marginal_pivot(state,i)
            @test pivot.selected == (i,a)
            @test length(pivot.column) == (criterion == :discounted ? 4 : 5)
            update = update_downshift!(state,i;check_inverse=true)
            @test update.accepted && update.method == :rank_one
            @test pivot.denominator == update.denominator
            next,record = refresh_marginal_cache(cache,pivot,rankone_evaluation(state);policy_index=k)
            @test audit_marginal_cache(state,next;policy_index=k).passed
            @test cache.actions == old_actions && cache.f == old_f && cache.g == old_g
            @test cache.rows == old_rows
            @test !((i,a) in next.frontier)
            @test a == 1 || (i,a-1) in record.exposed
            @test record.direct_evaluations == length(record.exposed)
            @test record.recurrence_refreshes == length(record.retained)
            @test Set(record.retained) == intersect(Set(cache.frontier),Set(next.frontier))
            if !isempty(next.frontier)
                e = rankone_evaluation(state)
                _,_,bound = MultiGearBandits._cached_frontier_metrics(next,e)
                _,_,direct_scale = MultiGearBandits._frontier_metrics(e,next.frontier)
                @test all(bound .+ 1e-10 .>= direct_scale)
            end
            cache = next
        end
        @test isempty(cache.frontier) && all(iszero,cache.actions)
    end
end

@testset "Restricted frontiers: removal and re-entry" begin
    for criterion in (:discounted,:average)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        model,family = cached_reentry_fixture()
        c = direct_downshift(model;kw...,family=family)
        m = cached_downshift(model;kw...,family=family,audit=true)
        @test c.complete && m.complete && m.policies == c.policies
        @test [s.selected for s in m.steps] == [(1,1),(3,1),(2,1),(4,1)]
        @test m.cache_exposures == m.cache_direct_evaluations == 5 # K+one re-entry
        @test m.cache_recurrence_refreshes == 0
        @test (2,1) in m.cache_records[1].exposed
        @test (2,1) in m.cache_records[2].discarded
        @test (2,1) in m.cache_records[3].exposed
        @test all(a -> a.passed,m.audits)
        @test isapprox(Float64.(m.mpi),Float64.(c.mpi);atol=1e-10,rtol=1e-9)
        old_index = findfirst(==((2,1)),m.steps[1].frontier)
        @test !isapprox(m.steps[1].frontier_f[old_index],m.steps[3].frontier_f[1];atol=1e-8,rtol=1e-8)
        # Ordered thresholds use another frontier mechanism.
        tm = two_state_model()
        tf = OrderedThresholdFamily([1,2])
        tc = direct_downshift(tm;kw...,family=tf)
        mc = cached_downshift(tm;kw...,family=tf,audit=true)
        @test mc.complete && mc.policies == tc.policies
        @test all(a -> a.passed,mc.audits)
    end
end

@testset "Cache rebuilding and accounting" begin
    model = two_state_model()
    for criterion in (:discounted,:average)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        base = cached_downshift(model;kw...,audit=true)
        for options in ((;refactor_every=1),(;refactor_every=2),
                        (;pivot_atol=1e6),(;cache_rebuild_every=1),(;cache_rebuild_every=2))
            run = cached_downshift(model;kw...,options...,audit=true)
            @test run.complete && run.policies == base.policies
            @test isapprox(Float64.(run.mpi),Float64.(base.mpi);atol=1e-10,rtol=1e-9)
            @test all(a -> a.passed,run.audits)
            @test run.cache_exposures == base.cache_exposures
            @test run.cache_direct_evaluations == run.cache_exposures+run.cache_rebuild_evaluations
            @test run.cache_direct_evaluations+run.cache_recurrence_refreshes == 6
            @test run.cache_rebuilds > 0
            for (k,update) in enumerate(run.updates)
                if update.method == :refactorization
                    record = run.cache_records[k+1]
                    @test record.reason == :policy_refactorization
                    @test record.recurrence_refreshes == 0
                    @test record.direct_evaluations == length(record.frontier)
                end
            end
        end
        cache_only = cached_downshift(model;kw...,cache_rebuild_every=1,audit=true)
        @test cache_only.initial_factorizations == 1 && cache_only.rank_one_updates == 4
        @test cache_only.refactorizations == 0 && cache_only.cache_recurrence_refreshes == 0
        @test cache_only.cache_direct_evaluations == 6 && cache_only.cache_rebuild_evaluations == 2
        denied = cached_downshift(model;kw...,pivot_atol=1e6,refactor_on_failure=false)
        @test denied.status == :unresolved_pivot && isempty(denied.steps)
        @test denied.cache_pivot_preparations == 1 && length(denied.updates) == 1
        @test length(denied.cache_records) == 1 && all(ismissing,denied.mpi)
    end
end

@testset "Exact ties, strict inequalities and partial/failure statuses" begin
    for criterion in (:discounted,:average)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        model = downshift_case("all_tied")
        c = direct_downshift(model;kw...)
        m = cached_downshift(model;kw...,audit=true)
        @test m.complete && m.policies == c.policies && all(a -> a.passed,m.audits)
        @test m.exact_comparison_solves > 0
        @test validate_downshift(m,enumerate_bellman(model;kw...)).execution == :exact_pass
        unresolved = cached_downshift(model;kw...,exact_fallback=false)
        @test unresolved.status == :unresolved_comparison && isempty(unresolved.steps)
    end
    close = close_unequal_model()
    m = cached_downshift(close;beta=0.5,audit=true)
    @test m.complete && m.steps[1].selected == (2,1)
    @test m.steps[1].comparison_evidence == :exact_comparison
    for (model,expected,k) in ((nonpositive_marginal_model(),:no_positive_candidate,1),
                               (downshift_case("zero_denominator"),:complete,2))
        run=cached_downshift(model;beta=.5,audit=true)
        @test run.status==expected && length(run.steps)==k
        @test run.cache_pivot_preparations==k
        @test run.audit_evaluations==k+1 && all(a->a.passed,run.audits)
    end
    model = two_state_model()
    disconnected = ExplicitPolicyFamily([[2,2],[0,0]])
    run = cached_downshift(model;beta=0.5,family=disconnected,audit=true)
    @test run.status == :empty_frontier && run.initial_factorizations == 0
    @test run.last_cache === nothing && isempty(run.cache_records)
    zero = cached_downshift(model;beta=0.5,max_steps=0,audit=true)
    @test zero.status == :step_limit && zero.last_cache === nothing
    @test zero.cache_direct_evaluations == 0 && zero.cache_pivot_preparations == 0
    prefix = cached_downshift(model;beta=0.5,max_steps=1,audit=true)
    @test prefix.status == :step_limit && length(prefix.steps) == 1
    @test prefix.audit_evaluations == 2 && length(prefix.cache_records) == 2
    @test count(ismissing,prefix.mpi) == 3
    @test prefix.cache_direct_evaluations == 3 # includes prepared successor frontier
    @test validate_downshift(prefix,enumerate_bellman(model;beta=0.5)).execution == :partial_path_checked
    for name in ("nonconcave_one_state","negative_off_frontier","bellman_not_sign")
        model = downshift_case(name)
        c = direct_downshift(model;beta=0.75)
        m = cached_downshift(model;beta=0.75,audit=true)
        @test m.status == c.status && m.policies == c.policies
        ref = enumerate_bellman(model;beta=0.75)
        cv,mv = validate_downshift(c,ref),validate_downshift(m,ref)
        @test mv.execution == cv.execution
        @test mv.bellman_path == cv.bellman_path
        @test mv.reference_sign_indexability == cv.reference_sign_indexability
    end
end

@testset "Uncontrollable/transient/periodic policies, normalization, precision" begin
    for model in (transient_model(),periodic_model(),downshift_case("uncontrollable")), criterion in (:discounted,:average)
        kw = criterion == :discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        c = direct_downshift(model;kw...)
        m = cached_downshift(model;kw...,audit=true)
        @test m.status == c.status && m.policies == c.policies
        @test all(a -> a.passed,m.audits)
        @test all(all(p[.!model.controllable] .== 0) for p in m.policies)
    end
    @test cached_downshift(multichain_model();criterion=:average).status == :unsupported_average_policy
    model = two_state_model()
    base = cached_downshift(model;criterion=:average,audit=true)
    for w in ([1.0,0.0],[0.0,1.0],[0.25,0.75],[2.0,-1.0])
        m = cached_downshift(model;criterion=:average,omega=w,audit=true)
        @test m.complete && m.policies == base.policies && all(a -> a.passed,m.audits)
        @test isapprox(Float64.(m.mpi),Float64.(base.mpi);atol=1e-10,rtol=1e-9)
    end
    for beta in (0.125,0.9,0.99,0.999)
        c = direct_downshift(model;beta=beta)
        m = cached_downshift(model;beta=beta,audit=true)
        @test m.complete && m.policies == c.policies && all(a -> a.passed,m.audits)
        @test isapprox(Float64.(m.mpi),Float64.(c.mpi);atol=1e-8,rtol=1e-8)
    end
    setprecision(BigFloat,256) do
        b = two_state_model(float_type=BigFloat)
        for criterion in (:discounted,:average)
            kw = criterion == :discounted ? (;criterion=criterion,beta=BigFloat(0.5)) : (;criterion=criterion)
            run = cached_downshift(b;kw...,audit=true)
            @test run.complete && all(a -> a.passed,run.audits)
            @test run.mpi[1,1] isa BigFloat
        end
    end
end

@testset "Stale/corrupt cache detection and input validation" begin
    model = two_state_model()
    state = initialize_rankone(model,[2,2];beta=0.5)
    cache,_ = initialize_marginal_cache(rankone_evaluation(state))
    @test audit_marginal_cache(state,cache).passed
    cache.f[1] += 0.01
    @test !audit_marginal_cache(state,cache).passed
    cache,_ = initialize_marginal_cache(rankone_evaluation(state))
    cache.rows[1][1] += 0.01
    @test !audit_marginal_cache(state,cache).passed
    cache,_ = initialize_marginal_cache(rankone_evaluation(state))
    cache.row_l1[1] = -1.0
    @test !audit_marginal_cache(state,cache).passed
    cache,_ = initialize_marginal_cache(rankone_evaluation(state))
    pivot = prepare_marginal_pivot(state,1)
    @test update_downshift!(state,1).accepted
    @test !audit_marginal_cache(state,cache).passed # stale policy labels
    next,_ = refresh_marginal_cache(cache,pivot,rankone_evaluation(state))
    @test audit_marginal_cache(state,next).passed
    @test_throws ArgumentError refresh_marginal_cache(next,pivot,rankone_evaluation(state))
    @test_throws ArgumentError cached_downshift(model;beta=0.5,cache_rebuild_every=-1)
    @test_throws ArgumentError cached_downshift(model;beta=1.0)
    @test_throws ArgumentError cached_downshift(model;criterion=:average,beta=0.5)
    @test_throws ArgumentError cached_downshift(model;beta=0.5,max_steps=5)
    @test_throws BoundsError prepare_marginal_pivot(state,0)
end

@testset "Independent jobs have serial/threaded agreement" begin
    seeds = collect(20260916:20260923)
    function cached_job(seed)
        model = dyadic_validation_model(seed;n=3,A=2)
        run = cached_downshift(model;beta=0.875)
        return (run.status,run.policies,run.mpi,run.cache_direct_evaluations,run.cache_recurrence_refreshes)
    end
    serial = [cached_job(seed) for seed in seeds]
    parallel = Vector{Any}(undef,length(seeds))
    Threads.@threads for k in eachindex(seeds)
        parallel[k] = cached_job(seeds[k])
    end
    @test isequal(serial,parallel)
end
