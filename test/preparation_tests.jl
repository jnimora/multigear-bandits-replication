# Load the allocation fixture explicitly; the correctness suite does not otherwise
# include this separate fixture file.
if !isdefined(@__MODULE__,:fp_memory_model)
    include(joinpath(@__DIR__,"fixtures_fp_memory.jl"))
end

# Test helpers are top-level and all per-job bindings are explicitly local.
# No RNGs, models, workspaces, or result bindings are shared between threads.
function preparation_thread_job(seed,criterion)
    local model,_=random_test_model(seed;n=4,A=3)
    local input=PerformanceInput(model)
    local kw=criterion==:discounted ? (;criterion=criterion,beta=0.875) : (;criterion=criterion)
    local trace=PreparationTrace()
    local run=prepare_performance_run(input,:FP;kw...,preparation_trace=trace)
    yield()
    run_performance!(run)
    local result=performance_snapshot(run)
    yield()
    return (seed=seed,status=result.status,order=result.order,mpi=result.mpi,
        columns=[r.columns for r in trace.residuals])
end

@testset "Old/new initialization, all solved-column checks, both criteria" begin
    for model in (one_state_model(),two_state_model()), criterion in (:discounted,:average)
        input=PerformanceInput(model)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        for block in (1,3,64), recovery in (false,true)
            a=audit_fp_initialization(input;kw...,keep_recovery=recovery,block_columns=block)
            @test a.passed && a.mapping_agreement
            @test a.old_audit_passed && a.new_audit_passed
            @test a.old_policy_factorizations==a.new_policy_factorizations==1
            @test a.old_reference_factorizations==a.new_reference_factorizations==1
        end
        for alg in (:C,:R,:M,:FP)
            trace=PreparationTrace()
            run=prepare_performance_run(input,alg;kw...,preparation_trace=trace)
            d=nstates(model)+(criterion==:average); C=ncontrollable(model)
            expected=alg==:C ? 2 : alg in (:R,:M) ? d+2 : 2*(C+2)
            @test sum(r.columns for r in trace.residuals)==expected
            @test all(r->0<=r.maximum_backward_error<=1e-10,trace.residuals)
            @test all(s->s.seconds>=0 && s.allocated_bytes>=0,trace.stages)
            @test audit_performance_state(run).passed
        end
        for terminal in (false,true)
            opts=PerformanceOptions(terminal_update=terminal)
            old=prepare_performance_run(input,:FP;kw...,options=opts,fp_initializer=:legacy)
            new=prepare_performance_run(input,:FP;kw...,options=opts,fp_initializer=:optimized)
            @test run_performance!(old)==:complete
            @test run_performance!(new)==:complete
            @test performance_agreement(performance_snapshot(old),performance_snapshot(new))
            @test performance_counts(old)==performance_counts(new)
        end
    end
end

@testset "Binary, suffix, uncontrollable, family and normalization cases" begin
    for n in (4,8), A in (1,3), criterion in (:discounted,:average)
        input=PerformanceInput(fp_memory_model(n,A))
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.99) : (;criterion=criterion,omega=fill(1/n,n))
        trace=PreparationTrace()
        w=initialize_reduced_fp_optimized(input;kw...,trace=trace,block_columns=3)
        @test audit_reduced_fp(w).passed
        @test w.counts.reference_factorizations==(A==1 ? 0 : 1)
        @test (A==1)==isempty(w.ell)
        @test w.recovery_W===nothing && w.recovery_values===nothing
        @test (A==1)==!any(s->s.stage==:reference_transform,trace.stages)
        @test audit_fp_initialization(input;kw...,block_columns=3).passed
        act=fill(A,n); act[1]=0; act[n]=1
        @test audit_fp_initialization(input;kw...,actions=act,keep_recovery=true,block_columns=3).passed
        @test audit_fp_initialization(input;kw...,actions=zeros(Int,n),block_columns=1).passed
    end
    for seed in 2101:2104, criterion in (:discounted,:average), block in (1,3,64)
        model,actions=random_test_model(seed;n=5,A=3)
        input=PerformanceInput(model)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.95) : (;criterion=criterion,omega=[0.1,0.2,0.3,0.2,0.2])
        @test audit_fp_initialization(input;kw...,block_columns=block).passed
        @test audit_fp_initialization(input;kw...,actions=actions,block_columns=block,keep_recovery=true).passed
    end
    for family in (OrderedThresholdFamily([1,2]),ExplicitPolicyFamily([[2,2],[1,2],[1,1],[0,1],[0,0]])),
            criterion in (:discounted,:average)
        kw=criterion==:discounted ? (;criterion=criterion,beta=0.5) : (;criterion=criterion)
        input=PerformanceInput(two_state_model())
        @test audit_fp_initialization(input;kw...,family=family).passed
        old=audit_performance_run(input,:FP;kw...,family=family,fp_initializer=:legacy)
        new=audit_performance_run(input,:FP;kw...,family=family)
        @test old.passed && new.passed
        @test performance_agreement(old.snapshot,new.snapshot)
    end
    # Only the optimized performance domain is restricted. The original general
    # initializer still handles these unichain but non-strictly-positive policies.
    for model in (transient_model(),periodic_model())
        @test audit_reduced_fp(initialize_reduced_fp(model;criterion=:average)).passed
        @test_throws ArgumentError initialize_reduced_fp_optimized(PerformanceInput(model);criterion=:average)
        @test audit_fp_initialization(PerformanceInput(model);beta=0.5).passed
    end
end

@testset "Corrupted solved columns and input validation are not accepted" begin
    M=Matrix{Float64}(I,3,3); B=reshape([1.0,2,3,4,5,6],3,2)
    @test MultiGearBandits._preparation_column_check(M,B,B,1e-10)==0.0
    @test MultiGearBandits._preparation_column_check(M,M,nothing,1e-10;identity_rows=1:3)==0.0
    X=copy(M); X[2,1]+=0.01
    @test_throws ErrorException MultiGearBandits._preparation_column_check(M,X,nothing,1e-10;identity_rows=1:3)
    V=copy(B); V[2,2]+=0.01
    @test_throws ErrorException MultiGearBandits._preparation_column_check(M,V,B,1e-10)
    V[1,1]=NaN
    @test_throws ErrorException MultiGearBandits._preparation_column_check(M,V,B,1e-10)
    @test_throws DimensionMismatch MultiGearBandits._preparation_column_check(M,B,ones(2,2),1e-10)
    @test_throws ArgumentError MultiGearBandits._preparation_column_check(M,B,B,-1.0)
    input=PerformanceInput(two_state_model())
    @test_throws ArgumentError initialize_reduced_fp_optimized(input;beta=0.5,block_columns=0)
    @test_throws ArgumentError initialize_reduced_fp_optimized(input;beta=0.5,actions=[3,0])
    @test_throws ArgumentError prepare_performance_run(input,:FP;beta=0.5,fp_initializer=:invented)
    @test_throws ArgumentError initialize_reduced_fp_optimized(input;criterion=:average,omega=[1.0,1.0])
end

@testset "Per-job ownership survives repeated task interleaving" begin
    seeds=collect(3101:3106)
    for criterion in (:discounted,:average)
        serial=[preparation_thread_job(seed,criterion) for seed in seeds]
        for _ in 1:3
            parallel=similar(serial)
            Threads.@threads for j in eachindex(seeds)
                parallel[j]=preparation_thread_job(seeds[j],criterion)
            end
            @test isequal(serial,parallel)
        end
    end
end

@testset "Completion, numerical MP monotonicity and DAI verification stay distinct" begin
    good=(complete=true,ratio=[1.0,1.0,2.0])
    bad=(complete=true,ratio=[1.0,0.8,2.0])
    partial=(complete=false,ratio=[1.0])
    @test performance_mpi_screen(good).status==:numerical_nondecreasing
    @test performance_mpi_screen(bad).status==:resolved_decrease
    @test performance_mpi_screen(bad).decreasing_positions==[2]
    @test performance_mpi_screen(partial).status==:incomplete_no_resolved_decrease
    @test !performance_mpi_screen(good).dai_verified
    @test !performance_mpi_screen(bad).dai_verified
    @test performance_mpi_screen((complete=true,ratio=[1.0,NaN])).status==:nonfinite
    @test_throws ArgumentError performance_mpi_screen(good;atol=-1.0)
end

# The preceding performance_tests.jl already included performance_support.jl.
include(joinpath(@__DIR__,"..","scripts","performance_replay_support.jl"))

@testset "Frozen input round-trip, tampering, batch state and paired reports" begin
    mktempdir() do dir
        model=two_state_model()
        hash=pilot_archive_model(dir,"example",model); path=joinpath(dir,"example.toml")
        loaded=replay_load_model(path,hash)
        @test isequal(reinterpret(UInt64,vec(loaded.P)),reinterpret(UInt64,vec(model.P)))
        @test isequal(reinterpret(UInt64,vec(loaded.h)),reinterpret(UInt64,vec(model.h)))
        @test isequal(reinterpret(UInt64,vec(loaded.c)),reinterpret(UInt64,vec(model.c)))
        @test_throws ErrorException replay_load_model(path,repeat("0",64))
        originalbytes=read(path); open(path,"a") do io; println(io,"# changed"); end
        @test_throws ErrorException replay_load_model(path,hash)
        write(path,originalbytes)
        @test file_sha(path)==hash
    end
    input=PerformanceInput(two_state_model()); kw=(criterion=:discounted,beta=0.5)
    options=PerformanceOptions()
    for variant in REPLAY_VARIANTS
        alg=variant==:FP_legacy ? :FP : variant
        init=variant==:FP_legacy ? :legacy : :optimized
        reference=audit_performance_run(input,alg;kw...,fp_initializer=init).snapshot
        saved=Dict("validation"=>Dict("C"=>Dict("result"=>pilot_snapshot_record(reference))))
        @test replay_matches_archive(reference,saved)
        for phase in PERFORMANCE_PHASES
            row=replay_measure(input,Val(variant),kw,options,3,phase,3,reference,1e-7;collect_gc=false)
            @test row["matches_validated_execution"] && row["complete"]
            @test row["equal_work_counts_in_batch"] && row["validated_outputs"]==3
            @test row["seconds"]==row["window_seconds"]/3
            @test row["allocated_bytes"]==row["window_allocated_bytes"]/3
            @test row["counts_per_evaluation"]["assignments"]==4
        end
    end
    c=replay_configuration(joinpath(@__DIR__,"..","configs","initialization_replay_smoke.toml"))
    @test c["require_clean_git"]
    @test c["reference_block_columns"]==64
    mktempdir() do dir
        small=copy(c); small["states"]=[2]; small["gears"]=[2]; small["draws"]=[1]
        small["eta"]=[0.0]; small["criteria"]=["discounted"]
        meta=Dict("protocol"=>"fair-performance-pilot-v1","terminal_update"=>false,
            "configuration"=>Dict("terminal_update"=>false))
        case=Dict("case_id"=>"N2_A2_draw1_eta0.0_discounted","N"=>2,"A"=>2,
            "draw"=>1,"eta"=>0.0,"criterion"=>"discounted","model_sha256"=>repeat("a",64),
            "validation"=>Dict("C"=>Dict("present"=>true)))
        pilot_write_toml(joinpath(dir,"metadata.toml"),meta)
        pilot_write_toml(joinpath(dir,"cases.toml"),Dict("case"=>[case]))
        _,selected=replay_cases(dir,small)
        @test length(selected)==1
        pilot_write_toml(joinpath(dir,"cases.toml"),Dict("case"=>[case,case]))
        @test_throws ErrorException replay_cases(dir,small)
        case["case_id"]="../escape_discounted"
        pilot_write_toml(joinpath(dir,"cases.toml"),Dict("case"=>[case]))
        @test_throws ErrorException replay_cases(dir,small)
        pilot_write_toml(joinpath(dir,"cases.toml"),Dict("case"=>Dict{String,Any}[]))
        @test_throws ErrorException replay_cases(dir,small)
    end
    # Historical reporting preserves the recorded eligibility of older nonmonotone
    # runs; the corrected driver stops them. Old initializer is a distinct label.
    rows=Dict{String,Any}[
        Dict("case_id"=>"a","algorithm"=>"FP_legacy","phase"=>"initialization","repetition"=>1,
            "eligible"=>true,"executed"=>true,"seconds"=>2.0,"allocated_bytes"=>4.0,"gc_seconds"=>0.0,
            "mpi_monotonicity"=>"resolved_decrease","dai_verified"=>false),
        Dict("case_id"=>"a","algorithm"=>"FP","phase"=>"initialization","repetition"=>1,
            "eligible"=>true,"executed"=>true,"seconds"=>1.0,"allocated_bytes"=>2.0,"gc_seconds"=>0.0,
            "mpi_monotonicity"=>"resolved_decrease","dai_verified"=>false)]
    _,ratios=pilot_summaries(rows)
    @test only(ratios)["baseline"]=="FP_legacy"
    @test only(ratios)["baseline_over_FP"]==2.0
    @test all(!r["dai_verified"] for r in rows)
end
