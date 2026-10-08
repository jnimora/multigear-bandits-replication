using Test,LinearAlgebra,Random,TOML
BLAS.set_num_threads(1)
include("../scripts/blocked_multigear/blocked_fp.jl")
using .MultiGearBandits
const B=BlockedMultiGearFP
const M=MultiGearBandits
include("fixtures/downshift_models.jl")
include("fixtures_fp_memory.jl")

function paircheck(ref,b)
    w=b.run.core;s=B.materialized_core(b);m=w.m
    @test ref.actions==w.actions && ref.state_at==w.state_at && ref.slot_of==w.slot_of
    @test ref.m==m
    @test ref.X[1:m,1:m]≈s.X[1:m,1:m] atol=3e-11 rtol=3e-9
    @test ref.phi[1:m]≈w.phi[1:m] atol=3e-10 rtol=3e-9
    @test ref.psi[1:m]≈w.psi[1:m] atol=3e-10 rtol=3e-9
    for t in 1:m
        i=w.state_at[t]
        if !w.frontier_only || M._fp_feasible(w,w.family,i)
            @test isapprox(ref.f[t],w.f[t];atol=3e-10,rtol=3e-9)
            @test isapprox(ref.g[t],w.g[t];atol=3e-10,rtol=3e-9)
        end
    end
    @test B.audit(b).passed
    for j in eachindex(b.reference_ids)
        b.reference_ids[j]>0 || continue
        @test b.Y[1:m,j]≈transpose(s.X[1:m,1:m])*b.L[1:m,j] atol=3e-10 rtol=3e-9
    end
end

@testset "General costs, families, reference rows, packing, and fresh solves" begin
    for criterion in (:discounted,:average),n in (4,7),A in (1,2,4),kind in (:free,:threshold,:rows,:explicit),masked in (false,true)
        original=dyadic_validation_model(130+n+A;n,A)
        mask=trues(n);masked && (mask[2]=false)
        model=FiniteMultiGearModel(original.P,original.h,original.c;controllable=mask)
        states=findall(mask);order=reverse(states)
        rows=[order[1:2:end],order[2:2:end]]
        seq=[(i,a) for i in order for a in A:-1:1]
        policies=[mask.*A];actions=copy(policies[1])
        for (i,a) in seq;actions=copy(actions);actions[i]-=1;push!(policies,actions);end
        family=kind==:free ? UnrestrictedFamily() : kind==:threshold ? OrderedThresholdFamily(order) :
            kind==:rows ? RowThresholdFamily(rows) : ExplicitPolicyFamily(policies)
        kw=criterion==:average ? (;criterion) : (;criterion,beta=.875)
        input=PerformanceInput(model)
        for width in (0,2,5),block in (1,3,16),layout in (:physical,:state)
            ref=prepare_performance_run(input,:FP;kw...,family)
            b=B.prepare(input;kw...,family,blocksize=block,reference_width=width,reference_panels=2,reference_order=layout)
            paircheck(ref.core,b)
            for (i,a) in seq
                @test M._fp_feasible(ref.core,family,i)
                @test M.pivot_reduced_fp!(ref.core,i;check_unichain=false).accepted
                @test B.pivot!(b,i;check_unichain=false).accepted
                paircheck(ref.core,b)
            end
        end
    end
end

@testset "Shared selection, positive work, monotonicity, and accepted prefixes" begin
    for seed in 12:17,criterion in (:discounted,:average),kind in (:free,:threshold,:rows)
        model=dyadic_validation_model(seed;n=4,A=3);input=PerformanceInput(model)
        family=kind==:free ? UnrestrictedFamily() : kind==:threshold ? OrderedThresholdFamily([1,2,3,4]) : RowThresholdFamily([[1,3],[2,4]])
        kw=criterion==:average ? (;criterion) : (;criterion,beta=.875)
        a=prepare_performance_run(input,:FP;kw...,family);b=B.prepare(input;kw...,family,blocksize=3,reference_width=2)
        while a.status in (:ready,:running) && b.run.status in (:ready,:running)
            paircheck(a.core,b);step_performance!(a);B.step!(b)
            @test a.status==b.run.status
            @test performance_agreement(performance_snapshot(a),B.snapshot(b))
        end
    end
    input=PerformanceInput(fp_memory_model(8,3))
    for fault in (:nearzero,:negative,:nonmonotone,:denominator)
        opts=fault==:denominator ? PerformanceOptions(pivot_rtol=2.0,exact_fallback=false) : PerformanceOptions(exact_fallback=false)
        a=prepare_performance_run(input,:FP;beta=.875,options=opts)
        b=B.prepare(input;beta=.875,options=opts,blocksize=3,reference_width=2)
        for r in (a,b.run)
            if fault==:nearzero;r.core.g[1]=0.0
            elseif fault==:negative;r.core.g.=-1.0
            elseif fault==:nonmonotone;r.log.k=1;r.log.ratio[1]=1e9
            end
        end
        saved=copy(b.run.core.actions);x=copy(b.run.core.X)
        step_performance!(a);B.step!(b)
        @test a.status==b.run.status
        @test b.run.core.actions==saved && b.run.core.X==x && b.q==0
    end
end

@testset "Synthetic complete paths and warmed allocation checks" begin
    for criterion in (:discounted,:average),A in (1,3),familykind in (:free,:threshold),width in (0,16)
        input=PerformanceInput(fp_memory_model(128,A));kw=criterion==:average ? (;criterion) : (;criterion,beta=.875)
        family=familykind==:free ? UnrestrictedFamily() : OrderedThresholdFamily(collect(1:128))
        B.run!(B.prepare(input;kw...,family,reference_width=width))
        a=prepare_performance_run(input,:FP;kw...,family);run_performance!(a)
        b=B.prepare(input;kw...,family,reference_width=width)
        bytes=@allocated B.run!(b)
        @test b.run.status==:complete
        @test performance_agreement(performance_snapshot(a),B.snapshot(b))
        @test B.audit(b).passed
        @test bytes==0
    end
end

@testset "Rejected updates preserve accepted state with pending corrections" begin
    for width in (0,2),fault in (:denominator,:finite_envelope,:negative,:nonmonotone)
        input=PerformanceInput(fp_memory_model(8,3))
        b=B.prepare(input;beta=.875,blocksize=4,reference_width=width,
                    options=PerformanceOptions(exact_fallback=false))
        B.step!(b);@test b.q==1
        w=b.run.core
        if fault==:negative;w.g.=-1.0
        elseif fault==:nonmonotone;b.run.log.ratio[b.run.log.k]=1e9
        elseif fault==:finite_envelope;b.absolute_bound=floatmax(Float64)
        end
        accepted=(X=copy(B.materialized_core(b).X),actions=copy(w.actions),phi=copy(w.phi),
                  psi=copy(w.psi),f=copy(w.f),g=copy(w.g),k=b.run.log.k,q=b.q,
                  U=copy(b.U[:,1:b.q]),V=copy(b.V[:,1:b.q]))
        if fault in (:denominator,:finite_envelope)
            j,_,e=M._performance_select(w,b.run.options);@test e==:numerical_screen
            report=B.pivot!(b,M._performance_state(w,j);check_unichain=false,
                pivot_rtol=fault==:denominator ? 2.0 : 1e-12)
            @test !report.accepted
            @test report.reason==(fault==:denominator ? :unsafe_denominator : :nonfinite_trial)
        else
            B.step!(b)
            @test b.run.status==(fault==:negative ? :no_positive_candidate : :nonmonotone)
        end
        @test b.q==accepted.q && b.run.log.k==accepted.k
        @test B.materialized_core(b).X==accepted.X && w.actions==accepted.actions
        @test w.phi==accepted.phi && w.psi==accepted.psi
        @test w.f==accepted.f && w.g==accepted.g
        @test b.U[:,1:b.q]==accepted.U && b.V[:,1:b.q]==accepted.V
    end
end

@testset "Public option compatibility and complete terminal updates" begin
    input=PerformanceInput(fp_memory_model(16,3))
    for criterion in (:average,:discounted),terminal in (false,true),frontier in (false,true),kind in (:free,:rows),width in (0,2)
        family=kind==:free ? UnrestrictedFamily() : RowThresholdFamily([collect(1:2:16),collect(2:2:16)])
        kw=criterion==:average ? (;criterion) : (;criterion,beta=.875)
        options=PerformanceOptions(terminal_update=terminal,fp_frontier_cache=frontier)
        a=prepare_performance_run(input,:FP;kw...,family,options);run_performance!(a)
        b=B.prepare(input;kw...,family,options,reference_width=width);B.run!(b)
        @test b.run.status==:complete
        @test performance_agreement(performance_snapshot(a),B.snapshot(b))
        @test B.audit(b).passed
        @test b.run.core.m==(terminal ? 0 : 1)
    end
end
