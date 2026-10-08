using Test, LinearAlgebra
include(joinpath(@__DIR__,"../scripts/binary_benchmark/blocked_fp.jl"))
const BF=BlockedBinaryFP
const BB=BinaryBenchmark
BLAS.set_num_threads(1)

@testset "Deferred tableau invariant at every pivot" begin
    for n in (2,7,31,66), beta in (.8,1.), b in (1,3,16,64), tail in (0,8)
        x=BB.generate(n,91+n,"exponential")
        ref=BB.prepare_fp(x,beta); w=BF.wrap(deepcopy(ref);blocksize=b,tail=tail)
        for k in 1:n-1
            p,v,st,_,_=BB.select_positive(ref.f,ref.g,ref.ids,ref.m,false,BB.Checks())
            @test st==:ok
            @test BB.pivot!(ref,p,BB.Checks())==BF.pivot!(w,p)==:ok
            m=ref.m; q=w.q
            actual=w.base.X[1:m,1:m]-w.U[1:m,1:q]*w.V[1:m,1:q]'
            @test actual≈ref.X[1:m,1:m] atol=3e-13 rtol=3e-12
            @test w.base.f[1:m]≈ref.f[1:m] atol=3e-13 rtol=3e-12
            @test w.base.g[1:m]≈ref.g[1:m] atol=3e-13 rtol=3e-12
            @test w.base.ids==ref.ids
        end
    end
end

@testset "Complete checked paths and independent fresh solves" begin
    for n in (2,9,33,129), seed in (41,82), beta in (.8,.99,1.), family in ("uniform","exponential")
        x=BB.generate(n,seed,family); ref=BB.run_fp!(BB.prepare_fp(x,beta))
        for b in (1,7,32,128)
            w=BF.run!(BF.prepare(x,beta;blocksize=b,tail=16)); s=w.base
            @test s.status==ref.status==:complete
            @test s.order==ref.order
            @test s.indices≈ref.indices atol=1e-11 rtol=1e-9
            @test s.rejected==ref.rejected
            audit=BB.validate_path(x,beta,s.order,s.indices;fresh_steps=unique([0,n÷2,n]))
            @test audit["status"]=="numerical_pcl_path_pass"
            @test audit["maximum_fresh_solve_error"]<1e-10
        end
    end
end

@testset "Selection failures, ties, and in-loop monotonicity" begin
    for case in (:negative,:nearzero,:nonfinite,:monotonicity,:pivot,:ties,:skip_negative)
        ref=BB.prepare_fp(BB.generate(5,44,"uniform"),.99)
        if case==:negative; ref.g.=-1
        elseif case==:nearzero; ref.g[1]=1e-14
        elseif case==:nonfinite; ref.f[2]=Inf
        elseif case==:pivot; ref.X.=0
        elseif case==:ties; ref.X.=Matrix{Float64}(I,5,5);ref.f.=1;ref.g.=1
        elseif case==:skip_negative;ref.g[1]=-1
        end
        w=BF.wrap(deepcopy(ref);blocksize=3,tail=0)
        if case==:monotonicity
            ref.k=1;ref.sequence[1]=1e4
            w.base.k=1;w.base.sequence[1]=1e4
        end
        BB.run_fp!(ref);BF.run!(w)
        @test w.base.status==ref.status
        @test w.base.k==ref.k
        @test w.base.order==ref.order
        @test w.base.rejected==ref.rejected
        case==:monotonicity && @test w.base.status==:nonmonotone
        case==:ties && @test w.base.order==collect(1:5)
    end
end

@testset "Exact Bellman oracle on dyadic fixtures" begin
    for seed in 1:5, beta in (.75,1.)
        n=3; rng=BB.Random.Xoshiro(seed); p0=zeros(n,n);p1=zeros(n,n)
        for p in (p0,p1),i in 1:n
            a=rand(rng,1:5); b=rand(rng,1:7-a);p[i,:]=[a,b,8-a-b]./8
        end
        x=BB.Input(p0,p1,zeros(n),Float64.(rand(rng,1:15,n))./16)
        oracle=BB.MGB.enumerate_bellman(BB.mg_input(x).model;criterion=beta==1 ? :average : :discounted,beta=beta==1 ? nothing : beta)
        @test oracle.complete
        if oracle.dai!==nothing
            s=BF.run!(BF.prepare(x,beta;blocksize=2,tail=0)).base
            @test s.status==:complete
            @test s.indices≈Float64.(oracle.dai[:,1]) atol=1e-10 rtol=1e-8
        end
    end
end

@testset "Bounded workspace and allocation-free loop" begin
    x=BB.generate(512,99,"exponential")
    BF.run!(BF.prepare(x,.99;blocksize=32))
    w=BF.prepare(x,.99;blocksize=32)
    bytes=@allocated BF.run!(w)
    @test w.base.status==:complete
    @test bytes<4096
    @test size(w.U)==size(w.V)==(512,32)
    @test_throws ArgumentError BF.prepare(x,.99;blocksize=0)
    @test_throws ArgumentError BF.prepare(x,.99;tail=-1)
    @test_throws ArgumentError BF.wrap(BB.prepare_fp(x,.99,true))
    println("Blocked index-loop Julia allocations: ",bytes," bytes")
end
