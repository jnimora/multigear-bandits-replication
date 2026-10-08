# DENSE restriction of companion eqs. exp-generic-costs and
# exp-transition-perturbation. Pilot grids and sample counts are separate.
# Each purpose has a stable string key. FNV-1a determines its UInt64 seed;
# generated primitives are archived, so RNG-version changes cannot lose inputs.
function performance_stream_seed(master::Integer,n::Integer,A::Integer,draw::Integer,purpose::Symbol)
    key="multigear-pilot-v1|$master|$n|$A|$draw|$purpose"
    seed=UInt64(0xcbf29ce484222325)
    for byte in codeunits(key)
        seed=(seed ⊻ UInt64(byte))*UInt64(0x100000001b3)
    end
    return seed
end

function performance_primitives(n::Integer,A::Integer,draw::Integer;master_seed::Integer=20260915)
    n>=2 && A>=1 && draw>=1 || throw(ArgumentError("Require n>=2, A>=1, and draw>=1."))
    q_rng=Xoshiro(performance_stream_seed(master_seed,n,A,draw,:resource))
    t_rng=Xoshiro(performance_stream_seed(master_seed,n,A,draw,:thresholds))
    b_rng=Xoshiro(performance_stream_seed(master_seed,n,A,draw,:basecost))
    p_rng=Xoshiro(performance_stream_seed(master_seed,n,A,draw,:transition))
    q=exp.(-0.25^2/2 .+ 0.25.*randn(q_rng,n,A))
    theta=0.5 .+ 2.0.*rand(t_rng,n,A)
    for i in 1:n; theta[i,:]=sort(theta[i,:];rev=true); end
    b=rand(b_rng,n)
    weights=randexp(p_rng,n,n,A+1)
    all(x->isfinite(x) && x>0,weights) || error("Invalid primitive weight: retain the failed draw, do not resample.")
    Pi=similar(weights); delta=0.05
    for a in 1:A+1, i in 1:n
        total=sum(@view weights[i,:,a])
        for j in 1:n; Pi[i,j,a]=(1-delta)*weights[i,j,a]/total; end
        Pi[i,i,a]+=delta/2; Pi[i,mod1(i+1,n),a]+=delta/2
    end
    return (n=Int(n),A=Int(A),draw=Int(draw),master_seed=Int(master_seed),q=q,theta=theta,
            b=b,weights=weights,Pi=Pi,delta=delta)
end

function performance_model(primitives,eta::Real)
    0<=eta<=1 && isfinite(eta) || throw(ArgumentError("eta must be in [0,1]."))
    p=primitives; n=p.n; A=p.A; P=similar(p.Pi); h=zeros(n,A+1); c=zeros(n,A+1)
    for i in 1:n
        h[i,A+1]=p.b[i]
        for a in A:-1:1; h[i,a]=h[i,a+1]+p.q[i,a]*p.theta[i,a]; end
        for a in 1:A; c[i,a+1]=c[i,a]+p.q[i,a]; end
        for j in 1:n
            P[i,j,1]=p.Pi[i,j,1]
            for a in 1:A; P[i,j,a+1]=(1-eta)*p.Pi[i,j,1]+eta*p.Pi[i,j,a+1]; end
        end
    end
    return FiniteMultiGearModel(P,h,c)
end

# Independent dense actions, unstructured rewards, unit gear resources. Keep
# the historical structured generator above unchanged for archive replay.
function random_dense_unit_primitives(n::Integer,A::Integer,draw::Integer;master_seed::Integer=20261002)
    n>=1 && A>=1 && draw>=1 || throw(ArgumentError("Require n>=1, A>=1, and draw>=1."))
    seeds=Dict(s=>performance_stream_seed(master_seed,n,A,draw,Symbol("iid_dense_",s))
               for s in ("transitions","rewards"))
    P=randexp(Xoshiro(seeds["transitions"]),n,n,A+1)
    all(x->isfinite(x) && x>0,P) || error("Invalid transition draw; do not resample.")
    for a in 1:A+1,i in 1:n
        total=sum(@view P[i,:,a])
        @inbounds for j in 1:n;P[i,j,a]/=total;end
    end
    h=-rand(Xoshiro(seeds["rewards"]),n,A+1)
    c=repeat(reshape(Float64.(0:A),1,A+1),n,1)
    return (;P,h,c,seeds)
end
