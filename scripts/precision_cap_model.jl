"""Reviewed larger-cap score/dual interface. It never calls or relaxes the old
H<=64 guard and never constructs a Whittle table-wide common denominator.
An independent scalar whole-charge certificate admits the exact block path;
policy-local exact equations then establish the common dual and its residuals.
"""
module PrecisionCapModel
using TOML
using ..RayleighAoII, ..RayleighBlock, ..RayleighTiming, ..RayleighLargeCap
using ..RayleighAllocation, ..ManyProjectAllocator, ..ManyProjectModel, ..PrecisionCapScores
const RA=RayleighAoII; const RB=RayleighBlock;const LC=RayleighLargeCap
const AL=RayleighAllocation;const MA=ManyProjectAllocator;const MM=ManyProjectModel;const PS=PrecisionCapScores
const Q=Rational{BigInt};const E=RB.Enclosed
qs(q::Q)=AL.qs(q)
struct Single
    model::RA.AoIIModel{Q}
    mpi::Matrix{Q}
end
struct Family
    id::String
    source_group::String
    single::Single
    weights::Vector{Q}
    base_counts::Vector{Int}
    budget_per_project::Q
    lambda::Q
    bound_per_project::Q
    bias::Vector{Vector{Q}}
    gains::Vector{Q}
    residual::Array{Q,3}
    nextbias::Array{Q,3}
    occupations::Vector{Vector{Q}}
    scores::Dict{Symbol,PS.Table}
    priority_rank::Matrix{Int}
    priority_positive::BitMatrix
    units::Vector{Int}
    tick::Q
    record::Dict{String,Any}
end
function guardtime(started,seconds)
    (time_ns()-started)/1e9<=seconds || throw(MA.AllocationLimit("Cap preparation time limit; no partial family admitted."))
end
"""Exact block path plus rates, O(AH) stored scalars rather than O(AH^2) policy
bias tables. The accepted exact block primitives are unchanged. Time/bit limits
are additional refusal gates, not numerical tolerances.
"""
function exact_block(m;seconds=600.0,max_bits=262144)
    H=m.H;A=length(m.p)-1;started=time_ns();w=RA.BlockWorkspace(m)
    order=Tuple{Int,Int}[];roots=Q[];hb=Q[];cb=Q[];mpi=Matrix{Q}(undef,H,A)
    for k in 1:A*H
        guardtime(started,seconds)
        e=RA.block_frontier!(w);isempty(e.frontier) && error("Premature empty exact frontier.")
        selected=first(e.frontier)
        for x in e.frontier
            isless((x[5],x[1],x[2]),(selected[5],selected[1],selected[2])) && (selected=x)
        end
        h,a,f,g,v=selected
        all(max(ndigits(abs(numerator(x));base=2),ndigits(denominator(x);base=2))<=max_bits for x in (v,e.hbar,e.cbar)) || throw(MA.AllocationLimit("Cap individual-rational bit limit."))
        push!(roots,v);push!(order,(h,a));push!(hb,e.hbar);push!(cb,e.cbar);mpi[h,a]=v
        RA.block_shift!(w,h,a)
    end
    w.terminal_gear==0 && w.lengths[1]==H-1 || error("Incomplete exact cap path.")
    # Once more, an O(H) per-state evaluation for the passive endpoint.
    endpoint=RA.excursion(m,zeros(Int,H));push!(hb,endpoint.hbar);push!(cb,endpoint.cbar)
    issorted(roots) && all(>(0),roots) || error("Exact cap path is not monotone positive.")
    for k in eachindex(roots)
        cb[k]>cb[k+1] || error("Path resource rates not strictly decreasing.")
        hb[k]+roots[k]*cb[k]==hb[k+1]+roots[k]*cb[k+1] || error("Adjacent gain-line intersection is not the assigned root.")
    end
    (order=order,roots=roots,hbar=hb,cbar=cb,mpi=mpi,seconds=(time_ns()-started)/1e9)
end
function actions_at(m,order,k)
    a=fill(length(m.p)-1,m.H)
    for j in 1:k-1;h,b=order[j];a[h]==b || error("Invalid prefix.");a[h]-=1;end
    a
end
function dual_at_path(m,path,weights,counts,b)
    base=sum(counts);rho=Q.(counts)./base;K=length(counts);A=length(m.p)-1;H=m.H
    # Sorted event merge. No evaluation of every affine line at every breakpoint.
    events=sort!([(weights[k]*path.roots[j],k,j) for k in 1:K for j in eachindex(path.roots)];by=identity)
    right=ones(Int,K);left=copy(right);lam=zero(Q)
    low=sum(rho[k]*path.cbar[1] for k in 1:K);high=low
    if low>b
        j=1;found=false
        while j<=length(events)
            lam=events[j][1];left=copy(right);high=low
            while j<=length(events) && events[j][1]==lam
                _,k,s=events[j];right[k]==s || error("Unordered class event prefix.")
                right[k]=s+1;j+=1
            end
            low=sum(rho[k]*path.cbar[right[k]] for k in 1:K)
            if low<=b;found=true;break;end
        end
        found || error("Dual budget crossing not found.")
    end
    low<=b && (lam==0 || high>=b) || error("Dual subgradient bracket.")
    theta=lam==0 || high==low ? zero(Q) : (b-low)/(high-low)
    0<=theta<=1 || error("Occupation mixture weight.")
    bias=Vector{Q}[];gains=Q[];occ=Vector{Q}[]
    residual=zeros(Q,K,H+1,A+1);nextbias=zeros(Q,K,H+1,A+1);checks=0
    for k in 1:K
        ar=actions_at(m,path.order,right[k]);al=actions_at(m,path.order,left[k])
        er=RA.excursion(m,ar);el=RA.excursion(m,al)
        er.hbar==path.hbar[right[k]] && er.cbar==path.cbar[right[k]] || error("Block/scalar right policy rate mismatch.")
        el.hbar==path.hbar[left[k]] && el.cbar==path.cbar[left[k]] || error("Block/scalar left policy rate mismatch.")
        v=vcat(zero(Q),weights[k].*er.phi.+lam.*er.gamma)
        v==vcat(zero(Q),weights[k].*el.phi.+lam.*el.gamma) || error("Exact bias at tied charge differs.")
        gain=weights[k]*er.hbar+lam*er.cbar
        push!(bias,v);push!(gains,gain)
        pr=AL.single_stationary(m,ar);pl=AL.single_stationary(m,al)
        push!(occ,(1-theta).*pr.+theta.*pl)
        nextbias[k,1,1]=(1-m.wR)*v[2];residual[k,1,1]=nextbias[k,1,1]-gain
        residual[k,1,1]==0 || error("Zero-state charged equality.")
        for h in 1:H,a in 0:A
            en=(1-m.p[a+1])*v[min(h+1,H)+1];nextbias[k,h+1,a+1]=en
            residual[k,h+1,a+1]=weights[k]*h+lam*m.c[a+1]+en-v[h+1]-gain
            residual[k,h+1,a+1]>=0 || error("Negative exact cap charged slack.")
            a==ar[h] && residual[k,h+1,a+1]!=0 && error("Right charged policy not optimal.")
            checks+=1
        end
    end
    ell=sum(rho.*gains)-lam*b
    for scale in (1,2,4,8,16,32)
        population=scale*base
        sum(scale*counts[k]*gains[k] for k in 1:K)-lam*population*b==population*ell || error("Nonproportional cap-family bound.")
    end
    resource=(1-theta)*low+theta*high
    cost=sum(rho[k]*weights[k]*((1-theta)*path.hbar[right[k]]+theta*path.hbar[left[k]]) for k in 1:K)
    cost==ell && resource<=b && lam*(b-resource)==0 || error("Exact relaxation primal/dual mismatch.")
    record=Dict{String,Any}("lambda_star"=>qs(lam),"bound_per_project"=>qs(ell),"budget_per_project"=>qs(b),
      "base_population"=>base,"base_class_counts"=>counts,"class_weights"=>qs.(weights),
      "left_policy_indices"=>left,"right_policy_indices"=>right,"relaxed_mixture_weight"=>qs(theta),
      "relaxed_resource_per_project"=>qs(resource),"relaxed_cost_per_project"=>qs(cost),
      "relaxed_primal_equals_dual"=>true,"all_action_Bellman_checks"=>checks,
      "class_biases"=>[qs.(v) for v in bias],"class_gains"=>qs.(gains),
      "relaxed_class_occupancies"=>[qs.(v) for v in occ],"proportional_scaling_verified"=>true,
      "charge_convention"=>"first sorted weighted root whose right aggregate resource is at most b; smallest nonnegative maximizer")
    (lambda=lam,ell=ell,bias=bias,gains=gains,residual=residual,nextbias=nextbias,occupations=occ,record=record)
end
function prepare(spec,certificate_config;max_population=1600,seconds=600.0,progress=(k,n)->nothing)
    started=time_ns();H=Int(spec["H"])
    1<=H<=2048 && length(spec["success"]) in (2,3) || error("Cap interface admits H<=2048 and A<=2 only.")
    m=RA.from_success(H;nsrc=Int(spec["source_states"]),wR=spec["wR"],success=spec["success"],kappa=spec["kappa"],epsilon=spec["epsilon"],chi="1")
    all(Q(Float64(p))==p && denominator(p)<=BigInt(2)^53 for p in vcat([m.wR],m.p)) || error("Exact sampler cannot round transition primitives.")
    path=exact_block(m;seconds=seconds)
    cert=try
        LC.certify_chain(m,path.order,certificate_config;progress=progress)
    catch err
        message=sprint(showerror,err)
        if startswith(message,"Certificate cooperative time limit") || startswith(message,"Unresolved certificate outside its saved exact-policy budget")
            throw(MA.AllocationLimit("Larger-cap gate refused within unchanged certificate limits: "*message))
        end
        rethrow()
    end
    for (q,r) in zip(path.roots,cert["path"])
        Q(r["root_lo"])<=q<=Q(r["root_hi"]) || error("Exact root misses independently evaluated enclosure.")
    end
    guardtime(started,seconds)
    weights=AL.parseq.(spec["weights"]);counts=Int.(spec["base_counts"])
    length(weights)==length(counts) && all(weights.>0) && all(counts.>0) && sum(counts)==50 || error("Cap class proportions changed.")
    # CAP PARTNERS ALL READ THIS SAME PRE-SPECIFIED BUDGET STRING.
    b=AL.parseq(spec["budget_per_project"]);b>=0 || error("Negative budget.")
    D=dual_at_path(m,path,weights,counts,b);K=length(weights);A=length(m.p)-1
    tables=Dict{Symbol,PS.Table}()
    for name in (:WHITTLE_K,:LAGRANGIAN_K,:MYOPIC_K)
        v=zeros(Q,K*H,A+1)
        for k in 1:K,h in 1:H,a in 1:A
            row=(k-1)*H+h;s=min(h+1,H)
            if name==:WHITTLE_K
                v[row,a+1]=v[row,a]+(m.c[a+1]-m.c[a])*weights[k]*path.mpi[h,a]
            elseif name==:LAGRANGIAN_K
                v[row,a+1]=(m.p[a+1]-m.p[1])*D.bias[k][s+1]-D.lambda*m.c[a+1]
            else
                v[row,a+1]=weights[k]*s*(m.p[a+1]-m.p[1])
            end
        end
        tables[name]=PS.compile(v;max_population=max_population)
        guardtime(started,seconds)
    end
    tables[:LAGRANGIAN_NI]=tables[:LAGRANGIAN_K];tables[:LAGRANGIAN_FC]=tables[:LAGRANGIAN_K]
    values=Q[weights[k]*path.mpi[h,a] for k in 1:K for h in 1:H for a in 1:A]
    all(>(0),values) || error("Nonpositive priority index.")
    ordered=sort!(unique(copy(values)));lookup=Dict(v=>i for (i,v) in enumerate(ordered));rank=zeros(Int,K*H,A)
    for k in 1:K,h in 1:H,a in 1:A;rank[(k-1)*H+h,a]=lookup[weights[k]*path.mpi[h,a]];end
    units,tick=MA.resource_encoding(m.c)
    encoding=Dict{String,Any}()
    for name in (:WHITTLE_K,:LAGRANGIAN_K,:MYOPIC_K)
        t=tables[name]
        encoding[String(name)]=Dict("table_common_denominator_constructed"=>false,
          "maximum_entry_numerator_bits"=>maximum(ndigits(abs(numerator(x));base=2) for x in t.values),
          "maximum_entry_denominator_bits"=>maximum(ndigits(denominator(x);base=2) for x in t.values),
          "retained_table_bytes"=>Base.summarysize(t))
    end
    record=Dict{String,Any}("id"=>spec["id"],"specification"=>Dict(spec),"dual"=>D.record,
      "single_project"=>Dict("H"=>H,"A"=>A,"roots"=>qs.(path.roots),"order"=>[collect(x) for x in path.order],
        "indices"=>[qs.(path.mpi[h,:]) for h in 1:H],"path_age_rates"=>qs.(path.hbar),"path_resource_rates"=>qs.(path.cbar),
        "reset_probabilities"=>qs.(m.p),"resource_menu"=>qs.(m.c),"source_stay_probability"=>qs(m.wR),"bellman_certificate"=>cert),
      "score_encodings"=>encoding,"table_common_denominator_constructed"=>false,
      "integer_resource_units"=>units,"resource_quantum"=>qs(tick),"dai_verified"=>true,
      "scope"=>"larger-cap exact per-row scores; independent fixed-instance whole-charge chain; selective exact allocation; not a uniform PCL theorem",
      "diagnostic_preparation_seconds"=>(time_ns()-started)/1e9)
    Family(String(spec["id"]),String(spec["source_group"]),Single(m,path.mpi),weights,counts,b,D.lambda,D.ell,
        D.bias,D.gains,D.residual,D.nextbias,D.occupations,tables,rank,trues(K*H,A),units,tick,record)
end
mutable struct Allocators
    work::Dict{Symbol,PS.Workspace}
    priority_actions::Vector{Int}
    heap::Vector{NTuple{3,Int}}
    max_spend::Dict{Int,Int}
    limits::MA.Limits
end
function Allocators(f::Family,L::Int;limits=MA.Limits())
    Allocators(Dict(p=>PS.Workspace(f.scores[p],L) for p in MA.POLICIES if p!=:WHITTLE_P),zeros(Int,L),NTuple{3,Int}[],Dict{Int,Int}(),limits)
end
function decide!(w::Allocators,f::Family,policy::Symbol,types::Vector{Int},C::Int)
    if policy==:WHITTLE_P
        r=MA.priority!(w.priority_actions,w.heap,types,f.priority_rank,f.priority_positive,f.units,C)
        return w.priority_actions,r
    end
    domain=policy==:LAGRANGIAN_NI ? :ni : (policy==:LAGRANGIAN_FC ? :fc : :k)
    r=PS.allocate!(w.work[policy],f.scores[policy],types,f.units,C,domain;limits=w.limits)
    w.work[policy].actions,r
end
end # module
