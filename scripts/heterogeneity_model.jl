"""Class-specific dynamics with a common gear/resource menu. All numerical
single-project, certification and online allocation kernels are reused unchanged.
The runtime object contains numerical data, not its human-readable report copy.
"""
module HeterogeneityModel
using TOML, SHA
using ..RayleighAoII, ..RayleighBlock, ..RayleighAllocation, ..RayleighLargeCap
using ..ManyProjectAllocator, ..ManyProjectSimulation, ..PrecisionCapScores, ..PrecisionCapModel
const RA=RayleighAoII;const RB=RayleighBlock;const AL=RayleighAllocation;const LC=RayleighLargeCap
const MA=ManyProjectAllocator;const MS=ManyProjectSimulation;const PS=PrecisionCapScores;const CM=PrecisionCapModel
const Q=Rational{BigInt}
const REVISION="rayleigh-heterogeneous-timescales-v1"
qs(q::Q)=AL.qs(q)
struct ClassData
    model::RA.AoIIModel{Q}
    order::Vector{Tuple{Int,Int}}
    roots::Vector{Q}
    hbar::Vector{Q}
    cbar::Vector{Q}
    mpi::Matrix{Q}
end
struct Family
    id::String
    source_group::String
    contrast::Int
    H::Int
    classes::Vector{ClassData}
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
    resource::Vector{Q}
end
function value_digest(v)
    ctx=SHA.SHA2_256_CTX()
    for q in v
        SHA.update!(ctx,codeunits(qs(q))); SHA.update!(ctx,UInt8[0x0a])
    end
    bytes2hex(SHA.digest!(ctx))
end
function make_model(d::Q,H::Int)
    0<d<=1//8 && 1<=H<=2048 || error("Unsupported time scale or cap.")
    m=RA.from_success(H;nsrc=2,wR=1-d,success=Q[0,d/(1-2d),3d/(1-2d)],
        kappa=1/d,epsilon=1/(32d*d),chi="1")
    m.p==Q[d,2d,4d] && m.c==Q[0,33//32,105//32] || error("Common-menu construction changed.")
    for p in vcat(m.p,[m.wR])
        MS.probability_threshold(p); Q(Float64(p))==p || error("Float64 primitives would change the exact model.")
    end
    m
end
function prepare_class(m,c;progress=(k,n)->nothing)
    p=CM.exact_block(m;seconds=Float64(c["resource_limits"]["preparation_seconds"]),
                     max_bits=Int(c["resource_limits"]["max_individual_rational_bits"]))
    cert=try
        LC.certify_chain(m,p.order,c;progress=progress)
    catch err
        msg=sprint(showerror,err)
        if startswith(msg,"Certificate cooperative time limit") || startswith(msg,"Unresolved certificate outside its saved exact-policy budget")
            throw(MA.AllocationLimit("New-study offline certificate budget reached: "*msg))
        end
        rethrow()
    end
    cert["passed"] && length(cert["path"])==length(p.roots) || error("Class certificate incomplete.")
    for (root,r) in zip(p.roots,cert["path"])
        Q(r["root_lo"])<=root<=Q(r["root_hi"]) || error("Exact root outside independent class certificate.")
    end
    d=ClassData(m,p.order,p.roots,p.hbar,p.cbar,p.mpi)
    record=Dict{String,Any}("H"=>m.H,"A"=>length(m.p)-1,"p_exact"=>qs.(m.p),
        "wR_exact"=>qs(m.wR),"u_exact"=>qs(1-m.wR),"c_exact"=>qs.(m.c),
        "order"=>[collect(pair) for pair in d.order],"root_digest"=>value_digest(d.roots),
        "age_rate_digest"=>value_digest(d.hbar),"resource_rate_digest"=>value_digest(d.cbar),
        "index_digest"=>value_digest(d.mpi),"certificate"=>cert,
        "dai_verified"=>true,"certificate_scope"=>"fixed class instance, entire charge line; not a uniform PCL theorem")
    d,record
end
function policy_at(d::ClassData,j::Int)
    CM.actions_at(d.model,d.order,j)
end
function charged_data(classes,weights,counts,b::Q)
    K=length(classes);H=classes[1].model.H;A=length(classes[1].model.p)-1
    rho=Q.(counts)./sum(counts)
    events=sort!([(weights[k]*classes[k].roots[j],k,j) for k in 1:K for j in eachindex(classes[k].roots)];by=identity)
    right=ones(Int,K);left=copy(right);lam=zero(Q)
    low=sum(rho[k]*classes[k].cbar[1] for k in 1:K);high=low
    if low>b
        i=1;found=false
        while i<=length(events)
            lam=events[i][1];left=copy(right);high=low
            while i<=length(events) && events[i][1]==lam
                _,k,j=events[i];right[k]==j || error("Wrong class event order.")
                right[k]=j+1;i+=1
            end
            low=sum(rho[k]*classes[k].cbar[right[k]] for k in 1:K)
            if low<=b;found=true;break;end
        end
        found || error("No dual resource bracket.")
    end
    low<=b && (lam==0 || b<=high) || error("Invalid common dual bracket.")
    theta=lam==0 || high==low ? zero(Q) : (b-low)/(high-low)
    0<=theta<=1 || error("Invalid relaxation mixture.")
    bias=Vector{Q}[];gains=Q[];occ=Vector{Q}[]
    residual=zeros(Q,K,H+1,A+1);nextbias=similar(residual);fill!(nextbias,0)
    classrecords=Dict{String,Any}[];checks=0
    for k in 1:K
        d=classes[k];m=d.model;ar=policy_at(d,right[k]);al=policy_at(d,left[k])
        er=RA.excursion(m,ar);el=RA.excursion(m,al)
        er.hbar==d.hbar[right[k]] && er.cbar==d.cbar[right[k]] || error("Right scalar/block rates disagree.")
        el.hbar==d.hbar[left[k]] && el.cbar==d.cbar[left[k]] || error("Left scalar/block rates disagree.")
        vr=vcat(zero(Q),weights[k].*er.phi.+lam.*er.gamma)
        vl=vcat(zero(Q),weights[k].*el.phi.+lam.*el.gamma)
        vr==vl || error("Boundary charged biases disagree.")
        gain=weights[k]*er.hbar+lam*er.cbar
        gain==weights[k]*el.hbar+lam*el.cbar || error("Boundary charged gains disagree.")
        pr=AL.single_stationary(m,ar);pl=AL.single_stationary(m,al)
        push!(bias,vr);push!(gains,gain);push!(occ,(1-theta).*pr.+theta.*pl)
        nextbias[k,1,1]=(1-m.wR)*vr[2];residual[k,1,1]=nextbias[k,1,1]-gain
        residual[k,1,1]==0 || error("Class-specific zero-state equality failed.")
        for h in 1:H,a in 0:A
            en=(1-m.p[a+1])*vr[min(h+1,H)+1];nextbias[k,h+1,a+1]=en
            residual[k,h+1,a+1]=weights[k]*h+lam*m.c[a+1]+en-vr[h+1]-gain
            residual[k,h+1,a+1]>=0 || error("Negative class-specific exact Bellman slack.")
            a==ar[h] && residual[k,h+1,a+1]!=0 && error("Chosen relaxed action not optimal.")
            checks+=1
        end
        # Full flows of the left/right policies, not a statewise mixture with an assumed invariant law.
        for (actions,pi) in ((ar,pr),(al,pl))
            dest=zeros(Q,H+1);dest[1]+=pi[1]*m.wR;dest[2]+=pi[1]*(1-m.wR)
            for h in 1:H
                a=actions[h]+1;dest[1]+=pi[h+1]*m.p[a]
                dest[min(h+1,H)+1]+=pi[h+1]*(1-m.p[a])
            end
            dest==pi || error("Relaxation occupation witness flow failed.")
        end
        push!(classrecords,Dict{String,Any}("class"=>k,"left_policy_index"=>left[k],"right_policy_index"=>right[k],
            "left_actions"=>al,"right_actions"=>ar,"weight_exact"=>qs(weights[k]),
            "gain_exact"=>qs(gain),"bias_digest"=>value_digest(vr),"occupation_digest"=>value_digest(occ[end]),
            "left_age_rate_exact"=>qs(el.hbar),"right_age_rate_exact"=>qs(er.hbar),
            "left_resource_rate_exact"=>qs(el.cbar),"right_resource_rate_exact"=>qs(er.cbar),
            "flows_verified"=>true,"charged_action_slacks_nonnegative"=>true))
    end
    ell=sum(rho.*gains)-lam*b
    resource=(1-theta)*low+theta*high
    cost=sum(rho[k]*weights[k]*((1-theta)*classes[k].hbar[right[k]]+theta*classes[k].hbar[left[k]]) for k in 1:K)
    resource<=b && lam*(b-resource)==0 && cost==ell || error("Expected-resource primal-dual equality failed.")
    record=Dict{String,Any}("lambda_exact"=>qs(lam),"bound_per_project_exact"=>qs(ell),
        "budget_per_project_exact"=>qs(b),"theta_exact"=>qs(theta),"left_resource_exact"=>qs(high),
        "right_resource_exact"=>qs(low),"relaxed_cost_exact"=>qs(cost),"relaxed_resource_exact"=>qs(resource),
        "all_action_checks"=>checks,"primal_equals_dual"=>true,"classes"=>classrecords,
        "charge_convention"=>"smallest nonnegative maximizing charge; process all exact ties together")
    (lambda=lam,ell=ell,bias=bias,gains=gains,residual=residual,nextbias=nextbias,occupations=occ,record=record)
end
function from_classes(id,group,contrast,classes,weights,counts,b;max_population=1600)
    K=length(classes);K==length(weights)==length(counts) || error("Invalid class inventory.")
    all(>(0),weights) && all(>(0),counts) && b>=0 || error("Invalid weights, populations or budget.")
    H=classes[1].model.H;c=classes[1].model.c;A=length(c)-1
    all(d.model.H==H && d.model.c==c && length(d.model.p)==A+1 for d in classes) || error("Common cap and exact resource menu required.")
    D=charged_data(classes,weights,counts,Q(b));tables=Dict{Symbol,PS.Table}()
    for name in (:WHITTLE_K,:LAGRANGIAN_K,:MYOPIC_K)
        v=zeros(Q,K*H,A+1)
        for k in 1:K,h in 1:H,a in 1:A
            row=(k-1)*H+h;m=classes[k].model;s=min(h+1,H)
            if name==:WHITTLE_K
                v[row,a+1]=v[row,a]+(c[a+1]-c[a])*weights[k]*classes[k].mpi[h,a]
            elseif name==:LAGRANGIAN_K
                v[row,a+1]=(m.p[a+1]-m.p[1])*D.bias[k][s+1]-D.lambda*c[a+1]
            else
                v[row,a+1]=weights[k]*s*(m.p[a+1]-m.p[1])
            end
        end
        tables[name]=PS.compile(v;max_population=max_population)
    end
    tables[:LAGRANGIAN_NI]=tables[:LAGRANGIAN_K];tables[:LAGRANGIAN_FC]=tables[:LAGRANGIAN_K]
    indices=Q[weights[k]*classes[k].mpi[h,a] for k in 1:K for h in 1:H for a in 1:A]
    all(>(0),indices) || error("Nonpositive DAI in priority policy.")
    vals=sort!(unique(copy(indices)));rankmap=Dict(v=>i for (i,v) in enumerate(vals));rank=zeros(Int,K*H,A)
    for k in 1:K,h in 1:H,a in 1:A;rank[(k-1)*H+h,a]=rankmap[weights[k]*classes[k].mpi[h,a]];end
    units,tick=MA.resource_encoding(c)
    f=Family(String(id),String(group),Int(contrast),H,classes,weights,counts,Q(b),D.lambda,D.ell,
        D.bias,D.gains,D.residual,D.nextbias,D.occupations,tables,rank,trues(K*H,A),units,tick,copy(c))
    for scale in (1,2,4,8,16,32)
        L=scale*sum(counts)
        sum(scale*counts[k]*f.gains[k] for k in 1:K)-f.lambda*L*f.budget_per_project==L*f.bound_per_project || error("Proportional bound failed.")
    end
    f,D.record
end
function prepare(spec,c;progress=(label,k,n)->nothing)
    R=Int(spec["contrast"]);H=Int(spec["H"]);R in (1,4,16,64) || error("Unplanned time-scale contrast.")
    fast=AL.parseq(c["fast_d"]);slow=fast/R;weights=AL.parseq.(c["weights"]);counts=Int.(c["base_counts"])
    # Independent class computations use separate function frames; no shared result bindings.
    function one(d,label)
        try
            (value=prepare_class(make_model(d,H),c;progress=(k,n)->progress(label,k,n)), failure=nothing)
        catch err
            (value=nothing, failure=err)
        end
    end
    if R==1
        x=one(fast,"identical");outcomes=[x,x]
    else
        tasks=[Threads.@spawn one(d,label) for (d,label) in ((slow,"slow"),(fast,"fast"))]
        outcomes=[fetch(task) for task in tasks]
    end
    # Wait for every private class task; preserve a known resource-limit exception
    # instead of hiding it inside a generic TaskFailedException.
    for x in outcomes;x.failure===nothing || throw(x.failure);end
    pairs=[x.value for x in outcomes]
    classes=ClassData[x[1] for x in pairs]
    alpha=AL.parseq(spec["alpha"]);alpha in (1//4,1//2,4//5) || error("Unplanned load.")
    for cl in classes
        m=cl.model
        (1-m.wR)/(1-m.wR+m.p[end])*m.c[end]==21//32 || error("All-high resource changes along heterogeneity axis.")
    end
    b=alpha*Q(21//32)
    group="timescale_R$(R)" # all caps/loads share shocks; no independent-replication claim across variants
    f,dual=from_classes(spec["id"],group,R,classes,weights,counts,b)
    record=Dict{String,Any}("revision"=>REVISION,"id"=>f.id,"specification"=>Dict(spec),
        "base_counts"=>counts,"weights_exact"=>qs.(weights),"slow_d_exact"=>qs(slow),"fast_d_exact"=>qs(fast),
        "resource_menu_exact"=>qs.(f.resource),"resource_units"=>f.units,"resource_quantum_exact"=>qs(f.tick),
        "common_all_high_resource_exact"=>"21/32","classes"=>[pairs[k][2] for k in 1:2],"dual"=>dual,
        "score_digests"=>Dict(String(name)=>value_digest(f.scores[name].values) for name in (:WHITTLE_K,:LAGRANGIAN_K,:MYOPIC_K)),
        "model_scope"=>"constructed synchronization-gain resources, not battery-energy calibration",
        "full_PCL_family_certificate"=>false,"dai_verified"=>true,"proportional_scaling_verified"=>true,
        "no_required_Whittle_advantage"=>true,"runtime_report_dictionary_retained"=>false)
    f,record
end
mutable struct Allocators
    work::Dict{Symbol,PS.Workspace}
    priority_actions::Vector{Int}
    heap::Vector{NTuple{3,Int}}
    max_spend::Dict{Int,Int}
    limits::MA.Limits
end
function Allocators(f::Family,L::Int;limits=MA.Limits())
    Allocators(Dict(p=>PS.Workspace(f.scores[p],L) for p in MA.POLICIES if p!=:WHITTLE_P),
        zeros(Int,L),NTuple{3,Int}[],Dict{Int,Int}(),limits)
end
function decide!(w::Allocators,f::Family,policy::Symbol,types::Vector{Int},C::Int)
    policy in MA.POLICIES || error("Unknown policy.")
    if policy==:WHITTLE_P
        r=MA.priority!(w.priority_actions,w.heap,types,f.priority_rank,f.priority_positive,f.units,C)
        return w.priority_actions,r
    end
    domain=policy==:LAGRANGIAN_NI ? :ni : (policy==:LAGRANGIAN_FC ? :fc : :k)
    r=PS.allocate!(w.work[policy],f.scores[policy],types,f.units,C,domain;limits=w.limits)
    w.work[policy].actions,r
end
end
