# Included within RayleighTiming. All routines in this file run OUTSIDE timers.
# No certificate, exact path or expected index is an input to a timed selector.

function model_record(m)
    Dict{String,Any}("H"=>m.H,"A"=>length(m.p)-1,"source_states"=>m.nsrc,
        "wR"=>string(m.wR),"wT"=>string(m.wT),"chi"=>string(m.chi),
        "kappa"=>string(m.kappa),"epsilon"=>string(m.epsilon),
        "success_exact"=>string.(m.success),"reset_exact"=>string.(m.p),
        "resource_exact"=>string.(m.c),"criterion"=>"average",
        "resource_scope"=>"constructed synchronization-gain menu; NOT calibrated battery energy")
end

"""Check an affine Bellman advantage b+s*lambda >= 0 on a closed interval.
Nothing is -infinity for lo and +infinity for hi. Singleton intervals retained.
"""
function affine_nonnegative(b::Q,s::Q,lo,hi)
    lo!==nothing && hi!==nothing && lo>hi && return false
    if lo===nothing
        s>0 && return false
    elseif b+s*lo<0
        return false
    end
    if hi===nothing
        s<0 && return false
    elseif b+s*hi<0
        return false
    end
    lo===nothing && hi===nothing && b<0 && return false
    true
end

"""Exact Bellman certificate for the WHOLE charge line using the computed chain.
At policy S, bias=phi+lambda*gamma. The advantage of any admissible gear a
against S(h)=b is -(p[a]-p[b])*phi[h+] + lambda*(c[a]-c[b]-(p[a]-p[b])*gamma[h+]).
Checking its endpoints/tail slope certifies every lambda in that interval.
All actions, including skipped/nonadjacent gears, are checked. This does NOT
claim to enumerate all stationary policies. Combined with policy monotonicity,
it independently certifies a global optimal threshold chain for the fixed model.
"""
function path_bellman_certificate(m,exact)
    H=m.H;A=length(m.p)-1;K=H*A
    length(exact.order)==K && length(exact.assignments)==K && issorted(exact.assignments) ||
        error("Incomplete/nonmonotone exact path; no charge-line certificate.")
    actions=fill(A,H);intervals=Dict{String,Any}[];checks=0;ties=0
    for k in 0:K
        lo=k==0 ? nothing : exact.assignments[k]
        hi=k==K ? nothing : exact.assignments[k+1]
        e=excursion(m,actions)
        for h in 1:H
            b=actions[h]+1;s=min(h+1,H)
            for a in 0:A
                a==actions[h] && continue
                dp=m.p[a+1]-m.p[b]
                intercept=-dp*e.phi[s]
                slope=m.c[a+1]-m.c[b]-dp*e.gamma[s]
                affine_nonnegative(intercept,slope,lo,hi) ||
                    error("Exact Bellman inequality failed at policy $k, age $h, action $a.")
                checks+=1
            end
        end
        lo!==nothing && hi!==nothing && lo==hi && (ties+=1)
        push!(intervals,Dict("policy_index"=>k,"actions"=>copy(actions),
            "lower"=>(lo===nothing ? "-Inf" : string(lo)),
            "upper"=>(hi===nothing ? "+Inf" : string(hi)),
            "holding_cost_rate"=>string(e.hbar),"resource_rate"=>string(e.cbar)))
        if k<K
            h,a=exact.order[k+1]
            actions[h]==a && (h==1 || actions[h-1]<a) || error("Infeasible exact chain step.")
            actions[h]-=1
        end
    end
    all(iszero,actions) || error("Exact chain did not end all-passive.")
    Dict{String,Any}("passed"=>true,"all_real_charges"=>true,"all_admissible_actions"=>true,
        "policy_intervals"=>K+1,"affine_inequalities"=>checks,"singleton_intervals"=>ties,
        "all_policy_enumeration"=>false,"intervals"=>intervals)
end

function certify_case(spec;limit=10000)
    spec["curvature_rule"]=="certified_dyadic" || error("This preflight requires a positive certified menu.")
    base=from_success(Int(spec["H"]);nsrc=Int(spec["source_states"]),wR=spec["wR"],
        success=spec["success"],kappa=spec["kappa"],chi=spec["chi"],epsilon=0)
    proof=curvature_bound(base;limit=limit)
    m=from_success(base.H;nsrc=base.nsrc,wR=base.wR,success=base.success,
        kappa=base.kappa,chi=base.chi,epsilon=proof.epsilon)
    m.epsilon>0 && (proof.upper===nothing || m.epsilon<proof.upper) || error("Invalid positive curvature.")
    cert=exact_certificate(m;limit=limit)
    cert["passed"] || error("Whole-family certificate failed: " * repr(cert["failures"]))
    scalar=exact_downshift(m;method=:scalar);block=exact_downshift(m;method=:block)
    for field in (:mpi,:order,:f,:g,:assignments,:hbar,:cbar)
        getproperty(scalar,field)==getproperty(block,field) || error("Exact scalar/block mismatch: $field")
    end
    bellman=path_bellman_certificate(m,scalar)
    fm=dense_model(m;exact=false);stored=ExactMultiGearModel(fm);exactmodel=dense_model(m)
    stored.P==exactmodel.P && stored.h==exactmodel.h && stored.c==exactmodel.c ||
        error("Float64 arrays are not the same exact certified model; no rounding-based certificate transfer.")
    A=length(m.p)-1;H=m.H
    groups=Dict{Q,Vector{Tuple{Int,Int}}}()
    for h in 1:H,a in 1:A;push!(get!(groups,scalar.mpi[h,a],Tuple{Int,Int}[]),(h,a));end
    ties=[Dict("value_exact"=>string(v),"age_gear_pairs"=>[collect(x) for x in groups[v]])
          for v in sort!(collect(keys(groups))) if length(groups[v])>1]
    Dict{String,Any}("id"=>String(spec["id"]),"passed"=>true,"dai_verified"=>true,
        "model"=>model_record(m),"specification"=>Dict(spec),"whole_family_certificate"=>cert,
        "curvature_open_upper"=>(proof.upper===nothing ? "+Inf" : string(proof.upper)),
        "curvature_constraint_count"=>proof.constraints,"curvature_structural_zeros"=>proof.structural_zeros,
        "bellman_path_certificate"=>bellman,"float64_model_exactly_identical"=>true,
        "threshold_policy_count"=>proof.threshold_policies,"scope"=>"finite-cap certificate, not a uniform cap theorem",
        "strict_gear_separation_at_all_ages"=>all(scalar.mpi[h,a]>scalar.mpi[h,a+1] for h in 1:H for a in 1:A-1),
        "exact_tie_groups"=>ties,"dai_exact_by_age"=>[string.(scalar.mpi[h,:]) for h in 1:H],
        "path"=>[Dict("age"=>h,"upper_gear"=>a,"index"=>string(scalar.assignments[k]),
            "f"=>string(scalar.f[k]),"g"=>string(scalar.g[k])) for (k,(h,a)) in enumerate(scalar.order)])
end

function certificate_model(record)
    spec=record["specification"];r=record["model"]
    m=from_success(Int(spec["H"]);nsrc=Int(spec["source_states"]),wR=spec["wR"],
        success=spec["success"],kappa=spec["kappa"],chi=spec["chi"],
        epsilon=replace(String(r["epsilon"]),"//"=>"/"))
    model_record(m)==r || error("Certified source primitives changed.")
    m
end

function exact_reference_record(record)
    H=Int(record["model"]["H"]);A=Int(record["model"]["A"])
    parseq(x)=parse_q(replace(String(x),"//"=>"/"))
    rows=record["path"]
    (order=[(Int(x["age"])+1,Int(x["upper_gear"])) for x in rows],
     ratio=[parseq(x["index"]) for x in rows], f=[parseq(x["f"]) for x in rows],
     g=[parseq(x["g"]) for x in rows],
     mpi=[parseq(record["dai_exact_by_age"][h][a]) for h in 1:H,a in 1:A])
end
