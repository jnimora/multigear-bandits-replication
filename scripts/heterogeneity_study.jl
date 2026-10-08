module HeterogeneityStudy
using TOML, SHA, Serialization, Dates, LinearAlgebra
using ..HeterogeneityModel, ..HeterogeneitySimulation, ..ManyProjectAllocator
using ..RayleighLargeCap, ..PrecisionCapContinuation
const HM=HeterogeneityModel;const HS=HeterogeneitySimulation;const MA=ManyProjectAllocator
const LC=RayleighLargeCap;const IO0=PrecisionCapContinuation;const Q=Rational{BigInt}
const REVISION=HM.REVISION
sha(path)=open(io->bytes2hex(sha256(io)),path)
function source_map(root)
    d=Dict{String,String}()
    for top in ("src","scripts","test","configs","docs")
        base=joinpath(root,top);isdir(base) || continue
        for (dir,dirs,files) in walkdir(base)
            filter!(x->x!="__pycache__",dirs)
            for f in files
                (f==".DS_Store" || endswith(f,".pyc")) && continue
                p=joinpath(dir,f);islink(p) && error("Source symlink refused.")
                d[replace(relpath(p,root),'\\'=>'/')]=sha(p)
            end
        end
    end
    for f in ("Project.toml","Manifest.toml","julia-version","README.md",".gitignore","LICENSE")
        p=joinpath(root,f);isfile(p) && (d[f]=sha(p))
    end
    d
end
fingerprint(d)=bytes2hex(sha256(join([k*"\0"*d[k]*"\n" for k in sort!(collect(keys(d)))],"")))
function load_config(root)
    c=TOML.parsefile(joinpath(root,"configs","rayleigh_heterogeneity.toml"))
    c["revision"]==REVISION && c["production_enabled"]==false || error("No production authorization.")
    c["contrasts"]==[1,4,16,64] && c["load_multipliers"]==["1/4","1/2","4/5"] || error("Changed primary design.")
    c["weights"]==["1","1"] && c["base_counts"]==[25,25] || error("Changed heterogeneity controls.")
    String.(MA.POLICIES)==Tuple(c["policies"]) || error("Policy list changed.")
    c
end
function families(c)
    result=Dict{String,Any}[]
    for r in c["contrasts"],a in c["load_multipliers"]
        push!(result,Dict("id"=>"R$(r)_H$(c["primary_cap"])_alpha"*replace(a,"/"=>"_"),
             "contrast"=>r,"H"=>c["primary_cap"],"alpha"=>a,"role"=>"primary"))
    end
    for x in c["extra_caps"]
        r=x["contrast"];h=x["H"];a=x["alpha"]
        push!(result,Dict("id"=>"R$(r)_H$(h)_alpha"*replace(a,"/"=>"_"),"contrast"=>r,"H"=>h,"alpha"=>a,"role"=>"cap_diagnostic"))
    end
    length(unique(x["id"] for x in result))==length(result)==15 || error("Invalid family inventory.")
    result
end
function jobs(c)
    out=Dict{String,Any}[]
    for s in families(c)
        r=s["contrast"];B=max(Int(c["short_burn"]),Int(c["slow_sojourns_burn"])*8r);T=Int(c["horizon_burn_multiple"])*B
        wins=[Dict("name"=>"base","burn"=>B,"T"=>T)]
        if any(x->x["contrast"]==r && x["H"]==s["H"] && x["alpha"]==s["alpha"],c["window_checks"])
            push!(wins,Dict("name"=>"double_burn","burn"=>2B,"T"=>T))
            push!(wins,Dict("name"=>"double_horizon","burn"=>B,"T"=>2T))
        end
        for rep in 1:Int(c["replications"])
            L=Int(c["population"])
            push!(out,Dict{String,Any}("id"=>s["id"]*"_L$(L)_zero_r$(rep)","family"=>s["id"],"L"=>L,
                "rep"=>rep,"initial"=>"zero","replications"=>c["replications"],"windows"=>deepcopy(wins)))
        end
    end
    length(out)==60 && sum(length(j["windows"]) for j in out)==76 || error("Frozen 60-job/76-window inventory changed.")
    out
end
limits(c)=MA.Limits(Int(c["allocator"]["max_states"]),Int(c["allocator"]["max_updates"]),Float64(c["allocator"]["seconds"]))
plain(x::Rational)=HM.qs(Q(x))
plain(x::Symbol)=String(x)
plain(x::NamedTuple)=Dict(String(k)=>plain(getfield(x,k)) for k in keys(x))
plain(x::AbstractDict)=Dict(String(k)=>plain(v) for (k,v) in x)
plain(x::Tuple)=collect(plain.(x))
plain(x::AbstractVector)=[plain(v) for v in x]
plain(x)=x
function probe_states(f,L,pattern)
    pattern=="all_cap" && return fill(f.H,L)
    pattern=="stratified" && return [mod(37i,f.H+1) for i in 1:L]
    pattern=="competition" || error("Unknown deterministic probe.")
    # A slow middle-age cohort competes with younger fast projects; no observed outcome chooses the ages.
    [i<=L÷2 ? min(f.H,8*f.contrast+mod(i,7)) : min(f.H,8+mod(i,7)) for i in 1:L]
end
function probe_family(f,c;on_probe=x->nothing)
    rows=Dict{String,Any}[]
    for L in Int.(c["probes"]["populations"]),pattern in c["probes"]["patterns"]
        labels=HS.labels_for(f,L);state=probe_states(f,L,pattern)
        types=[state[i]==0 ? 0 : (labels[i]-1)*f.H+state[i] for i in 1:L]
        w=HM.Allocators(f,L;limits=limits(c));C=MA.budget_units(L*f.budget_per_project,f.tick,f.units,L)
        for policy in MA.POLICIES
            begun=time_ns()
            r=try
                a,info=HM.decide!(w,f,policy,types,C)
                dg=HS.diagnostics(w,f,types,a,C,info.resource_units)
                policy in (:LAGRANGIAN_NI,:LAGRANGIAN_FC) && dg.nonmaximal && error("NI domain mismatch.")
                policy==:LAGRANGIAN_FC && dg.attainable_gap_units!=0 && error("FC domain mismatch.")
                Dict{String,Any}("passed"=>true,"status"=>"complete","actions"=>copy(a),"diagnostics"=>plain(dg),"allocator"=>plain(info))
            catch err
                err isa MA.AllocationLimit || rethrow()
                Dict{String,Any}("passed"=>false,"status"=>"limit","error"=>sprint(showerror,err),"actions"=>Int[])
            end
            merge!(r,Dict("family"=>f.id,"L"=>L,"pattern"=>pattern,"policy"=>String(policy),"states"=>state,
                 "types"=>types,"budget_units"=>C,"diagnostic_seconds"=>(time_ns()-begun)/1e9))
            on_probe(r);push!(rows,r)
        end
    end
    rows
end
function commit(out,stem,record;kind="evidence",identity=stem)
    IO0.commit_record(out,stem,plain(record);kind=kind,identity=identity)
end
function prepare(req,c,root)
    out=req["output"];spec=only(filter(x->x["id"]==req["family"],families(c)))
    f,record=HM.prepare(spec,c;progress=(lab,k,n)->(k==1 || k%1024==0 || k==n ?
        (println("Class certificate ",lab," ",k,"/",n);flush(stdout)) : nothing))
    commit(out,"family",record;kind="family",identity=f.id)
    probes=probe_family(f,c;on_probe=r->begin
        stem="probe_L$(r["L"])_$(r["pattern"])_$(r["policy"])"
        commit(out,stem,r;kind="probe",identity=stem)
    end)
    ok=all(x->x["passed"],probes)
    gate=Dict{String,Any}("passed"=>ok,"status"=>(ok ? "PASS" : "REVIEW REQUIRED"),"family"=>f.id,
        "model_certified"=>true,"probe_count"=>length(probes),"probe_limits"=>count(x->!x["passed"],probes),
        "production_authorized"=>false,"retained_family_bytes"=>Base.summarysize(f))
    commit(out,"gate",gate;kind="gate",identity=f.id)
    ok || return gate
    path=joinpath(out,"runtime_cache.jls");ispath(path) && error("Cache exists.")
    header=Dict("revision"=>REVISION,"julia_version"=>string(VERSION),"source_fingerprint"=>req["source_fingerprint"],"family"=>f)
    serialize(path*".part",header);mv(path*".part",path;force=false)
    meta=Dict{String,Any}("passed"=>true,"family"=>f.id,"revision"=>REVISION,"julia_version"=>string(VERSION),
        "source_fingerprint"=>req["source_fingerprint"],"cache_sha256"=>sha(path),"retained_family_bytes"=>Base.summarysize(f),
        "large_reporting_dictionary_in_cache"=>false,"local_cache_not_portable"=>true)
    commit(out,"cache_metadata",meta;kind="cache_metadata",identity=f.id)
    gate
end
function load_cache(req)
    meta=TOML.parsefile(IO0.safe_input(req["cache_metadata"],req["cache_metadata_sha256"]))
    meta["revision"]==REVISION && meta["julia_version"]==string(VERSION) && meta["source_fingerprint"]==req["source_fingerprint"] || error("Incompatible cache metadata.")
    p=IO0.safe_input(req["cache_path"],meta["cache_sha256"])
    data=deserialize(p)
    data["revision"]==REVISION && data["julia_version"]==string(VERSION) && data["source_fingerprint"]==req["source_fingerprint"] || error("Cache header changed.")
    f=data["family"];f isa HM.Family && f.id==req["family"]==meta["family"] || error("Cache family changed.")
    f
end
function simulate(req,c,root)
    f=load_cache(req);j=only(filter(x->x["id"]==req["job"],jobs(c)));j["family"]==f.id || error("Wrong trajectory family.")
    out=req["output"];println("Heterogeneity job ",j["id"]," START");flush(stdout)
    function savewindow(r)
        r["planned_replications"]=j["replications"];r["trajectory_job_id"]=j["id"]
        r["job_id"]=j["id"]*"_"*r["window"];r["stage"]="heterogeneity_preflight"
        spec=only(filter(x->x["id"]==f.id,families(c)));r["alpha"]=spec["alpha"];r["role"]=spec["role"]
        commit(out,"window_"*r["window"],r;kind="window",identity=r["job_id"])
    end
    function failure(r)
        rr=copy(r);delete!(rr,"records");commit(out,"failure",rr;kind="failure",identity=j["id"])
    end
    result=HS.run_windows(f,j["L"],j["rep"],j["windows"];initial=j["initial"],master_seed=c["master_seed"],
        limits=limits(c),seconds=Float64(c["resource_limits"]["trajectory_seconds"]),
        collect_between_policies=true,retain_records=false,on_window=savewindow,on_failure=failure,
        heartbeat=(t,n)->(t%4096==0 || t==n ? (println(j["id"]," ",t,"/",n);flush(stdout)) : nothing))
    isempty(result["records"]) || error("Worker retained window dictionaries.")
    delete!(result,"records");result["job_id"]=j["id"];result["specification"]=j
    commit(out,"job",result;kind="job",identity=j["id"])
    Dict{String,Any}("passed"=>result["passed"],"status"=>(result["passed"] ? "PASS" : result["status"]=="limit" ? "REVIEW REQUIRED" : "FAILED"),
        "job_id"=>j["id"],"completed_windows"=>result["completed_windows"],"production_authorized"=>false)
end
end
