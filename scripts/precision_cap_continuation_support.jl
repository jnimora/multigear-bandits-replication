"""Memory-bounded continuation of the frozen precision/cap pilot.
Only orchestration, workspace ownership and record retention change. Scientific
configurations, seeds, allocator limits and exact ledger arithmetic are inherited.
"""
module PrecisionCapContinuation
using TOML, SHA, Dates, Serialization
using ..ManyProjectAllocator, ..ManyProjectModel, ..PrecisionCapModel
using ..PrecisionCapSimulation, ..PrecisionCapStudy, ..RayleighLargeCap
const MA=ManyProjectAllocator; const MM=ManyProjectModel; const CM=PrecisionCapModel
const PC=PrecisionCapSimulation; const ST=PrecisionCapStudy; const LC=RayleighLargeCap
const REVISION="precision-cap-memory-continuation-v1"
sha(path)=open(io->bytes2hex(sha256(io)),path)

# Deliberately narrow: only nondeterministic diagnostic durations/storage are
# omitted when comparing a replayed window/family to already accepted records.
function scientific(x)
    if x isa AbstractDict
        Dict{String,Any}(String(k)=>scientific(v) for (k,v) in x
            if !occursin("seconds",String(k)) && !(String(k) in
                ("retained_table_bytes","retained_family_bytes","diagnostic_preparation_bytes")))
    elseif x isa AbstractArray
        map(scientific,x)
    else
        x
    end
end
function safeid(id)
    occursin(r"^[A-Za-z0-9][A-Za-z0-9_-]*$",String(id)) || error("Unsafe record identity.")
    String(id)
end
function safe_input(path,expected)
    p=abspath(path);isfile(p) && !islink(p) || error("Missing or linked input: "*p)
    q=p
    while dirname(q)!=q
        islink(q) && error("Input path contains symlink: "*p);q=dirname(q)
    end
    sha(p)==expected || error("Input checksum mismatch: "*p)
    p
end
"""Write two records, then a last-written commit marker. Incomplete pairs are
never accepted by the continuation planner. Existing records are never replaced.
"""
function commit_record(folder,stem,d;kind,identity)
    safeid(stem);mkpath(folder)
    targets=[joinpath(folder,stem*ext) for ext in (".toml",".json",".commit.toml",".commit.json")]
    any(ispath,targets) && error("Refusing record overwrite: "*stem)
    temp=mktempdir(folder;prefix=".checkpoint_",cleanup=false)
    try
        base=joinpath(temp,stem);LC.write_record(base,d)
        marker=Dict{String,Any}("revision"=>REVISION,"kind"=>kind,"identity"=>identity,
            "toml_sha256"=>sha(base*".toml"),"json_sha256"=>sha(base*".json"),"committed"=>true)
        LC.write_record(base*".commit",marker)
        for ext in (".toml",".json",".commit.toml",".commit.json")
            mv(base*ext,joinpath(folder,stem*ext);force=false)
        end
        marker
    finally
        rm(temp;recursive=true,force=true)
    end
end
function expected_jobs(c,stage)
    stage=="precision" && return ST.precision_jobs(c)
    stage=="caps" || error("No new study or production stage is admitted.")
    reduce(vcat,[ST.cap_jobs(c,H) for H in c["cap"]["caps"]])
end
function family_spec(c,stage,id)
    if stage=="precision"
        return only(filter(s->s["id"]==id,c["family"]))
    end
    specs=[ST.cap_spec(H,c) for H in c["cap"]["caps"]]
    only(filter(s->s["id"]==id,specs))
end
function cap_probes(f,c)
    rows=Dict{String,Any}[];L=50;H=f.single.model.H;labels=PC.labels_for(f,L)
    for pattern in ("all_cap","stratified")
        state=pattern=="all_cap" ? fill(H,L) : [mod(37i,H+1) for i in 1:L]
        types=[state[i]==0 ? 0 : (labels[i]-1)*H+state[i] for i in 1:L]
        C=MA.budget_units(L*f.budget_per_project,f.tick,f.units,L)
        w=CM.Allocators(f,L;limits=ST.limits(c))
        for p in MA.POLICIES
            t=time_ns();a,info=CM.decide!(w,f,p,types,C)
            d=PC.diagnostics(w,f,types,a,C,info.resource_units)
            p in (:LAGRANGIAN_NI,:LAGRANGIAN_FC) && d.nonmaximal && error("NI probe domain violation.")
            p==:LAGRANGIAN_FC && d.attainable_gap_units!=0 && error("FC probe domain violation.")
            push!(rows,Dict{String,Any}("family"=>f.id,"L"=>L,"pattern"=>pattern,"policy"=>String(p),
                "passed"=>true,"status"=>"complete","actions"=>copy(a),"engine"=>String(info.engine),
                "work"=>info.work,"peak_states"=>info.peak_states,"resource_units"=>info.resource_units,
                "nonmaximal"=>d.nonmaximal,"attainable_gap_units"=>d.attainable_gap_units,
                "diagnostic_elapsed_seconds"=>(time_ns()-t)/1e9))
        end
    end
    length(rows)==12 || error("Incomplete cap-admission probes.")
    rows
end
function check_family_scaling(f)
    for scale in (1,2,4,8,16,32)
        L=50*scale;labels=PC.labels_for(f,L)
        all(count(==(k),labels)==scale*f.base_counts[k] for k in eachindex(f.base_counts)) || error("Class scaling changed.")
        sum(f.gains[k]*count(==(k),labels) for k in eachindex(f.weights))-f.lambda*L*f.budget_per_project==L*f.bound_per_project || error("Bound is not proportional.")
    end
end
function prepare_cache(req,c,root)
    out=req["output"];mkpath(out);stage=req["stage"];id=safeid(req["family"])
    spec=family_spec(c,stage,id)
    println("Prepare/certify cache: ",stage," / ",id);flush(stdout)
    f=stage=="precision" ? MM.prepare_family(spec) : CM.prepare(spec,
        TOML.parsefile(joinpath(root,"configs","rayleigh_largecap.toml"));
        seconds=Float64(c["cap"]["preparation_seconds_per_cap"]),
        progress=(k,n)->(k%1024==0 ? (println("Certificate ",k,"/",n);flush(stdout)) : nothing))
    check_family_scaling(f)
    for old in get(req,"existing_family",Any[])
        p=safe_input(old["path"],old["sha256"])
        scientific(f.record)==scientific(TOML.parsefile(p)) || error("Prepared family differs from accepted historical scientific record.")
    end
    probes=stage=="caps" ? cap_probes(f,c) : Dict{String,Any}[]
    commit_record(req["data_output"],"family_"*id,f.record;kind="family",identity=id)
    gate=Dict{String,Any}("passed"=>true,"status"=>"complete","family"=>id,"stage"=>stage,
        "proportional_scaling_verified"=>true,"all_original_cap_probes"=>stage=="caps",
        "probes"=>probes,"production_authorized"=>false)
    commit_record(req["data_output"],"gate_"*id,gate;kind="gate",identity=id)
    # This private acceleration cache is not long-term scientific evidence.
    # Only bytes written by this process, hashed before load under the same
    # Julia version and exact source fingerprint, may later be deserialized.
    path=joinpath(out,"family_cache.jls");ispath(path) && error("Cache already exists.")
    temp=path*".part"
    serialize(temp,Dict("revision"=>REVISION,"julia_version"=>string(VERSION),
        "source_fingerprint"=>req["source_fingerprint"],"family"=>f))
    mv(temp,path;force=false)
    meta=Dict{String,Any}("passed"=>true,"revision"=>REVISION,"family"=>id,"stage"=>stage,
        "julia_version"=>string(VERSION),"source_fingerprint"=>req["source_fingerprint"],
        "cache_sha256"=>sha(path),"cache_file"=>"family_cache.jls","retained_family_bytes"=>Base.summarysize(f),
        "cache_is_local_only"=>true,"production_authorized"=>false)
    LC.write_record(joinpath(out,"cache_metadata"),meta)
    println("Family cache and gate: PASS");flush(stdout)
    meta
end
function load_cache(req)
    meta=TOML.parsefile(safe_input(req["cache_metadata"],req["cache_metadata_sha256"]))
    meta["passed"] && meta["revision"]==REVISION && meta["julia_version"]==string(VERSION) || error("Incompatible local cache metadata.")
    meta["source_fingerprint"]==req["source_fingerprint"] || error("Cache source differs.")
    p=safe_input(req["cache_path"],meta["cache_sha256"])
    d=deserialize(p)
    d["revision"]==REVISION && d["julia_version"]==string(VERSION) && d["source_fingerprint"]==req["source_fingerprint"] || error("Cache header differs.")
    f=d["family"];f.id==req["family"]==meta["family"] || error("Cache has wrong family.")
    f,meta
end
function replay_comparison(record,previous)
    scientific(record)==scientific(previous) || error("Restarted trajectory disagrees with previously committed window; original kept.")
    true
end
function one_trajectory(f,j,c,req,postlock;collect_between_policies::Bool=false,phase_event=(phase,policy)->nothing)
    out=req["data_output"];stage=req["stage"]
    old=Dict(x["identity"]=>x for x in get(req,"existing_windows",Any[]))
    checked=String[];written=String[]
    onwin=function(r)
        r["planned_replications"]=j["replications"];r["job_id"]=j["id"]*"_"*r["window"]
        r["trajectory_job_id"]=j["id"];r["stage"]=stage
        if haskey(old,r["job_id"])
            olditem=old[r["job_id"]];p=safe_input(olditem["path"],olditem["sha256"])
            replay_comparison(r,TOML.parsefile(p));push!(checked,r["job_id"])
        else
            commit_record(out,"rep_"*r["job_id"],r;kind="window",identity=r["job_id"])
            push!(written,r["job_id"])
        end
    end
    onfail=function(r)
        d=copy(r);delete!(d,"records")
        commit_record(out,"partial_"*j["id"],d;kind="partial",identity=j["id"])
    end
    ST.progress(req["output"],"Continue "*j["id"]*" from original seed START")
    stagecfg=c[stage=="caps" ? "cap" : "precision"]
    result=PC.run_windows(f,j["L"],j["rep"],j["windows"];initial=j["initial"],
        master_seed=c["master_seed"],limits=ST.limits(c),seconds=Float64(stagecfg["max_replication_seconds"]),
        retain_records=false,postprocess_lock=postlock,collect_between_policies=collect_between_policies,
        phase_event=phase_event,on_window=onwin,on_failure=onfail,
        heartbeat=(t,n)->(t%10000==0 ? ST.progress(req["output"],j["id"]*" period $(t)/$(n)") : nothing))
    isempty(result["records"]) || error("Streaming worker retained window dictionaries.")
    delete!(result,"records");result["job_id"]=j["id"]
    commit_record(out,"job_"*j["id"],result;kind="job",identity=j["id"])
    receipt=Dict{String,Any}("job_id"=>j["id"],"stage"=>stage,"status"=>result["status"],
        "passed"=>result["passed"],"new_windows"=>written,"replayed_windows_verified"=>checked,
        "allocator_bundles_per_trajectory"=>1,"window_results_retained_in_worker"=>false,
        "restarted_from_original_seed"=>true)
    ST.progress(req["output"],j["id"]*" "*result["status"])
    receipt
end
function admission(f,req,jobs)
    maximum_tasks=Int(req["max_tasks"])
    1<=maximum_tasks<=2 || error("Memory policy allows at most two simultaneous trajectories.")
    familybytes=Base.summarysize(f)
    free=Int128(Sys.free_memory());budget=min(Int128(6)*1024^3,free)-familybytes-Int128(1024)^3
    # Reservation/admission estimates complement the external RSS monitor; they
    # are neither a hard per-task bound nor a guarantee against transient peaks.
    slots=Int(max(0,min(Int128(2),fld(budget,Int128(2)*1024^3))))
    high=maximum(j["L"] for j in jobs)>=800 || f.single.model.H>=512
    n=min(maximum_tasks,Threads.nthreads(:default),length(jobs),slots,high ? 1 : 2)
    n>=1 || throw(MA.AllocationLimit("Insufficient memory admission for one trajectory; no work launched."))
    n,familybytes
end
function run_batch(req,c,root)
    f,meta=load_cache(req)
    planned=Dict(j["id"]=>j for j in expected_jobs(c,req["stage"]))
    ids=String.(req["job_ids"]);length(ids)==length(unique(ids)) && !isempty(ids) || error("Empty/duplicated batch job identities.")
    all(haskey(planned,id) for id in ids) || error("Unknown trajectory identity.")
    jobs=[planned[id] for id in ids]
    all(j["family"]==f.id for j in jobs) || error("Mixed families in a worker.")
    length(jobs)<=4 && (req["stage"]!="caps" || length(jobs)<=2) || error("Oversized short-lived worker batch.")
    f.single.model.H>=512 && length(jobs)>1 && error("Large-cap workers run one trajectory then exit.")
    n,fbytes=admission(f,req,jobs);postlock=n==1 ? nothing : Base.Semaphore(1);results=Dict{String,Any}[]
    ST.progress(req["output"],"Admitted $(n) trajectory task(s); family bytes=$(fbytes). Exact postprocessing serialized.")
    for offset in 1:n:length(jobs)
        selected=jobs[offset:min(offset+n-1,length(jobs))]
        # Only invocations capture private locals. No collector is captured.
        tasks=[Threads.@spawn one_trajectory(f,j,c,req,postlock) for j in selected]
        wave=[fetch(t) for t in tasks];append!(results,wave)
        tasks=nothing;wave=nothing
        # Orchestration collection, outside allocation measurements. All tasks
        # in this wave have exited; no result dictionaries survive in receipts.
        GC.gc(true)
        any(!r["passed"] for r in results) && break
    end
    complete=length(results)==length(jobs) && all(r["passed"] for r in results)
    status=complete ? "PASS" : (any(r["status"]=="error" for r in results) ? "FAILED" : "REVIEW REQUIRED")
    r=Dict{String,Any}("passed"=>complete,"status"=>status,"revision"=>REVISION,"stage"=>req["stage"],
        "family"=>f.id,"planned_jobs"=>ids,"job_receipts"=>results,"admitted_tasks"=>n,
        "retained_family_bytes"=>fbytes,"worker_batch_maximum"=>length(jobs),"postprocessing_serialized"=>true,
        "original_rng_and_windows_preserved"=>true,"production_authorized"=>false)
    LC.write_record(joinpath(req["output"],"worker_review"),r)
    r
end
end # module
