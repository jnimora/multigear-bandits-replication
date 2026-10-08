"""Additive execution adapter for the frozen confirmation manifests.
The fixed model, policies, transition and ledger routines are unchanged. The
large-core score-comparison revision has an explicit historical-source/cache
transition; old evidence retains its original fingerprints.
"""
module HeterogeneityConfirmation
using TOML, SHA, Dates, Serialization, LinearAlgebra
using ..HeterogeneityModel, ..HeterogeneitySimulation, ..HeterogeneityStudy
using ..ManyProjectAllocator, ..ManyProjectSimulation, ..PrecisionCapContinuation
const HM=HeterogeneityModel; const HS=HeterogeneitySimulation; const ST=HeterogeneityStudy
const MA=ManyProjectAllocator; const MS=ManyProjectSimulation; const PRE=PrecisionCapContinuation
const Q=Rational{BigInt}
const REVISION="heterogeneity-confirmation-execution-v1"
const MAIN_STAGES=("main","cap_diagnostic","start_diagnostic")
sha(path)=ST.sha(path)
plain(x)=ST.plain(x)
safeid(x)=PRE.safeid(x)

function load_plan(root)
    cfg=joinpath(root,"configs")
    binding=TOML.parsefile(joinpath(cfg,"heterogeneity_confirmation_binding.toml"))
    binding["revision"]==REVISION || error("Different confirmation revision.")
    for (name,h) in binding["design_files"]
        PRE.safe_input(joinpath(cfg,"heterogeneity_confirmation",name),h)
    end
    p=TOML.parsefile(PRE.safe_input(joinpath(cfg,"heterogeneity_confirmation_native.toml"),binding["native_manifest_sha256"]))
    d=p["design"];s=p["summary"]
    d["revision"]=="heterogeneity-confirmation-design-v1" && d["master_seed"]==2026092502 && d["replications"]==32 || error("Frozen seed/sample policy changed.")
    collect(String.(MA.POLICIES))==d["policies"] || error("Policy inventory changed.")
    length(p["families"])==15 && length(p["jobs"])==896 && length(p["probes"])==450 || error("Incomplete frozen inventory.")
    length(unique(j["id"] for j in p["jobs"]))==896 || error("Duplicate job identities.")
    sum(length(j["windows"]) for j in p["jobs"])==1088 || error("Window inventory differs.")
    count(j->j["stage"] in MAIN_STAGES,p["jobs"])==576 || error("Main/diagnostic inventory differs.")
    count(j->j["stage"]=="scaling",p["jobs"])==320 || error("Scaling inventory differs.")
    d["execution_requirements"]["fixed_trajectory_wall_limit_seconds"]==7200 || error("Trajectory ceiling changed.")
    for (name,h) in binding["numerical_baseline_sha256"]
        PRE.safe_input(joinpath(root,name),h)
    end
    p["design_fingerprint"]=ST.fingerprint(binding["design_files"])
    p["native_sha256"]=binding["native_manifest_sha256"]
    p["native_path"]=joinpath(cfg,"heterogeneity_confirmation_native.toml")
    p
end

function solver_config(root,plan)
    b=TOML.parsefile(joinpath(root,"configs","heterogeneity_confirmation_binding.toml"))
    c=TOML.parsefile(PRE.safe_input(joinpath(root,"configs","rayleigh_heterogeneity.toml"),b["accepted_numerical_config_sha256"]))
    e=plan["design"]["execution_requirements"]
    c["allocator"]["max_states"]==e["allocator_max_states"]==200000 || error("Allocator state limit changed.")
    c["allocator"]["max_updates"]==e["allocator_max_work"]==20000000 || error("Allocator work limit changed.")
    c["allocator"]["seconds"]==e["allocator_seconds"]==30 || error("Allocator time limit changed.")
    c["certificate"]["seconds_per_case"]==e["offline_class_certificate_seconds"]==600 || error("Offline class limit changed.")
    c["certificate"]["exact_max_H"]==e["offline_exact_certificate_max_H"]==2048 || error("Offline cap changed.")
    c["certificate"]["maximum_exact_policy_fallbacks"]==e["offline_max_exact_policy_fallbacks"]==4096 || error("Offline fallback count changed.")
    # Only this NEW protocol's predeclared trajectory duration and seed differ.
    c["resource_limits"]["trajectory_seconds"]=Float64(e["fixed_trajectory_wall_limit_seconds"])
    c["resource_limits"]["preparation_seconds"]=Float64(e["whole_family_preparation_seconds"])
    c["master_seed"]=plan["design"]["master_seed"]
    c["replications"]=32
    c
end

classid(cl)="d"*replace(cl["u"],"/"=>"_")*"_H$(cl["H"])"
function class_specs(plan)
    found=Dict{String,Any}()
    for f in plan["families"],cl in f["classes"]
        s=Dict{String,Any}(k=>deepcopy(cl[k]) for k in ("u","wR","p","H","decoding_success","kappa","epsilon"))
        s["resources"]=copy(f["resources"]);s["id"]=classid(cl)
        haskey(found,s["id"]) && found[s["id"]]!=s && error("Conflicting class primitive identity.")
        found[s["id"]]=s
    end
    found
end

function bound_record(req)
    Dict{String,Any}("revision"=>REVISION,"design_fingerprint"=>req["design_fingerprint"],
        "source_fingerprint"=>req["source_fingerprint"])
end
function commit(req,stem,r;kind="evidence",identity=stem)
    # plain preserves a homogeneous input's narrow value type (for example Bool).
    # Metadata also contains strings, so widen only this new top-level dictionary.
    # The caller's record and the strict checkpoint writer remain unchanged.
    d=Dict{String,Any}(plain(r))
    merge!(d,bound_record(req))
    ST.commit(req["output"],stem,d;kind=kind,identity=identity)
end
function read_ref(ref)
    base=joinpath(ref["folder"],ref["stem"])
    marker=TOML.parsefile(PRE.safe_input(base*".commit.toml",ref["marker_toml_sha256"]))
    PRE.safe_input(base*".commit.json",ref["marker_sha256"])
    marker["committed"] && marker["identity"]==ref["identity"] && marker["kind"]==ref["kind"] || error("Wrong committed evidence.")
    marker["json_sha256"]==ref["json_sha256"] && marker["toml_sha256"]==ref["toml_sha256"] || error("Marker binding differs.")
    PRE.safe_input(base*".json",ref["json_sha256"])
    TOML.parsefile(PRE.safe_input(base*".toml",ref["toml_sha256"]))
end
function same_binding(r,req)
    r["revision"]==REVISION && r["source_fingerprint"]==req["source_fingerprint"] && r["design_fingerprint"]==req["design_fingerprint"] || error("Evidence/cache from another source or design.")
    true
end
function write_cache(req,identity,kind,value)
    out=req["output"];path=joinpath(out,"runtime_cache.jls")
    ispath(path) && error("Refusing cache overwrite.")
    header=merge(bound_record(req),Dict{String,Any}("julia_version"=>string(VERSION),"architecture"=>string(Sys.ARCH),
        "kernel"=>string(Sys.KERNEL),"word_size"=>Sys.WORD_SIZE,"identity"=>identity,"cache_kind"=>kind,"value"=>value))
    serialize(path*".part",header);mv(path*".part",path;force=false)
    metadata=merge(bound_record(req),Dict{String,Any}("passed"=>true,"julia_version"=>string(VERSION),
        "architecture"=>string(Sys.ARCH),"kernel"=>string(Sys.KERNEL),"word_size"=>Sys.WORD_SIZE,
        "identity"=>identity,"cache_kind"=>kind,"cache_sha256"=>sha(path),"cache_file"=>"runtime_cache.jls",
        "retained_numerical_bytes"=>Base.summarysize(value),"report_dictionary_in_runtime_object"=>false,
        "local_acceleration_cache_not_portable"=>true))
    commit(req,"cache_metadata",metadata;kind="cache_metadata",identity=identity)
    metadata
end
function load_cache(ref,req,identity,kind)
    m=read_ref(ref);same_binding(m,req)
    m["passed"] && m["identity"]==identity && m["cache_kind"]==kind || error("Wrong cache identity.")
    m["julia_version"]==string(VERSION) && m["architecture"]==string(Sys.ARCH) && m["kernel"]==string(Sys.KERNEL) && m["word_size"]==Sys.WORD_SIZE || error("Incompatible local binary cache.")
    # Deserialize ONLY own previously committed bytes under the exact source/version.
    file=PRE.safe_input(joinpath(ref["folder"],m["cache_file"]),m["cache_sha256"])
    d=deserialize(file);same_binding(d,req)
    d["julia_version"]==string(VERSION) && d["identity"]==identity && d["cache_kind"]==kind && d["architecture"]==string(Sys.ARCH) && d["kernel"]==string(Sys.KERNEL) && d["word_size"]==Sys.WORD_SIZE || error("Cache header disagrees with authenticated metadata.")
    d["value"]
end

function rebind_family(req,plan,root)
    transition=get(req,"cache_transition","scaling_bounds_transition")
    transition in ("scaling_bounds_transition","scaling_memory_transition","scaling_memory_measurement_transition","scaling_collection_transition") || error("Unknown cache transition selector.")
    t=TOML.parsefile(joinpath(root,"configs",transition*".toml"))
    expected_revision=transition=="scaling_bounds_transition" ? "scaling-dyadic-score-bounds-v1" :
        transition=="scaling_memory_transition" ? "scaling-packed-limb-memory-v1" :
        transition=="scaling_memory_measurement_transition" ? "scaling-os-memory-measurement-v1" : "scaling-period-collection-v1"
    t["revision"]==expected_revision && t["design_fingerprint"]==plan["design_fingerprint"] || error("Unknown cache transition.")
    id=req["identity"];haskey(t["cache_families"],id) || error("Unapproved cache family.")
    expected=t["cache_families"][id];ref=req["old_family_cache"]
    m=read_ref(ref)
    ref["json_sha256"]==expected["cache_metadata_sha256"] || error("Unreviewed old cache metadata.")
    m["source_fingerprint"]==t["old_source_fingerprint"] || error("Unreviewed cache source.")
    # In the memory transition the pinned metadata digest above authenticates its
    # recorded cache SHA256. The ordinary loader still verifies actual bytes.
    (!haskey(expected,"cache_sha256") || m["cache_sha256"]==expected["cache_sha256"]) || error("Unreviewed binary cache.")
    oldreq=copy(req);oldreq["source_fingerprint"]=t["old_source_fingerprint"]
    # load_cache still checks known bytes, version, architecture, identity, design
    # and the OLD header. It never accepts a new source with an old fingerprint.
    family=load_cache(ref,oldreq,id,"family")
    family isa HM.Family && family.id==id || error("Wrong rebound runtime layout.")
    fr=read_ref(req["family_record"])
    req["family_record"]["json_sha256"]==expected["family_record_sha256"] || error("Unreviewed family mathematics.")
    for policy in (:WHITTLE_K,:LAGRANGIAN_K,:MYOPIC_K)
        HM.value_digest(family.scores[policy].values)==expected["score_digests"][String(policy)]==fr["score_digests"][String(policy)] || error("Cached score digest changed.")
    end
    metadata=write_cache(req,id,"family",family)
    # No field in family is changed; no model, index, bias or dual is recomputed.
    result=Dict{String,Any}("passed"=>true,"status"=>"PASS","identity"=>id,
        "transition_revision"=>t["revision"],"original_cache"=>ref,"new_cache_metadata"=>metadata,
        "runtime_numerical_object_preserved"=>true,"source_fingerprint_before"=>t["old_source_fingerprint"],
        "source_fingerprint_after"=>req["source_fingerprint"],"no_trajectory_replayed"=>true)
    commit(req,"source_rebinding",result;kind="source_rebinding",identity=id)
    result
end

function prepare_class(req,plan,c)
    spec=class_specs(plan)[req["identity"]]
    m=HM.make_model(HM.AL.parseq(spec["u"]),Int(spec["H"]))
    HM.qs.(m.p)==spec["p"] && HM.qs.(m.c)==[HM.qs(HM.AL.parseq(x)) for x in spec["resources"]] || error("Frozen class arrays differ.")
    data,record=HM.prepare_class(m,c;progress=(k,n)->begin
        if k==1 || k%1024==0 || k==n;println("Class ",spec["id"]," ",k,"/",n);flush(stdout);end
    end)
    meta=write_cache(req,spec["id"],"class",data)
    result=Dict{String,Any}("passed"=>true,"status"=>"PASS","identity"=>spec["id"],"specification"=>spec,
        "class_record"=>record,"cache_metadata"=>meta,"fixed_instance_whole_charge_certificate"=>true)
    commit(req,"class",result;kind="class",identity=spec["id"])
    result
end

function family_from_classes(spec,classes,plan)
    d=plan["design"];weights=HM.AL.parseq.(d["model"]["holding_weights"]);counts=Int.(d["model"]["base_class_counts"])
    b=HM.AL.parseq(spec["budget_per_project"])
    for (cl,s) in zip(classes,spec["classes"])
        cl.model.H==s["H"] && HM.qs.(cl.model.p)==s["p"] || error("Wrong class reused in family.")
        cl.model.c==HM.AL.parseq.(spec["resources"]) || error("Resource menu differs.")
        (1-cl.model.wR)/(1-cl.model.wR+cl.model.p[end])*cl.model.c[end]==21//32 || error("All-high resource normalization changed.")
    end
    f,dual=HM.from_classes(spec["id"],spec["source_group"],Int(spec["contrast"]),classes,weights,counts,b;max_population=1600)
    f.units==spec["resource_units"] && f.tick==HM.AL.parseq(spec["resource_quantum"]) || error("Lossless resource encoding changed.")
    f,dual
end

function run_probes(req,f,plan,c,release)
    wanted=filter(p->p["family"]==f.id && p["release_for"]==release,plan["probes"])
    isempty(wanted) && error("Empty readiness inventory.")
    cc=deepcopy(c);cc["probes"]=Dict("populations"=>sort!(unique(Int[p["L"] for p in wanted])),
                                    "patterns"=>["all_cap","stratified","competition"])
    expected=release=="scaling" ? TOML.parsefile(joinpath(@__DIR__,"..","test","fixtures","scaling_bounds_probes.toml"))["probes"] : Any[]
    ids=String[]
    rows=ST.probe_family(f,cc;on_probe=r->begin
        id="$(f.id)_L$(r["L"])_$(r["pattern"])_$(r["policy"])"
        id in [p["id"] for p in wanted] || error("Unexpected readiness probe.")
        if release=="scaling" && r["passed"]
            e=only(filter(x->x["id"]==id,expected))
            r["states"]==e["states"] && r["types"]==e["types"] && r["budget_units"]==e["budget_units"] || error("Readiness input changed.")
            r["actions"]==e["actions"] && r["allocator"]["resource_units"]==e["resource_units"] || error("Exact labelled readiness optimum changed.")
        end
        commit(req,"probe_L$(r["L"])_$(r["pattern"])_$(r["policy"])",r;kind="probe",identity=id)
        push!(ids,id)
    end)
    length(ids)==length(wanted) && Set(ids)==Set(p["id"] for p in wanted) || error("Incomplete readiness gate.")
    Dict{String,Any}("passed"=>all(r["passed"] for r in rows),"probe_ids"=>ids,"probe_count"=>length(ids),
        "probe_limits"=>count(r->!r["passed"],rows),"family"=>f.id,"release_for"=>release,
        "policy_ranking_is_not_a_release_criterion"=>true)
end
function prepare_family(req,plan,c)
    spec=only(filter(f->f["id"]==req["identity"],plan["families"]))
    refs=req["class_caches"]
    classes=HM.ClassData[load_cache(only(filter(x->x["identity"]==classid(cl),refs)),req,classid(cl),"class") for cl in spec["classes"]]
    f,dual=family_from_classes(spec,classes,plan)
    gate=run_probes(req,f,plan,c,"main_and_diagnostics")
    commit(req,"gate",gate;kind="gate",identity=f.id)
    gate["passed"] || return merge(gate,Dict("status"=>"REVIEW REQUIRED"))
    cache=write_cache(req,f.id,"family",f)
    result=Dict{String,Any}("passed"=>true,"status"=>"PASS","identity"=>f.id,"specification"=>spec,
        "dual"=>dual,"score_digests"=>Dict(String(p)=>HM.value_digest(f.scores[p].values) for p in (:WHITTLE_K,:LAGRANGIAN_K,:MYOPIC_K)),
        "cache_metadata"=>cache,"class_evidence"=>req["class_records"],"class_caches"=>refs,
        "main_probe_gate_passed"=>true,"main_probe_count"=>gate["probe_count"],
        "main_probe_ids"=>gate["probe_ids"],"proportional_bound_verified"=>true,
        "fixed_instance_certificate_not_uniform_PCL_theorem"=>true)
    commit(req,"family",result;kind="family",identity=f.id)
    result
end
function scaling_check(req,plan,c)
    f=load_cache(req["family_cache"],req,req["identity"],"family")
    f isa HM.Family || error("Wrong runtime family type.")
    gate=run_probes(req,f,plan,c,"scaling")
    gate["status"]=gate["passed"] ? "PASS" : "REVIEW REQUIRED"
    gate["family_cache"]=req["family_cache"]
    commit(req,"scaling_gate",gate;kind="scaling_gate",identity=f.id)
    gate
end

function verify_job(f,j,master)
    j["family"]==f.id && j["H"]==f.H && j["contrast"]==f.contrast || error("Wrong job family.")
    L=Int(j["L"]);labs=HS.labels_for(f,L)
    [count(==(k),labs) for k in eachindex(f.classes)]==j["class_counts"] || error("Class proportions differ.")
    L*f.budget_per_project==HM.AL.parseq(j["budget_exact"]) || error("Nonproportional physical budget.")
    MA.budget_units(L*f.budget_per_project,f.tick,f.units,L)==j["budget_units"] || error("Resource units changed.")
    string(MS.seed_for(master,f.source_group,L,Int(j["rep"]),"transitions-v1"))==j["transition_seed"] || error("Transition seed differs.")
    string(MS.seed_for(master,f.source_group,L,Int(j["rep"]),"initial-v1"))==j["initial_seed"] || error("Initial seed differs.")
    j["replications"]==32 || error("Changed independent sample size.")
    true
end
function scientific(x)
    if x isa AbstractDict
        return Dict{String,Any}(String(k)=>scientific(v) for (k,v) in x if String(k)!="execution_binding" && !occursin("seconds",String(k)))
    elseif x isa AbstractArray
        return map(scientific,x)
    end
    x
end
function annotate_window!(r,f,j,req)
    w=only(filter(w->w["name"]==r["window"],j["windows"]))
    r["transition_seed"]==j["transition_seed"] && r["initial_seed"]==j["initial_seed"] || error("Actual stream differs from manifest.")
    r["burn_in"]==w["burn"] && r["horizon"]==w["T"] || error("Unplanned window.")
    r["planned_replications"]=j["replications"];r["trajectory_job_id"]=j["id"]
    r["job_id"]=j["id"]*"_"*r["window"];r["stage"]=j["stage"];r["role"]=j["stage"]
    r["alpha"]=j["alpha"];r["class_counts"]=copy(j["class_counts"])
    r["budget_exact"]=j["budget_exact"];r["budget_units"]=j["budget_units"]
    r["execution_binding"]=bound_record(req)
    merge!(r,bound_record(req))
    r
end

"""Testable adapter. The production dispatcher obtains j ONLY from the frozen plan."""
function run_job(f,j,req,c;provided_uniforms=nothing,names=collect(MA.POLICIES),after_window=r->nothing)
    verify_job(f,j,c["master_seed"])
    old=Dict(r["identity"]=>r for r in get(req,"existing_windows",Any[]));verified=String[];written=String[]
    save=function(r)
        annotate_window!(r,f,j,req)
        if haskey(old,r["job_id"])
            prev=read_ref(old[r["job_id"]]);scientific(prev)==scientific(r) || error("Replayed scientific window disagrees; original kept.")
            push!(verified,r["job_id"])
            commit(req,"duplicate_"*r["window"],Dict("passed"=>true,"identity"=>r["job_id"],"previous"=>old[r["job_id"]],
                "scientific_fields_equal"=>true,"original_not_overwritten"=>true);kind="duplicate",identity=r["job_id"])
        else
            commit(req,"window_"*r["window"],r;kind="window",identity=r["job_id"]);push!(written,r["job_id"])
        end
        after_window(r)
    end
    failed=function(r)
        rr=copy(r);delete!(rr,"records");rr["specification"]=j
        commit(req,"failure",rr;kind="failure",identity=j["id"])
    end
    event=(phase,policy)->begin
        println("Confirmation phase: ",phase," / ",j["id"]," / ",policy);flush(stdout)
    end
    println("Confirmation trajectory ",j["id"]," START (original prospective seed)");flush(stdout)
    r=HS.run_windows(f,Int(j["L"]),Int(j["rep"]),j["windows"];initial=j["initial"],master_seed=c["master_seed"],
        limits=ST.limits(c),seconds=Float64(c["resource_limits"]["trajectory_seconds"]),provided_uniforms=provided_uniforms,names=names,
        retain_records=false,collect_between_policies=true,postprocess_lock=nothing,
        on_window=save,on_failure=failed,phase_event=event,
        heartbeat=(t,n)->begin if t==1 || t%10000==0 || t==n;println(j["id"]," ",t,"/",n);flush(stdout);end;end)
    isempty(r["records"]) || error("Trajectory retained heavy completed-window dictionaries.")
    delete!(r,"records");r["job_id"]=j["id"];r["specification"]=j
    r["replayed_complete_windows_verified"]=verified;r["new_complete_windows"]=written
    commit(req,"job",r;kind="job",identity=j["id"])
    Dict{String,Any}("passed"=>r["passed"],"status"=>(r["passed"] ? "PASS" : r["status"]=="limit" ? "REVIEW REQUIRED" : "FAILED"),
        "job_id"=>j["id"],"completed_windows"=>r["completed_windows"])
end

# Used only after an explicit include; both binding lookup and invocation are
# evaluated within invokelatest. Numerical dispatch stays unchanged.
function latest_module_call(parent::Module,module_name::Symbol,function_name::Symbol,args...;kwargs...)
    implementation=getfield(getfield(parent,module_name),function_name)
    implementation(args...;kwargs...)
end

function dispatch(req,root)
    plan=load_plan(root)
    plan["design_fingerprint"]==req["design_fingerprint"] && plan["native_sha256"]==req["native_manifest_sha256"] || error("Requested plan changed.")
    ST.fingerprint(ST.source_map(root))==req["source_fingerprint"] || error("Request source differs.")
    c=solver_config(root,plan);op=req["operation"]
    if op=="plan_check"
        for j in plan["jobs"]
            s="timescale_R$(j["contrast"])"
            string(MS.seed_for(c["master_seed"],s,Int(j["L"]),Int(j["rep"]),"transitions-v1"))==j["transition_seed"] || error("Cross-language seed mismatch.")
            string(MS.seed_for(c["master_seed"],s,Int(j["L"]),Int(j["rep"]),"initial-v1"))==j["initial_seed"] || error("Cross-language initial seed mismatch.")
        end
        return Dict{String,Any}("passed"=>true,"status"=>"PASS","jobs"=>896,"windows"=>1088,"seeds"=>1792)
    elseif op=="rebind_family"
        return rebind_family(req,plan,root)
    elseif op=="prepare_class"
        return prepare_class(req,plan,c)
    elseif op=="prepare_family"
        return prepare_family(req,plan,c)
    elseif op=="scaling_check"
        return scaling_check(req,plan,c)
    elseif op=="memory_calibration"
        parent=parentmodule(@__MODULE__)
        if !Base.invokelatest(isdefined,parent,:ScalingMemoryObserver)
            Base.include(parent,joinpath(@__DIR__,"scaling_memory_observer.jl"))
        end
        result=Base.invokelatest(latest_module_call,parent,:ScalingMemoryObserver,:native_controls;
            sink=(phase,r)->commit(req,"calibration_"*phase,r;kind="memory_observation",identity=req["identity"]))
        commit(req,"measurement_calibration",result;kind="measurement_calibration",identity=req["identity"])
        return result
    elseif op=="memory_check"
        calibration=read_ref(req["measurement_calibration"])
        same_binding(calibration,req)
        calibration["passed"] && calibration["measurement_revision"]=="scaling-os-memory-measurement-v1" &&
            get(calibration,"calibration_revision","")=="native-memory-control-semantics-v2" &&
            calibration["native_controls_run"] && calibration["random_trajectories_run"]==0 || error("Missing native memory calibration.")
        f=load_cache(req["family_cache"],req,req["identity"],"family")
        parent=parentmodule(@__MODULE__)
        if !Base.invokelatest(isdefined,parent,:ScalingMemoryChecks)
            Base.include(parent,joinpath(@__DIR__,"scaling_memory_checks.jl"))
        end
        result=Base.invokelatest(latest_module_call,parent,:ScalingMemoryChecks,:sustained,f,root;
            checkpoint=r->commit(req,"memory_checkpoint_$(r["completed_calls"])",r;kind="memory_progress",identity=req["identity"]),
            observation=(phase,r)->commit(req,"memory_"*phase,r;kind="memory_observation",identity=req["identity"]))
        result["measurement_calibration"]=req["measurement_calibration"]
        result["native_controls_passed"]=true
        result["calibration_revision"]=calibration["calibration_revision"]
        commit(req,"memory_gate",result;kind="memory_gate",identity=req["identity"])
        return result
    elseif op=="simulate"
        j=only(filter(j->j["id"]==req["identity"],plan["jobs"]))
        f=load_cache(req["family_cache"],req,j["family"],"family")
        f isa HM.Family || error("Wrong runtime family type.")
        return run_job(f,j,req,c)
    end
    error("Unknown operation.")
end
end
