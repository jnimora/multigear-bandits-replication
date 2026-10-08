# Input-only archives are NOT timing runs. Included by the replay loader.
const INSTANCE_ARCHIVE_PROTOCOL = "multigear-instance-archive-v1"

function instance_archive_path(root::AbstractString, rel::AbstractString)
    parts=split(String(rel),'/')
    !isempty(parts) && all(p->!isempty(p) && p!="." && p!="..",parts) &&
        !occursin('\\',rel) && !occursin(':',rel) && !isabspath(rel) ||
        error("Unsafe input-archive path: $rel")
    path=String(root)
    isdir(path) && !islink(path) || error("Archive root missing or linked: $path")
    for part in parts
        path=joinpath(path,part)
        islink(path) && error("Linked archive path is not supported: $path")
    end
    isfile(path) || error("Missing archive file: $rel")
    return path
end

function instance_archive_hash_check(root,rel,expected)
    expected isa AbstractString && occursin(r"^[0-9a-f]{64}$",expected) ||
        error("Invalid SHA256 record for $rel")
    path=instance_archive_path(root,rel)
    file_sha(path)==expected || error("Input-archive checksum mismatch: $rel")
    return path
end

function instance_archive_case_id(n,A,draw,eta,criterion)
    return "N$(n)_A$(A)_draw$(draw)_eta$(Float64(eta))_$(criterion)"
end

function instance_archive_manifest_check(root,meta)
    get(meta,"protocol","")==INSTANCE_ARCHIVE_PROTOCOL || error("Unsupported input-only archive.")
    get(meta,"archive_complete",false)===true || error("Input archive was not finalized; retain its partial records.")
    get(meta,"integrity_checks_passed",false)===true || error("Input-archive integrity checks did not pass.")
    get(meta,"timing_measurements_present",true)===false || error("Input-only archive must not claim timing evidence.")
    config=meta["configuration"]
    get(config,"protocol","")=="instance-execution-v1" || error("Invalid execution configuration.")
    get(meta,"terminal_update",nothing)==get(config,"terminal_update",nothing) || error("Endpoint conflict.")
    rows=TOML.parsefile(instance_archive_hash_check(root,"cases.toml",meta["cases_sha256"]))["case"]
    expected=Set(instance_archive_case_id(n,A,d,e,k) for n in config["states"] for A in config["gears"]
        for d in config["draws"] for e in config["eta"] for k in config["criteria"])
    ids=[r["case_id"] for r in rows]
    length(ids)==length(unique(ids)) && Set(ids)==expected && length(ids)==meta["expected_cases"] ||
        error("Input-archive case inventory is incomplete or duplicated.")
    files=meta["archived_input_sha256"]
    for (rel,hash) in files
        startswith(rel,"inputs/") || error("Unexpected input location: $rel")
        instance_archive_hash_check(root,rel,hash)
    end
    for row in rows
        id=instance_archive_case_id(row["N"],row["A"],row["draw"],row["eta"],row["criterion"])
        row["case_id"]==id || error("Case labels disagree: $(row["case_id"])")
        prefix=chop(id;tail=length("_"*row["criterion"]))
        rel="inputs/"*prefix*".toml"
        get(files,rel,nothing)==row["model_sha256"] || error("Uninventoried model: $id")
        prim="inputs/primitives_N$(row["N"])_A$(row["A"])_draw$(row["draw"]).toml"
        get(files,prim,nothing)==row["primitive_sha256"] || error("Uninventoried primitives: $id")
        haskey(row,"validation") && haskey(row["validation"],"C") || error("Missing C reference record: $id")
        get(row,"dai_verified",true)===false || error("Input generation must not assert DAI verification.")
    end
    return rows
end
