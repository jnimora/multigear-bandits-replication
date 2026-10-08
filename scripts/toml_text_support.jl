# Report serialization only. Never called from an algorithm's timed kernel.
# Preserve malformed string bytes instead of guessing their original encoding.
module PilotTOMLText

using TOML

const EVENT_KEY = "__multigear_text_encoding__"
const DISPLAY_PREFIX = "[non-UTF-8 metadata; Julia-escaped bytes] "

# Use typed path components to distinguish an array position from a key.
_key_step(k::AbstractString) = Dict{String,Any}("kind"=>"key", "value"=>String(k))
_index_step(i::Integer) = Dict{String,Any}("kind"=>"index", "value"=>Int(i))

function _find_invalid!(events, value, path)
    if value isa AbstractString
        if !isvalid(value)
            push!(events, Dict{String,Any}(
                "path"=>copy(path),
                "original_hex"=>bytes2hex(codeunits(value)),
                "byte_count"=>ncodeunits(value),
                "stored_representation"=>"prefix followed by Julia escape_string; original_hex is authoritative"))
        end
    elseif value isa AbstractDict
        for (key, child) in value
            key isa AbstractString || throw(ArgumentError("TOML report keys must be strings."))
            # Renaming malformed keys could collide with real keys. Fail closed.
            isvalid(key) || throw(ArgumentError("Invalid UTF-8 in TOML key; original key bytes (hex): " * bytes2hex(codeunits(key))))
            push!(path, _key_step(key))
            _find_invalid!(events, child, path)
            pop!(path)
        end
    elseif value isa AbstractVector
        # Numeric arrays cannot contain malformed strings. Do not copy/scan them.
        eltype(value) <: Number && return nothing
        for i in eachindex(value)
            push!(path, _index_step(i))
            _find_invalid!(events, value[i], path)
            pop!(path)
        end
    end
    return nothing
end

function _safe_copy(value)
    if value isa AbstractString
        isvalid(value) && return value
        escaped = DISPLAY_PREFIX * escape_string(value)
        isvalid(escaped) || error("The escaped metadata string is still invalid UTF-8.")
        return escaped
    elseif value isa AbstractDict
        return Dict{String,Any}(String(k)=>_safe_copy(v) for (k,v) in value)
    elseif value isa AbstractVector
        eltype(value) <: Number && return value
        return map(_safe_copy, value)
    end
    # Numbers, Boolean flags, dates and other supported TOML scalars are untouched.
    return value
end

"""
    prepare_document(object::AbstractDict)

Return `(document, events)` for writing. Valid documents are returned unchanged.
Only malformed STRING VALUES are escaped; every such value is recorded with its
original byte sequence and exact path. Keys are never renamed. The supplied
object is not mutated. A reserved-key collision is rejected, not overwritten.
"""
function prepare_document(object::AbstractDict)
    events = Dict{String,Any}[]
    _find_invalid!(events, object, Dict{String,Any}[])
    isempty(events) && return (document=object, events=events)
    haskey(object, EVENT_KEY) && throw(ArgumentError("Reserved TOML encoding-audit key already exists: " * EVENT_KEY))
    safe = _safe_copy(object)
    safe[EVENT_KEY] = Dict{String,Any}(
        "version"=>1,
        "policy"=>"lossless escape of malformed UTF-8 string values; no encoding inferred; valid text and numerical values unchanged",
        "display_prefix"=>DISPLAY_PREFIX,
        "event_count"=>length(events),
        "events"=>events)
    return (document=safe, events=events)
end

function describe_path(path)
    io = IOBuffer()
    print(io, "root")
    for step in path
        if step["kind"] == "key"
            print(io, '[', repr(step["value"]), ']')
        else
            print(io, '[', step["value"], ']')
        end
    end
    return String(take!(io))
end

"""
    write_report(path, object; warn_invalid=true)

Complete TOML serialization before opening/replacing the destination. Stage the
bytes in the destination directory, then move them into place. In particular,
encoding or unsupported-value errors do not truncate an existing report. This
is not a durability guarantee against disk failure or concurrent writers.
"""
function write_report(path::AbstractString, object::AbstractDict; warn_invalid::Bool=true)
    prepared = prepare_document(object)
    buffer = IOBuffer()
    TOML.print(buffer, prepared.document; sorted=true)
    bytes = take!(buffer)
    # UTF-8 serialization is complete before the destination is touched.
    target = abspath(path)
    islink(target) && throw(ArgumentError("Refusing to overwrite a symbolic-link report: " * target))
    ispath(target) && !isfile(target) && throw(ArgumentError("Report destination is not a regular file: " * target))
    mkpath(dirname(target))
    temporary, io = mktemp(dirname(target); cleanup=false)
    try
        write(io, bytes)
        flush(io)
        close(io)
        mv(temporary, target; force=true)
    finally
        isopen(io) && close(io)
        isfile(temporary) && rm(temporary)
    end
    if warn_invalid && !isempty(prepared.events)
        paths = [describe_path(event["path"]) for event in prepared.events]
        @warn "Non-UTF-8 report text preserved as escaped bytes; see __multigear_text_encoding__ in this report." report=target fields=paths
    end
    return nothing
end

end # module PilotTOMLText
