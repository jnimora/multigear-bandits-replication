# Validation/reporting helper only: call AFTER the measured expression.
# No algorithm-work counter may be dropped, rounded, or silently reset.

function timing_checked_count_dict(counts)
    counts isa AbstractDict || error("Work counts must be a dictionary.")
    out=Dict{String,Int}()
    for (key,value) in counts
        key isa AbstractString || error("Work-counter names must be strings.")
        value isa Integer && !(value isa Bool) && value>=0 ||
            error("Work counter $(repr(key)) must be a nonnegative integer.")
        out[String(key)]=Int(value)
    end
    return out
end

"""
Compare every non-audit work counter exactly with the independent reference.
The sole excluded reference counter is `audit_solves`: the reference may have
been checked against fresh solves at each policy. Every timed output must have
ZERO such validation audits. The original snapshots/count dictionaries are not
modified. Exact-comparison safeguards, rebuilds, and all other work still count.
"""
function timing_work_count_check(outputs,reference)
    expected=timing_checked_count_dict(reference.counts)
    reference_audits=get(expected,"audit_solves",0)
    differences=Dict{String,Any}[]
    max_timed_audits=0
    isempty(outputs) && push!(differences,Dict("kind"=>"empty_output_batch"))
    for (j,snapshot) in enumerate(outputs)
        actual=timing_checked_count_dict(snapshot.counts)
        audits=get(actual,"audit_solves",0)
        max_timed_audits=max(max_timed_audits,audits)
        # Require the original schema too: omission is not a way to conceal audits.
        if haskey(actual,"audit_solves")!=haskey(expected,"audit_solves")
            push!(differences,Dict("output"=>j,"counter"=>"audit_solves",
                "kind"=>"audit_counter_schema_mismatch",
                "actual_present"=>haskey(actual,"audit_solves"),
                "reference_present"=>haskey(expected,"audit_solves")))
        end
        if audits!=0
            push!(differences,Dict("output"=>j,"counter"=>"audit_solves",
                "kind"=>"validation_audit_in_timed_execution","actual"=>audits,"expected"=>0))
        end
        for key in sort!(collect(union(keys(actual),keys(expected))))
            key=="audit_solves" && continue
            if !haskey(actual,key) || !haskey(expected,key) || actual[key]!=expected[key]
                push!(differences,Dict("output"=>j,"counter"=>key,"kind"=>"algorithm_work_mismatch",
                    "actual"=>get(actual,key,"absent"),"expected"=>get(expected,key,"absent")))
            end
        end
    end
    return (passed=isempty(differences),reference_audit_solves=reference_audits,
        maximum_timed_audit_solves=max_timed_audits,differences=differences)
end

# Explicit stage/method/phase diagnostics. Warmup may still compile; numerical
# agreement, completion, and the work-count contract may NEVER be bypassed.
function timing_require_matching_batch(row,variant,phase,stage;require_steady=false)
    numerical=get(row,"matches_validated_execution",false)
    complete=get(row,"complete",false)
    counts=get(row,"equal_work_counts_in_batch",false)
    eligible=get(row,"execution_eligible",false)
    if !(numerical && complete && counts && (!require_steady || eligible))
        details=get(row,"work_count_mismatches",Dict{String,Any}[])
        error("Timing batch check failed at $stage for $variant/$phase: " *
            "numerical_match=$numerical, complete=$complete, algorithm_work_match=$counts, " *
            "execution_eligible=$eligible; reference audit_solves=" *
            string(get(row,"reference_validation_audit_solves","unrecorded")) *
            ", maximum timed audit_solves=" *
            string(get(row,"maximum_timed_validation_audit_solves","unrecorded")) *
            "; counter differences=" * repr(details))
    end
    return nothing
end
