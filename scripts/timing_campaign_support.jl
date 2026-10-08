# Campaign diagnostics; none of these functions executes inside an algorithm timer.
# The default controlled-run policy leaves normal garbage collection enabled.
# Full collections are never requested by the measured runner's batch calls.

function timing_optional_collection(enabled::Bool, collector::F=GC.gc) where {F}
    enabled || return (calls=0, seconds=0.0)
    started=time_ns()
    collector()
    return (calls=1, seconds=Float64(time_ns()-started)/1e9)
end

mutable struct TimingCampaignProgress
    path::String
    output::IO
    interval_ns::UInt64
    started_ns::UInt64
    last_ns::UInt64
end

function TimingCampaignProgress(path::AbstractString; output::IO=stdout,
                                interval_seconds::Real=10.0, start_ns::UInt64=time_ns())
    isfinite(interval_seconds) && 0<interval_seconds<=3600 ||
        error("Progress interval must be in (0, 3600] seconds.")
    p=String(path)
    isempty(p) || mkpath(dirname(abspath(p)))
    return TimingCampaignProgress(p,output,UInt64(round(Int,interval_seconds*1e9)),
                                  start_ns,start_ns)
end

function timing_progress_due(p::TimingCampaignProgress; now_ns::UInt64=time_ns())
    now_ns>=p.last_ns || error("Progress clock moved backwards.")
    return now_ns-p.last_ns>=p.interval_ns
end

# The caller invokes this only outside timed batches (between complete rounds).
# Flushing makes each message visible in Terminal and the append-only progress log.
# This is not a background task, watchdog, or hard timeout on a blocked operation.
function timing_progress!(p::TimingCampaignProgress,message::AbstractString;
                          force::Bool=false,now_ns::UInt64=time_ns())
    now_ns>=p.started_ns && now_ns>=p.last_ns || error("Progress clock moved backwards.")
    (force || timing_progress_due(p;now_ns=now_ns)) || return false
    elapsed=round(Float64(now_ns-p.started_ns)/1e9;digits=3)
    line="[campaign $(elapsed) s] "*String(message)
    println(p.output,line);flush(p.output)
    if !isempty(p.path)
        open(p.path,"a") do io
            println(io,line);flush(io)
        end
    end
    p.last_ns=now_ns
    return true
end

function timing_record_batch_cost!(row::AbstractDict,started_ns::UInt64,gc;
                                   now_ns::UInt64=time_ns())
    now_ns>=started_ns || error("Batch accounting clock moved backwards.")
    total=Float64(now_ns-started_ns)/1e9
    kernel=Float64(row["window_seconds"])
    isfinite(kernel) && 0<=kernel<=total || error("Invalid batch/kernel elapsed accounting.")
    row["batch_total_wall_seconds"]=total
    row["batch_outside_timer_seconds"]=total-kernel
    row["forced_gc_calls"]=gc.calls
    row["forced_gc_seconds"]=gc.seconds
    row["gc_regime"]=gc.calls==0 ? "automatic" : "explicit_collection_diagnostic"
    return row
end

function timing_record_sample_cost!(row::AbstractDict,executed)
    complete=!isempty(executed) && all(haskey(r,"batch_total_wall_seconds") &&
        haskey(r,"batch_outside_timer_seconds") && haskey(r,"forced_gc_calls") &&
        haskey(r,"forced_gc_seconds") for r in executed)
    row["batch_cost_accounting_complete"]=complete
    if complete
        row["batch_total_wall_seconds"]=sum(r["batch_total_wall_seconds"] for r in executed)
        row["batch_outside_timer_seconds"]=sum(r["batch_outside_timer_seconds"] for r in executed)
        row["forced_gc_calls"]=sum(r["forced_gc_calls"] for r in executed)
        row["forced_gc_seconds"]=sum(r["forced_gc_seconds"] for r in executed)
        row["gc_regime"]=row["forced_gc_calls"]==0 ? "automatic" : "explicit_collection_diagnostic"
    end
    return row
end

# Scalar display helper; avoids shadowing Base.round inside round callbacks.
timing_progress_seconds(x::Real)=round(Float64(x);digits=6)
