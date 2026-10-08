# Timing diagnostics only. No model, pivot, solver or package settings are changed.
# The explicit ABI below is restricted to 64-bit Darwin and Linux.
# See docs/CONTROLLED_TIMING.md for source references and measurement boundaries.
struct TimingTimespec
    seconds::Clong
    nanoseconds::Clong
end

struct TimingClocks
    cpu_id::Cint
    continuous_id::Cint
    cpu_buffer::Base.RefValue{TimingTimespec}
    continuous_buffer::Base.RefValue{TimingTimespec}
    cpu_resolution_seconds::Float64
    continuous_resolution_seconds::Float64
end

function timing_clock_read!(id::Cint, buffer::Base.RefValue{TimingTimespec})
    rc = ccall(:clock_gettime, Cint, (Cint, Ref{TimingTimespec}), id, buffer)
    rc == 0 || error("clock_gettime failed for clock $(id); errno=$(Base.Libc.errno()).")
    t = buffer[]
    t.seconds >= 0 && 0 <= t.nanoseconds < 1_000_000_000 || error("Invalid clock value.")
    return Int64(t.seconds) * 1_000_000_000 + Int64(t.nanoseconds)
end

function timing_clock_resolution(id::Cint)
    b = Ref(TimingTimespec(0, 0))
    rc = ccall(:clock_getres, Cint, (Cint, Ref{TimingTimespec}), id, b)
    rc == 0 || error("clock_getres failed for clock $(id); errno=$(Base.Libc.errno()).")
    return Float64(b[].seconds) + Float64(b[].nanoseconds) / 1e9
end

function TimingClocks()
    Sys.WORD_SIZE == 64 && sizeof(Clong) == 8 || error("This timing adapter requires a 64-bit Unix ABI.")
    # Darwin: CLOCK_PROCESS_CPUTIME_ID=12, CLOCK_MONOTONIC_RAW=4 (continuous).
    # Linux:  CLOCK_PROCESS_CPUTIME_ID=2, CLOCK_BOOTTIME=7 (includes suspension).
    cpu_id, continuous_id = if Sys.isapple()
        (Cint(12), Cint(4))
    elseif Sys.islinux()
        (Cint(2), Cint(7))
    else
        error("Process-CPU/continuous-clock adapter supports macOS and Linux only; no silent fallback.")
    end
    c = TimingClocks(cpu_id, continuous_id, Ref(TimingTimespec(0,0)),
        Ref(TimingTimespec(0,0)), timing_clock_resolution(cpu_id),
        timing_clock_resolution(continuous_id))
    a = timing_clock_read!(c.cpu_id,c.cpu_buffer)
    b = timing_clock_read!(c.cpu_id,c.cpu_buffer)
    b >= a || error("Process CPU clock is not nondecreasing.")
    timing_clock_read!(c.continuous_id,c.continuous_buffer)
    return c
end

timing_cpu!(c::TimingClocks) = timing_clock_read!(c.cpu_id,c.cpu_buffer)
timing_continuous!(c::TimingClocks) = timing_clock_read!(c.continuous_id,c.continuous_buffer)

# CPU and continuous clocks bracket the @timed wrapper, not just its expression.
# Their small extra envelope is recorded, NOT subtracted. The kernel's @timed
# allocation count excludes these diagnostic clock calls and subsequent reports.
function timing_instrument(c::TimingClocks, f::F, args::Vararg{Any,N}) where {F,N}
    cpu0 = timing_cpu!(c)
    continuous0 = timing_continuous!(c)
    outer0 = time_ns()
    measured = @timed f(args...)
    outer1 = time_ns()
    continuous1 = timing_continuous!(c)
    cpu1 = timing_cpu!(c)
    cpu1 >= cpu0 && continuous1 >= continuous0 || error("Timing clock went backwards.")
    return (measured=measured, cpu_seconds=(cpu1-cpu0)/1e9,
        continuous_seconds=(continuous1-continuous0)/1e9,
        outer_wall_seconds=Float64(outer1-outer0)/1e9)
end

# Fixed, prospective diagnostic rules. Flags do not remove or replace samples.
function timing_diagnostics(kernel_seconds, outer_seconds, cpu_seconds, continuous_seconds, c)
    all(x->isfinite(x) && x>=0, (kernel_seconds,outer_seconds,cpu_seconds,continuous_seconds)) ||
        error("Invalid measured duration.")
    ratio = outer_seconds > 0 ? cpu_seconds / outer_seconds : 0.0
    long_enough = outer_seconds >= c["diagnostic_minimum_batch_seconds"]
    low_cpu = long_enough && ratio < c["cpu_wall_ratio_low"]
    high_cpu = long_enough && ratio > c["cpu_wall_ratio_high"]
    clock_gap = abs(continuous_seconds-outer_seconds) > max(
        c["clock_gap_absolute_seconds"], c["clock_gap_relative"]*outer_seconds)
    # Duration credit cannot be satisfied merely by a long descheduled/sleeping
    # interval, or by extra simultaneous CPU threads: credit <= each clock.
    credit = min(kernel_seconds,cpu_seconds)
    return (cpu_to_outer_wall=ratio, low_cpu_fraction=low_cpu,
        high_cpu_fraction=high_cpu, continuous_clock_gap=clock_gap,
        diagnostic_flag=low_cpu || high_cpu || clock_gap, duration_credit_seconds=credit)
end

function timing_clock_metadata(c::TimingClocks)
    Dict{String,Any}("process_cpu_clock_id"=>Int(c.cpu_id),
        "continuous_clock_id"=>Int(c.continuous_id),
        "process_cpu_resolution_seconds"=>c.cpu_resolution_seconds,
        "continuous_resolution_seconds"=>c.continuous_resolution_seconds,
        "primary_elapsed"=>"Julia @timed expression elapsed seconds",
        "cpu_scope"=>"user+system CPU of the entire Julia process, including GC/runtime threads; not other processes",
        "cpu_boundary"=>"encloses @timed metadata collection and two continuous-clock reads; not an exact kernel-only CPU time",
        "continuous_scope"=>"suspension-inclusive monotonic diagnostic envelope",
        "flags"=>"diagnostics, not diagnoses of sleep or proof of uncontended execution")
end

# Optional read-only observations: failures are saved as text, never disguised
# as a power or clock setting. Called outside all timed batches.
function timing_command(args)
    try
        strip(read(pipeline(Cmd(args),stderr=devnull),String))
    catch err
        "unavailable: " * sprint(showerror,err)
    end
end
function timing_system_snapshot()
    d=Dict{String,Any}("utc"=>string(now(UTC)),"pid"=>getpid())
    if Sys.isapple()
        d["power_source"]=timing_command(["/usr/bin/pmset","-g","batt"])
        d["power_settings"]=timing_command(["/usr/bin/pmset","-g","custom"])
        d["sleep_assertions"]=timing_command(["/usr/bin/pmset","-g","assertions"])
        d["thermal_status"]=timing_command(["/usr/bin/pmset","-g","therm"])
        d["os_version"]=timing_command(["/usr/bin/sw_vers"])
    elseif Sys.islinux()
        d["os_version"]=timing_command(["uname","-a"])
        d["load_average"]=isfile("/proc/loadavg") ? strip(read("/proc/loadavg",String)) : "unavailable"
        d["sleep_assertions"]="not collected on Linux"
    end
    return d
end
