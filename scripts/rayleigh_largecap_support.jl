"""Large-cap frozen-input/measurement adapters. Existing kernels remain unchanged.
The mathematical certificate is in rayleigh_largecap_certificate.jl. Neither
certificates nor reference paths are inputs to any measured numerical selector.
"""
module RayleighLargeCap
using LinearAlgebra, TOML, Dates, SHA, Random, Statistics, Printf
using ..MultiGearBandits
using ..RayleighAoII
using ..RayleighBlock
using ..RayleighTiming
const RT=RayleighTiming
const RB=RayleighBlock
const Q=Rational{BigInt}
const E=RB.Enclosed
const REVISION="rayleigh-largecap-fixed-chain-v1"
const METHODS=(:C,:R,:M,:FP_legacy,:FP,:BLOCK)
const PHASES=(:initialization,:index_loop,:end_to_end)

# A small strict JSON writer avoids a new project dependency or requiring tomllib
# in the Python 3.8+ supervisor. Numerical nonfinite values are refused, not zeroed.
function json_string(io,s)
    print(io,'"')
    for c in String(s)
        if c=='"';print(io,"\\\"")
        elseif c=='\\';print(io,"\\\\")
        elseif c=='\n';print(io,"\\n")
        elseif c=='\r';print(io,"\\r")
        elseif c=='\t';print(io,"\\t")
        elseif Int(c)<32;@printf(io,"\\u%04x",Int(c))
        else;print(io,c)
        end
    end
    print(io,'"')
end
function json_value(io,x)
    if x===nothing || x===missing;print(io,"null")
    elseif x isa Bool;print(io,x ? "true" : "false")
    elseif x isa AbstractString || x isa Symbol;json_string(io,string(x))
    elseif x isa Integer;print(io,x)
    elseif x isa AbstractFloat
        isfinite(x) || error("Nonfinite JSON number.");print(io,repr(x))
    elseif x isa AbstractDict
        print(io,'{');keys0=sort!(collect(keys(x));by=string)
        for (j,k) in enumerate(keys0)
            j>1 && print(io,',');json_string(io,string(k));print(io,':');json_value(io,x[k])
        end
        print(io,'}')
    elseif x isa Tuple || x isa AbstractVector
        print(io,'[')
        for (j,v) in enumerate(x);j>1 && print(io,',');json_value(io,v);end
        print(io,']')
    else
        error("Unsupported JSON record type $(typeof(x)).")
    end
end
function write_json(path,d)
    mkpath(dirname(path));tmp=path*".tmp"
    open(tmp,"w") do io;json_value(io,d);println(io);end
    mv(tmp,path;force=true)
end
function write_record(stem,d)
    RT.pilot_write_toml(stem*".toml",d);write_json(stem*".json",d)
end

function configuration(path)
    c=TOML.parsefile(path)
    c["revision"]==REVISION && c["criterion"]=="average" &&
    c["resource_scope"]=="constructed_sync_gain_not_battery_calibration" || error("Unknown large-cap protocol.")
    c["caps"]==[16,32,64,128,256,512,1024,2048] && c["gears"]==[2,3] &&
    c["regimes"]==["anchor","persistent"] || error("Do not change the prespecified grid without a new review.")
    c["preflight_ids"]==["anchor_H16_A3","anchor_H32_A2","anchor_H512_A3",
       "persistent_H512_A2","anchor_H2048_A2","persistent_H2048_A3"] || error("Preflight inventory changed.")
    Tuple(Symbol.(c["variants"]))==METHODS && !c["terminal_update"] || error("Methods/endpoint changed.")
    c["source_states"]==2 && c["kappa"]=="2" && c["epsilon"]=="1/8" && c["chi"]=="1" || error("Frozen primitives changed.")
    c["output_atol"]==1e-9 && c["output_rtol"]==1e-9 || error("Common numerical agreement changed.")
    RT.block_options(c) # exact equality to the previously accepted BlockOptions
    expected=Dict("comparison_atol"=>1e-11,"comparison_rtol"=>1e-9,"pivot_atol"=>0.0,
      "pivot_rtol"=>1e-12,"exact_fallback"=>true,"exact_max_states"=>8,
      "residual_fallback"=>true,"residual_max_states"=>256)
    c["screening"]==expected || error("Accepted generic safeguards changed.")
    t=c["timing"];t["protocol"]=="isolated-fixed-count-complete-runs-v1" &&
      Tuple(Symbol.(t["phases"]))==PHASES && t["repetitions"]==3 && t["warmups_per_phase"]==2 &&
      t["batch_size"]==1 && t["reference_block_columns"]==64 || error("Complete-run sampling protocol changed.")
    for (key,want) in (("diagnostic_minimum_batch_seconds",0.0001),("cpu_wall_ratio_low",0.75),
       ("cpu_wall_ratio_high",1.25),("clock_gap_absolute_seconds",0.05),("clock_gap_relative",0.05))
        t[key]==want || error("Diagnostic rule changed: $key")
    end
    for (key,lo,hi) in (("method_process_seconds",30,1800),("rss_stop_gib",1,16),
       ("model_memory_admission_gib",1,16),("rss_poll_seconds",0.2,5),
       ("preflight_campaign_seconds",60,14400),("full_campaign_seconds",60,86400))
        x=t[key];x isa Real && !(x isa Bool) && isfinite(x) && lo<=x<=hi || error("Invalid resource limit: $key")
    end
    p=c["certificate"]
    p["root_atol"]==1e-10 && p["root_rtol"]==1e-11 &&
      p["maximum_exact_policy_fallbacks"]==64 && p["exact_max_H"]==2048 && p["exact_max_A"]==3 &&
      p["exact_input_bits"]==2048 && p["seconds_per_case"]==600.0 && p["external_seconds"]==1200 ||
        error("Certificate budget or error rules changed.")
    c
end
function make_spec(H::Int,A::Int,regime::String)
    1<=H<=2048 && A in (2,3) && regime in ("anchor","persistent") || error("Invalid model dimensions/regime.")
    d=regime=="anchor" ? 4 : 1024
    s=A==2 ? ["0","1/$d","3/$d"] : ["0","1/$d","2/$d","3/$d"]
    Dict{String,Any}("id"=>"$(regime)_H$(H)_A$(A)","H"=>H,"A"=>A,"regime"=>regime,
      "source_states"=>2,"wR"=>"$(d-1)/$d","success"=>s,"kappa"=>"2","epsilon"=>"1/8","chi"=>"1")
end
function specification_model(spec)
    spec==make_spec(Int(spec["H"]),Int(spec["A"]),String(spec["regime"])) || error("Frozen model specification mismatch.")
    m=from_success(Int(spec["H"]);nsrc=2,wR=spec["wR"],success=spec["success"],kappa=spec["kappa"],
       epsilon=spec["epsilon"],chi=spec["chi"])
    # Check source rows without ever materializing the dense matrices. Checking
    # 1-p separately covers the rounded subtraction in the generic dense builder.
    xs=vcat(m.p,m.c,[m.wR,one(Q)-m.wR,one(Q)])
    all(Q(Float64(x))==x for x in xs) && all(Q(1.0-Float64(x))==1-x for x in m.p) ||
       error("The generic Float64 arrays would not equal the certified rational model.")
    m
end
function model_specs(c,stage)
    stage in ("preflight","full") || error("Stage must be preflight or full.")
    allspec=[make_spec(Int(H),Int(A),String(r)) for r in c["regimes"] for H in c["caps"] for A in c["gears"]]
    stage=="full" && return allspec
    index=Dict(s["id"]=>s for s in allspec)
    [index[id] for id in c["preflight_ids"]]
end

include("rayleigh_largecap_certificate.jl")

# BLOCK's common input is O(A), and its output is O(AH). It must not allocate a
# dense transition array simply to accommodate a generic baseline's interface.
struct SparseContext
    model::AoIIModel{Q}
    options::BlockOptions
end
mutable struct SparseBlockRun
    workspace::NumericBlockWorkspace
    mpi::Matrix{Union{Missing,Float64}}
    state::Vector{Int}
    gear::Vector{Int}
    f::Vector{Float64}
    g::Vector{Float64}
    ratio::Vector{Float64}
    modes::Vector{Symbol}
    bounds::Matrix{Float64}
    k::Int
    status::Symbol
    message::String
    terminal_update_skipped::Bool
end
function lc_prepare(ctx::SparseContext,::Val{:BLOCK})
    w=NumericBlockWorkspace(ctx.model;options=ctx.options);H=ctx.model.H;A=length(ctx.model.p)-1;K=H*A
    mpi=Matrix{Union{Missing,Float64}}(undef,H,A);fill!(mpi,missing)
    SparseBlockRun(w,mpi,Vector{Int}(undef,K),Vector{Int}(undef,K),Vector{Float64}(undef,K),
      Vector{Float64}(undef,K),Vector{Float64}(undef,K),Vector{Symbol}(undef,K),
      Matrix{Float64}(undef,K,6),0,:ready,"",false)
end
lc_prepare(ctx::RT.AppContext,v::Val)=RT.app_prepare(ctx,v)
function lc_run!(r::SparseBlockRun)
    r.k==0 && r.status==:ready || error("Fresh workspace required.")
    w=r.workspace;K=length(r.state);started=time_ns()
    for k in 1:K
        if (time_ns()-started)/1e9>w.options.seconds_limit
            r.status=:time_limit;r.message="Accepted block cooperative time limit";return r.status
        end
        try
            selected=choose_frontier!(w);x=selected.candidate
            if k<K;numerical_shift!(w,x.h,x.a);else;r.terminal_update_skipped=true;end
            r.state[k]=x.h+1;r.gear[k]=x.a;r.f[k]=x.f.value;r.g[k]=x.g.value
            r.ratio[k]=x.index.value;r.mpi[x.h,x.a]=x.index.value;r.modes[k]=selected.mode
            r.bounds[k,:].=(x.index.lo,x.index.hi,x.f.lo,x.f.hi,x.g.lo,x.g.hi)
            r.k=k;r.status=k==K ? :complete : :running
        catch err
            err isa RB.NumericalIssue || rethrow()
            r.status=err.status;r.message=err.message;return r.status
        end
    end
    r.status
end
lc_run!(r::MultiGearBandits.PerformanceRun)=run_performance!(r)
function lc_snapshot(r::SparseBlockRun)
    k=r.k;H=size(r.mpi,1);A=size(r.mpi,2);counts=copy(r.workspace.counts)
    counts["assignments"]=k;counts["terminal_numerical_updates_skipped"]=Int(r.terminal_update_skipped)
    (algorithm=:BLOCK,status=r.status,complete=r.status==:complete,states=collect(2:H+1),mpi=copy(r.mpi),
     order=[(r.state[j],r.gear[j]) for j in 1:k],f=r.f[1:k],g=r.g[1:k],ratio=r.ratio[1:k],
     dimension=fill(A+1,k),evidence=r.modes[1:k],counts=counts,
     terminal_update_skipped=r.terminal_update_skipped,residual_comparisons=Dict{String,Any}[])
end
lc_snapshot(r::MultiGearBandits.PerformanceRun)=performance_snapshot(r)
function lc_complete(ctx,v)
    r=lc_prepare(ctx,v);lc_run!(r);lc_snapshot(r)
end
function lc_reference(ctx,v)
    r=lc_prepare(ctx,v);lc_run!(r);s=lc_snapshot(r)
    d=RT.pilot_snapshot_record(s)
    d["retained_workspace_bytes_including_shared_inputs"]=Base.summarysize(r)
    if r isa SparseBlockRun
        d["returned_enclosures"]=[collect(r.bounds[k,:]) for k in 1:r.k]
    end
    s,d
end
function snapshot_check(s,cert,c)
    spec=cert["specification"];H=Int(spec["H"]);A=Int(spec["A"]);rows=cert["path"]
    s.complete && s.status==:complete && s.terminal_update_skipped && s.states==collect(2:H+1) &&
      length(s.order)==H*A && size(s.mpi)==(H,A) || return false
    for (k,r) in enumerate(rows)
        s.order[k]==(Int(r["age"])+1,Int(r["upper_gear"])) || return false
        x=s.ratio[k];lo=Float64(r["root_lo"]);hi=Float64(r["root_hi"])
        isfinite(x) && max(abs(x-lo),abs(x-hi))<=c["output_atol"]+c["output_rtol"]*abs(x) || return false
        isequal(s.mpi[Int(r["age"]),Int(r["upper_gear"])],x) || return false
    end
    true
end
function own_reference_check(s,ref,c)
    snapshot_check(s,ref.certificate,c) || return false
    r=ref.snapshot
    s.order==r.order && s.counts==r.counts || return false
    all(isapprox(x,y;atol=c["output_atol"],rtol=c["output_rtol"]) for field in (:f,:g,:ratio)
       for (x,y) in zip(getproperty(s,field),getproperty(r,field)))
end
function primary_call(ctx,v,phase,clock)
    if phase==:initialization
        t=RT.timing_instrument(clock,lc_prepare,ctx,v)
        run=t.measured.value;lc_run!(run);s=lc_snapshot(run)
    elseif phase==:index_loop
        run=lc_prepare(ctx,v);t=RT.timing_instrument(clock,lc_run!,run);s=lc_snapshot(run)
    elseif phase==:end_to_end
        t=RT.timing_instrument(clock,lc_complete,ctx,v);s=t.measured.value
    else;error("Invalid phase.")
    end
    t,s
end
function sample_record(ctx,v,phase,clock,ref,c,repetition,warmup)
    started=time_ns();t,s=primary_call(ctx,v,phase,clock);a=t.measured
    passed=own_reference_check(s,ref,c)
    d=RT.timing_diagnostics(a.time,t.outer_wall_seconds,t.cpu_seconds,t.continuous_seconds,c["timing"])
    steady=a.compile_time==0 && a.recompile_time==0
    row=Dict{String,Any}("algorithm"=>string(typeof(v).parameters[1]),"phase"=>string(phase),
      "repetition"=>repetition,"warmup"=>warmup,"evaluations"=>1,"elapsed_seconds"=>a.time,
      "allocated_bytes"=>Int(a.bytes),"gc_seconds"=>a.gctime,"compile_seconds"=>a.compile_time,
      "recompile_seconds"=>a.recompile_time,"outer_wall_seconds"=>t.outer_wall_seconds,
      "cpu_seconds"=>t.cpu_seconds,"continuous_seconds"=>t.continuous_seconds,
      "CPU_to_outer_wall"=>d.cpu_to_outer_wall,"low_cpu_fraction"=>d.low_cpu_fraction,
      "high_cpu_fraction"=>d.high_cpu_fraction,"continuous_clock_gap"=>d.continuous_clock_gap,
      "diagnostic_flag"=>d.diagnostic_flag,"matches_certified_and_method_reference"=>passed,
      "complete"=>s.complete,"status"=>string(s.status),"counts"=>s.counts,
      "primary_eligible"=>!warmup && passed && steady && isfinite(a.time) && a.time>0,
      "whole_call_envelope_seconds"=>(time_ns()-started)/1e9,
      "clock_scope"=>"single complete evaluation; CPU envelope includes timer metadata; no duration-target qualification",
      "sampling_protocol"=>c["timing"]["protocol"])
    row,s
end
function estimated_dense_bytes(H,A)
    # Conservative ADMISSION ESTIMATE only: includes multiple potential work
    # arrays and 1 GiB runtime slack. RSS monitoring is a separate safeguard.
    Int128(8)*(20*(A+1)+32)*Int128(H+1)^2 + (Int128(1)<<30)
end

end # module
