"""Prospective windows from one shared trajectory, all six policies.
Old simulator/seed functions remain unchanged. This layer instruments BOTH burn-in
and retained phases. No concurrent allocation-byte measurement is claimed.
"""
module HeterogeneitySimulation
const PERIOD_COLLECTION_INTERVAL = 32
using Random, Statistics
using ..ManyProjectAllocator, ..ManyProjectModel, ..ManyProjectSimulation, ..HeterogeneityModel
const MA=ManyProjectAllocator;const MM=ManyProjectModel;const MS=ManyProjectSimulation;const HM=HeterogeneityModel
const Q=Rational{BigInt};qs(q::Q)=MM.qs(q)
labels_for(f,L)=begin
    base=sum(f.base_counts);L>0 && L%base==0 || error("Nonproportional class counts.")
    vcat([fill(k,n*div(L,base)) for (k,n) in enumerate(f.base_counts)]...)
end
allocators(f::HM.Family,L;limits)=HM.Allocators(f,L;limits=limits)
decide!(w,f::HM.Family,p,types,C)=HM.decide!(w,f,p,types,C)
function initial_states(f,labels,kind::String,seed::UInt64)
    kind in ("zero","cap","relaxed") || error("Unknown initial law.")
    kind=="zero" && return zeros(Int,length(labels))
    kind=="cap" && return fill(f.H,length(labels))
    rng=Xoshiro(seed);cumulative=Vector{BigInt}[];den=BigInt[]
    for k in eachindex(f.weights)
        pi=f.occupations[k];dd=foldl(lcm,(denominator(x) for x in pi);init=BigInt(1))
        cum=cumsum(BigInt[numerator(x*dd) for x in pi]);cum[end]==dd || error("Initial occupancy normalization.")
        push!(cumulative,cum);push!(den,dd)
    end
    [searchsortedfirst(cumulative[k],rand(rng,BigInt(0):(den[k]-1))+1)-1 for k in labels]
end
function diagnostics(w,f,types,actions,C,spent)
    n=count(>(0),types)
    maximum=get!(w.max_spend,n) do;MA.maximum_spend(f.units,n,C);end
    idle=any(types[i]>0 && actions[i]<length(f.units)-1 && C-spent>=f.units[actions[i]+2]-f.units[actions[i]+1] for i in eachindex(types))
    spent<=maximum<=C || error("Attainable expenditure inconsistency.")
    (nonmaximal=idle,belowmax=spent<maximum,attainable_gap_units=maximum-spent)
end
mutable struct Usage
    calls::Int
    seconds::Float64
    max_seconds::Float64
    work::Int128
    peak_states::Int
    engines::Dict{String,Int}
    exact_comparisons::Int
    underlying_engines::Dict{String,Int}
end
Usage()=Usage(0,0.0,0.0,Int128(0),0,Dict{String,Int}(),0,Dict{String,Int}())
function record!(u::Usage,info,dt)
    u.calls+=1;u.seconds+=dt;u.max_seconds=max(u.max_seconds,dt);u.work+=info.work
    u.peak_states=max(u.peak_states,info.peak_states);key=String(info.engine)
    u.engines[key]=get(u.engines,key,0)+1
    hasproperty(info,:exact_comparisons) && (u.exact_comparisons+=info.exact_comparisons)
    if hasproperty(info,:underlying_engine)
        key=String(info.underlying_engine);u.underlying_engines[key]=get(u.underlying_engines,key,0)+1
    end
end
usage(u::Usage)=Dict{String,Any}("calls"=>u.calls,"elapsed_seconds_diagnostic"=>u.seconds,
    "maximum_call_seconds_diagnostic"=>u.max_seconds,"work"=>string(u.work),"peak_reported_states"=>u.peak_states,
    "engine_calls"=>copy(u.engines),"underlying_engine_calls"=>copy(u.underlying_engines),"selective_exact_comparisons"=>u.exact_comparisons,
    "allocation_bytes_not_measured_concurrently"=>true)
const Ledger=MS.Ledger
const count_rows=MS.count_rows
function finalize_checked_ledger(f,ledger::Ledger,L::Int,burn::Int,T::Int,policy::Symbol)
    H=f.H;A=length(f.units)-1;K=length(f.weights);x=ledger.visits;z=ledger.resets
    sum(x)==L*T && all((z .>= 0) .& (z .<= x)) || error("Incomplete transition ledger.")
    for k in 1:K,a in 2:A+1;x[k,1,a]==0 || error("Positive gear at synchronized state.");end
    nextcounts=zeros(Int128,K,H+1);counts=zeros(Int128,K,H+1)
    raw=zero(Q);resource=zero(Q);slack=zero(Q);expected_next=zero(Q);realized_next=zero(Q);shadow=zero(Q)
    capvis=0;tailvis=0;tailcost=zero(Q);tail_start=max(1,cld(3H,4));gearvis=zeros(Int128,A+1)
    for k in 1:K,h in 0:H,a in 0:A
        n=x[k,h+1,a+1];n==0 && continue
        r=z[k,h+1,a+1];s=h==0 ? 1 : min(h+1,H)
        counts[k,h+1]+=n;nextcounts[k,1]+=r;nextcounts[k,s+1]+=n-r
        raw+=n*f.weights[k]*h;resource+=n*f.resource[a+1]
        slack+=n*f.residual[k,h+1,a+1]
        expected_next+=n*f.nextbias[k,h+1,a+1]
        realized_next+=(n-r)*f.bias[k][s+1]
        gearvis[a+1]+=n
        h==H && (capvis+=n)
        if h>=tail_start;tailvis+=n;tailcost+=n*f.weights[k]*h;end
    end
    all((nextcounts .- counts) .== (Int128.(ledger.finish_counts) .- Int128.(ledger.start_counts))) || error("Exact empirical stationary-flow boundary identity failed.")
    before=sum(ledger.start_counts[k,h+1]*f.bias[k][h+1] for k in 1:K,h in 0:H)
    after=sum(ledger.finish_counts[k,h+1]*f.bias[k][h+1] for k in 1:K,h in 0:H)
    B=L*f.budget_per_project;LB=L*f.bound_per_project
    priced=f.lambda*(T*B-resource);boundary=before-after;martingale=realized_next-expected_next
    gap=raw-T*LB;gap==slack+priced+boundary+martingale || error("EXACT pathwise Lagrangian gap decomposition failed.")
    slack>=0 && priced>=0 && resource<=T*B || error("Negative certified slack or aggregate resource infeasibility.")
    policy in (:LAGRANGIAN_NI,:LAGRANGIAN_FC) && ledger.nonmaximal!=0 && error("Non-idling rule left a feasible upgrade.")
    policy==:LAGRANGIAN_FC && ledger.attainable_gap_units!=0 && error("Forced rule did not maximize attainable resource.")
    for k in 1:K;shadow+=ledger.shadow_sum[k]*f.weights[k];end
    shadow>=raw || error("Uncapped shadow cost below capped cost.")
    LB>0 || error("This admitted positive-cost control requires a positive lower bound.")
    den=L*T
    exact=Dict{String,Q}("cost_per_project"=>raw/den,"bound_per_project"=>f.bound_per_project,
        "total_cost"=>raw/T,"total_bound_gap"=>gap/T,"relative_bound_gap"=>gap/(T*LB),
        "raw_bound_gap_per_project"=>gap/den,"charged_action_slack_per_project"=>slack/den,
        "priced_unused_capacity_per_project"=>priced/den,"residual_gap_per_project"=>(slack+priced)/den,
        "charged_bias_boundary_per_project"=>boundary/den,"martingale_per_project"=>martingale/den,
        "bias_adjusted_gap_per_project"=>(gap-boundary)/den,"resource_per_project"=>resource/den,
        "unused_budget_per_project"=>(T*B-resource)/den,"attainable_resource_gap_per_project"=>Q(ledger.attainable_gap_units)*f.tick/den,
        "shadow_cost_per_project"=>shadow/den,"shadow_minus_capped_cost_per_project"=>(shadow-raw)/den,
        "cap_fraction_per_project"=>Q(capvis)/den,"top_tail_fraction_per_project"=>Q(tailvis)/den,
        "top_tail_cost_per_project"=>tailcost/den)
    out=Dict{String,Any}("policy"=>String(policy),"projects"=>L,"burn_in"=>burn,"retained_periods"=>T,
        "exact_pathwise_identity"=>true,"exact_flow_identity"=>true,"hard_budget_violations"=>0,
        "lambda_star"=>qs(f.lambda),"total_budget"=>qs(B),"total_lower_bound"=>qs(LB),
        "nonmaximal_fraction"=>ledger.nonmaximal/T,"below_maximum_resource_fraction"=>ledger.belowmax/T,"any_cap_fraction"=>ledger.cap_any/T,
        "gear_fractions"=>Float64.(gearvis)./den,"start_counts"=>[collect(ledger.start_counts[k,:]) for k in 1:K],
        "finish_counts"=>[collect(ledger.finish_counts[k,:]) for k in 1:K],"visits"=>count_rows(x),"resets"=>count_rows(z),
        "shadow_sum_by_class"=>string.(ledger.shadow_sum),"attainable_gap_units_sum"=>string(ledger.attainable_gap_units),
        "ordered_allocator_calls"=>ledger.ordered_calls,"dp_allocator_calls"=>ledger.dp_calls,"dual_core_allocator_calls"=>ledger.dual_core_calls,
        "priority_allocator_calls"=>ledger.priority_calls,"allocator_work"=>string(ledger.allocator_work),"max_dp_states"=>ledger.max_dp_states,
        "online_allocation_seconds_diagnostic"=>ledger.allocator_seconds,
        "scope"=>"class-specific finite-window Monte Carlo; exact ledger; no stationary or controlled-timing claim",
        "shadow_scope"=>"controller observes capped ages; shadow starts at observed initial age; no uncapped stationary claim",
        "zero_sample_residual_is_not_zero_stationary_gap"=>true)
    for (key,value) in exact;out[key]=Float64(value);out[key*"_exact"]=qs(value);end
    pqv=zero(Q)
    for k in 1:K,h in 0:H,a in 0:A
        n=x[k,h+1,a+1];n==0 && continue
        m=f.classes[k].model;p=h==0 ? m.wR : m.p[a+1];s=h==0 ? 1 : min(h+1,H)
        pqv+=n*p*(1-p)*f.bias[k][s+1]^2
    end
    out["predictable_quadratic_variation_exact"]=qs(pqv)
    out["realized_martingale_sd_scale_per_project"]=sqrt(Float64(pqv))/den
    class_metrics=Dict{String,Any}[]
    for k in 1:K
        nk=sum(ledger.start_counts[k,:]);nk>0 || error("Empty class in trajectory.")
        costk=sum(Q(x[k,h+1,a+1])*f.weights[k]*h for h in 0:H,a in 0:A)
        rk=sum(Q(x[k,h+1,a+1])*f.resource[a+1] for h in 0:H,a in 0:A)
        sk=sum(Q(x[k,h+1,a+1])*f.residual[k,h+1,a+1] for h in 0:H,a in 0:A)
        capk=sum(x[k,H+1,a+1] for a in 0:A)
        push!(class_metrics,Dict{String,Any}("class"=>k,"population"=>nk,
          "cost_per_class_project_exact"=>qs(costk/(nk*T)),"cost_per_class_project"=>Float64(costk/(nk*T)),
          "resource_per_class_project_exact"=>qs(rk/(nk*T)),"resource_per_class_project"=>Float64(rk/(nk*T)),
          "charged_slack_per_class_project_exact"=>qs(sk/(nk*T)),"charged_slack_per_class_project"=>Float64(sk/(nk*T)),
          "cap_fraction_per_class_project_exact"=>qs(Q(capk)/(nk*T)),"cap_fraction_per_class_project"=>Float64(Q(capk)/(nk*T)),
          "gear_fractions"=>[Float64(sum(x[k,:,a+1]))/(nk*T) for a in 0:A]))
    end
    out["class_metrics"]=class_metrics
    out["class_specific_transition_ledger"]=true
    out["quadratic_variation_scope"]="realized predictable variance scale, not an unconditional standard error or p-value"
    out
end
# Semaphore admission releases its internal lock before user work begins. Unlike
# ReentrantLock, it does not inhibit finalizers throughout exact arithmetic.
# The keyword retains its former name for call-site compatibility, but its value
# is now either nothing (single trajectory) or a counting semaphore.
function acquire_postprocess(gate)
    gate === nothing && return nothing
    gate isa Base.Semaphore || throw(ArgumentError("Use nothing or Base.Semaphore(1), not ReentrantLock, for exact postprocessing."))
    Base.acquire(gate)
    nothing
end
function release_postprocess(gate)
    gate === nothing || Base.release(gate)
    nothing
end
function finalize_window_rows(f,ledgers,before,during,states,labels,names,L,rep,w,initial,tseed,iseed;
    collect_between_policies::Bool=false,phase_event=(phase,policy)->nothing)
    rows=Dict{String,Any}[]
    for j in eachindex(names)
        phase_event("exact_policy_start",String(names[j]))
        MS.state_counts!(ledgers[j].finish_counts,states[j],labels)
        # The class-specific finalizer preserves the exact ledger identity.
        row=finalize_checked_ledger(f,ledgers[j],L,w["burn"],w["T"],names[j])
        before[j].calls==w["burn"] && during[j].calls==w["T"] || error("Burn/retained call accounting.")
        row["burn_in_allocator"]=usage(before[j]);row["retained_allocator"]=usage(during[j])
        row["replication"]=rep;row["initial_law"]=initial;row["transition_seed"]=string(tseed);row["initial_seed"]=string(iseed)
        push!(rows,row)
        phase_event("exact_policy_done",String(names[j]))
        # Only completed report values remain live. Collection is an orchestration
        # cost, not a change to the estimators or any controlled timing protocol.
        collect_between_policies && GC.gc(true)
    end
    rows
end

function run_windows(f,L::Int,rep::Int,windows;initial="zero",master_seed=2026092501,
    names=collect(MA.POLICIES),limits=MA.Limits(),seconds=600.0,provided_uniforms=nothing,
    on_window=r->nothing,on_failure=r->nothing,heartbeat=(t,n)->nothing,
    retain_records::Bool=true,postprocess_lock=nothing,period_collected=t->nothing,
    collect_between_policies::Bool=false,phase_event=(phase,policy)->nothing)
    (postprocess_lock===nothing || postprocess_lock isa Base.Semaphore) || throw(ArgumentError("Exact postprocessing requires a semaphore, not a held lock."))
    !isempty(windows) && length(unique((w["burn"],w["T"]) for w in windows))==length(windows) || error("Duplicate or empty windows.")
    all(w->w["burn"]>=0 && w["T"]>0,windows) || error("Invalid windows.")
    !isempty(names) && length(unique(names))==length(names) && all(p->p in MA.POLICIES,names) || error("Invalid policy inventory.")
    1<=L<=1600 && rep>=1 && seconds>0 || error("Invalid bounded trajectory.")
    H=f.H;A=length(f.units)-1;K=length(f.weights);labels=labels_for(f,L)
    total=maximum(w["burn"]+w["T"] for w in windows)
    BigInt(L)*total*(H+total)<typemax(Int64) || error("Trajectory integer range exceeded.")
    C=MA.budget_units(L*f.budget_per_project,f.tick,f.units,L)
    tseed=MS.seed_for(master_seed,f.source_group,L,rep,"transitions-v1")
    iseed=MS.seed_for(master_seed,f.source_group,L,rep,"initial-v1")
    st0=initial_states(f,labels,initial,iseed);states=[copy(st0) for _ in names];shadows=deepcopy(states)
    # A bundle ALREADY contains a distinct workspace for every policy.
    # Policy decisions are sequential within this trajectory. Never share this
    # bundle between trajectory tasks; share only the immutable family tables.
    work=allocators(f,L;limits=limits)
    ledgers=[[Ledger(K,H,A) for _ in names] for _ in windows]
    before=[[Usage() for _ in names] for _ in windows];during=[[Usage() for _ in names] for _ in windows]
    completed=falses(length(windows));out=Dict{String,Any}[];rng=Xoshiro(tseed);U=zeros(UInt64,L);types=zeros(Int,L)
    p=[MS.probability_threshold(f.classes[k].model.p[a+1]) for k in 1:K,a in 0:A];pzero=[MS.probability_threshold(d.model.wR) for d in f.classes]
    provided_uniforms!==nothing && size(provided_uniforms)!=(L,total) && throw(DimensionMismatch("Frozen trajectory shocks."))
    started=time_ns();last_period=0;last_policy="none";last_phase="before trajectory";last_elapsed=0.0;partial_types=Int[]
    try
        for t in 1:total
            (time_ns()-started)/1e9<=seconds || throw(MA.AllocationLimit("Prospective replication wall limit; no shortened accepted trajectory."))
            last_period=t
            provided_uniforms===nothing ? rand!(rng,U) : copyto!(U,view(provided_uniforms,:,t))
            for (j,name) in enumerate(names)
                st=states[j];shadow=shadows[j];last_policy=String(name)
                last_phase=any(w->w["burn"]<t<=w["burn"]+w["T"],windows) ? "retained in at least one window" : "burn-in"
                for (v,w) in enumerate(windows)
                    t==w["burn"]+1 && MS.state_counts!(ledgers[v][j].start_counts,st,labels)
                end
                for i in 1:L;types[i]=st[i]==0 ? 0 : (labels[i]-1)*H+st[i];end
                t0=time_ns()
                acts,info=try
                    decide!(work,f,name,types,C)
                catch
                    partial_types=copy(types);last_elapsed=(time_ns()-t0)/1e9;rethrow()
                end
                dt=(time_ns()-t0)/1e9;last_elapsed=dt
                info.resource_units<=C || error("Realized capacity violation.")
                diag=diagnostics(work,f,types,acts,C,info.resource_units)
                name in (:LAGRANGIAN_NI,:LAGRANGIAN_FC) && diag.nonmaximal && error("Non-idling allocation incomplete.")
                name==:LAGRANGIAN_FC && diag.attainable_gap_units!=0 && error("FC below attainable expenditure.")
                for (v,w) in enumerate(windows)
                    t<=w["burn"] && record!(before[v][j],info,dt)
                    w["burn"]<t<=w["burn"]+w["T"] || continue
                    record!(during[v][j],info,dt);led=ledgers[v][j]
                    led.allocator_seconds+=dt;led.allocator_work+=info.work
                    info.engine in (:integer_dp_exact,:dual_core_dp_exact,:cap_active_exact) && (led.dp_calls+=1;led.max_dp_states=max(led.max_dp_states,info.peak_states))
                    (info.engine==:dual_core_dp_exact || (hasproperty(info,:underlying_engine) && info.underlying_engine==:dual_core_dp_exact)) && (led.dual_core_calls+=1)
                    info.engine in (:ordered_exact,:cap_ordered_verified) && (led.ordered_calls+=1)
                    info.engine==:exposed_priority_exact && (led.priority_calls+=1)
                    led.nonmaximal+=diag.nonmaximal;led.belowmax+=diag.belowmax;led.attainable_gap_units+=diag.attainable_gap_units
                    led.cap_any+=any(==(H),st)
                end
                for i in 1:L
                    h=st[i];a=acts[i];k=labels[i]
                    h==0 && a!=0 && error("Positive gear at zero.")
                    min(shadow[i],H)==h || error("Shadow projection failed.")
                    reset=(U[i]>>11)<(h==0 ? pzero[k] : p[k,a+1])
                    for (v,w) in enumerate(windows)
                        w["burn"]<t<=w["burn"]+w["T"] || continue
                        led=ledgers[v][j];led.visits[k,h+1,a+1]+=1;led.resets[k,h+1,a+1]+=reset;led.shadow_sum[k]+=shadow[i]
                    end
                    st[i]=reset ? 0 : (h==0 ? 1 : min(h+1,H));shadow[i]=reset ? 0 : shadow[i]+1
                end
            end
            # All policy decisions, ledgers and state updates for t are complete.
            # Reclaim transient GMP storage before the next period/report; this
            # fixed orchestration schedule does not enter allocator timings.
            if t % PERIOD_COLLECTION_INTERVAL == 0
                GC.gc(true)
                period_collected(t)
            end
            for (v,w) in enumerate(windows)
                t==w["burn"]+w["T"] || continue
                # No finalizer-inhibiting lock spans this computation or write.
                acquire_postprocess(postprocess_lock)
                try
                rows=finalize_window_rows(f,ledgers[v],before[v],during[v],states,labels,names,L,rep,w,initial,tseed,iseed;
                    collect_between_policies=collect_between_policies,phase_event=phase_event)
                r=Dict{String,Any}("passed"=>true,"status"=>"complete","family"=>f.id,"source_group"=>f.source_group,"contrast"=>f.contrast,"H"=>f.H,"L"=>L,
                   "replication"=>rep,"initial_law"=>initial,"burn_in"=>w["burn"],"horizon"=>w["T"],"window"=>w["name"],
                   "transition_seed"=>string(tseed),"initial_seed"=>string(iseed),"policy_results"=>rows,
                   "policy_names"=>collect(String.(names)),"common_trajectory_windows"=>true,"stationarity_not_asserted"=>true,
                   "diagnostic_elapsed_seconds"=>(time_ns()-started)/1e9,"hard_feasibility_checks_through_window"=>t*length(names),
                   "no_optional_stopping_or_seed_replacement"=>true)
                # Commit before marking complete. Streamed continuation keeps no
                # policy-result dictionaries after their callback returns.
                phase_event("window_commit_start",String(w["name"]))
                on_window(r);completed[v]=true
                phase_event("window_committed",String(w["name"]))
                if retain_records
                    push!(out,r)
                else
                    empty!(ledgers[v])
                    r=nothing;rows=nothing
                end
                finally
                    release_postprocess(postprocess_lock)
                end
                collect_between_policies && GC.gc(true)
            end
            (t==1 || t%1000==0 || t==total) && heartbeat(t,total)
        end
        Dict{String,Any}("passed"=>true,"status"=>"complete","records"=>out,"completed_windows"=>count(completed),
            "periods_run"=>total,"full_system_feasibility_checks"=>total*length(names),"diagnostic_elapsed_seconds"=>(time_ns()-started)/1e9)
    catch err
        r=Dict{String,Any}("passed"=>false,"status"=>(err isa MA.AllocationLimit ? "limit" : "error"),
          "error"=>sprint(showerror,err,catch_backtrace()),"family"=>f.id,"L"=>L,"replication"=>rep,"initial_law"=>initial,
          "last_period"=>last_period,"last_policy"=>last_policy,"last_phase"=>last_phase,"last_allocation_seconds"=>last_elapsed,
          "failed_allocation_types"=>partial_types,"integer_budget"=>C,"resource_units"=>f.units,
          "completed_windows"=>count(completed),"records"=>out,
          "partial_window_use"=>"completed predetermined windows retained; no partial window is a shortened valid estimate",
          "usage_prefix"=>[Dict("window"=>windows[v]["name"],"policy"=>String(names[j]),"burn_in"=>usage(before[v][j]),
            "retained"=>usage(during[v][j])) for v in eachindex(windows) for j in eachindex(names)])
        on_failure(r);r
    end
end
end # module
