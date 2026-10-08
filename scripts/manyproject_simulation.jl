"""Many-project Monte Carlo with no joint-state or action-table enumeration.
An exact state/action/transition-count ledger checks the Lagrangian gap identity
for EACH retained path. Numerical displays and SEs are produced only afterward.
A statistical outlier never changes a seed or becomes a computational failure.
"""
module ManyProjectSimulation
using Random, Statistics, SHA
using ..ManyProjectAllocator, ..ManyProjectModel
const MA=ManyProjectAllocator;const MM=ManyProjectModel;const Q=Rational{BigInt}
qs(x::Q)=MM.qs(x)
function seed_for(master::Integer,group::String,L::Int,rep::Int,purpose::String)
    key=join((string(master),group,string(L),string(rep),purpose),"|")
    parse(UInt64,bytes2hex(sha256(key))[1:16];base=16)
end
function probability_threshold(p::Q)
    0<=p<=1 || throw(ArgumentError("Invalid probability."))
    n=p*(BigInt(2)^53);denominator(n)==1 || error("Probability is not on the admitted exact 53-bit dyadic grid.")
    UInt64(numerator(n))
end
function initial_states(f,labels,kind::String,seed::UInt64)
    H=f.single.model.H;kind in ("zero","cap","relaxed") || throw(ArgumentError("Unknown initial law."))
    kind=="zero" && return zeros(Int,length(labels))
    kind=="cap" && return fill(H,length(labels))
    rng=Xoshiro(seed);K=length(f.weights);cumulative=Vector{BigInt}[];den=BigInt[]
    for k in 1:K
        p=f.occupations[k];dd=foldl(lcm,(denominator(x) for x in p);init=BigInt(1))
        cum=cumsum(BigInt[numerator(x*dd) for x in p]);cum[end]==dd || error("Initial-law normalization.")
        push!(cumulative,cum);push!(den,dd)
    end
    [searchsortedfirst(cumulative[k],rand(rng,BigInt(0):(den[k]-1))+1)-1 for k in labels]
end
mutable struct Ledger
    visits::Array{Int64,3}
    resets::Array{Int64,3}
    start_counts::Matrix{Int64}
    finish_counts::Matrix{Int64}
    shadow_sum::Vector{Int64}
    cap_any::Int
    nonmaximal::Int
    belowmax::Int
    attainable_gap_units::Int128
    allocator_seconds::Float64
    ordered_calls::Int
    dp_calls::Int
    dual_core_calls::Int
    priority_calls::Int
    allocator_work::Int128
    max_dp_states::Int
end
function Ledger(K,H,A)
    Ledger(zeros(Int64,K,H+1,A+1),zeros(Int64,K,H+1,A+1),zeros(Int64,K,H+1),zeros(Int64,K,H+1),
        zeros(Int64,K),0,0,0,Int128(0),0.0,0,0,0,0,Int128(0),0)
end
function state_counts!(out,states,labels)
    fill!(out,0)
    for i in eachindex(states);out[labels[i],states[i]+1]+=1;end
end
# Dictionary rows rather than tuples/multidimensional arrays in final TOML.
function count_rows(x::Array{Int64,3})
    K,H1,A1=size(x)
    [Dict("class"=>k,"state"=>h-1,"gear"=>a-1,"count"=>x[k,h,a]) for k in 1:K for h in 1:H1 for a in 1:A1 if x[k,h,a]!=0]
end
function finalize_ledger(f::MM.Family,ledger::Ledger,L::Int,burn::Int,T::Int,policy::Symbol)
    m=f.single.model;H=m.H;A=length(m.p)-1;K=length(f.weights);x=ledger.visits;z=ledger.resets
    sum(x)==L*T && all((z .>= 0) .& (z .<= x)) || error("Incomplete transition ledger.")
    for k in 1:K,a in 2:A+1;x[k,1,a]==0 || error("Positive gear at synchronized state.");end
    nextcounts=zeros(Int128,K,H+1);counts=zeros(Int128,K,H+1)
    raw=zero(Q);resource=zero(Q);slack=zero(Q);expected_next=zero(Q);realized_next=zero(Q);shadow=zero(Q)
    capvis=0;tailvis=0;tailcost=zero(Q);tail_start=max(1,cld(3H,4));gearvis=zeros(Int128,A+1)
    for k in 1:K,h in 0:H,a in 0:A
        n=x[k,h+1,a+1];n==0 && continue
        r=z[k,h+1,a+1];s=h==0 ? 1 : min(h+1,H)
        counts[k,h+1]+=n;nextcounts[k,1]+=r;nextcounts[k,s+1]+=n-r
        raw+=n*f.weights[k]*h;resource+=n*m.c[a+1]
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
        "scope"=>"finite-window Monte Carlo; exact postprocessed ledger; not a certified stationary estimate or controlled timing",
        "shadow_scope"=>"controller observes capped ages; shadow starts at observed initial age; no uncapped stationary claim",
        "zero_sample_residual_is_not_zero_stationary_gap"=>true)
    for (key,value) in exact;out[key]=Float64(value);out[key*"_exact"]=qs(value);end
    out
end
function run_replication(f::MM.Family,L::Int,rep::Int;burn::Int,T::Int,initial::String="zero",master_seed::Int=20260924,
        names=collect(MA.POLICIES),limits=MA.Limits(),seconds::Float64=300.0,provided_uniforms=nothing)
    length(names)>0 && length(unique(names))==length(names) && all(p in MA.POLICIES for p in names) || error("Invalid policy set.")
    burn>=0 && T>0 && rep>0 && L<=1600 || error("Invalid bounded preflight dimensions.")
    H=f.single.model.H;A=length(f.single.model.p)-1;K=length(f.weights);labels=MM.class_labels(f,L)
    BigInt(L)*T*(H+burn+T)<typemax(Int64) || error("Sufficient-statistic integer range exceeded.")
    C=MA.budget_units(L*f.budget_per_project,f.tick,f.units,L)
    tseed=seed_for(master_seed,f.source_group,L,rep,"transitions-v1")
    iseed=seed_for(master_seed,f.source_group,L,rep,"initial-v1")
    startstate=initial_states(f,labels,initial,iseed)
    states=[copy(startstate) for p in names];shadows=deepcopy(states)
    alloc=[MM.Allocators(f,L;limits=limits) for _ in names];ledgers=[Ledger(K,H,A) for _ in names]
    rng=Xoshiro(tseed);U=zeros(UInt64,L);types=zeros(Int,L)
    thresh=probability_threshold.(f.single.model.p);zero_threshold=probability_threshold(f.single.model.wR)
    checked=0;started=time_ns()
    if provided_uniforms!==nothing;size(provided_uniforms)==(L,burn+T) || throw(DimensionMismatch("Frozen shock array shape."));end
    for t in 1:burn+T
        (time_ns()-started)/1e9<=seconds || throw(MA.AllocationLimit("Replication cooperative wall limit; no shortened estimate accepted."))
        # One shock per project and time, generated OUTSIDE the policy loop.
        # Explicit RNG is private to this replication. Adding policies cannot
        # advance an old policy's random stream. Bulk shape and Julia are pinned.
        if provided_uniforms===nothing;rand!(rng,U)
        else;copyto!(U,view(provided_uniforms,:,t));end
        for (j,p) in enumerate(names)
            st=states[j];shadow=shadows[j];led=ledgers[j]
            if t==burn+1;state_counts!(led.start_counts,st,labels);end
            for i in 1:L;types[i]=st[i]==0 ? 0 : (labels[i]-1)*H+st[i];end
            t0=time_ns();acts,info=MM.decide!(alloc[j],f,p,types,C);dt=(time_ns()-t0)/1e9
            info.resource_units<=C || error("Realized hard-budget violation.")
            diag=MM.capacity_diagnostics(alloc[j],f,types,acts,C,info.resource_units)
            checked+=1
            p==:LAGRANGIAN_FC && diag.attainable_gap_units!=0 && error("Forced allocation leaves attainable expenditure unused.")
            if t>burn
                led.allocator_seconds+=dt;led.allocator_work+=info.work
                info.engine in (:integer_dp_exact,:dual_core_dp_exact) && (led.max_dp_states=max(led.max_dp_states,info.peak_states))
                info.engine==:dual_core_dp_exact && (led.dual_core_calls+=1)
                info.engine==:ordered_exact && (led.ordered_calls+=1)
                info.engine in (:integer_dp_exact,:dual_core_dp_exact) && (led.dp_calls+=1)
                info.engine==:exposed_priority_exact && (led.priority_calls+=1)
                led.nonmaximal+=diag.nonmaximal;led.belowmax+=diag.belowmax;led.attainable_gap_units+=diag.attainable_gap_units
                led.cap_any+=any(==(H),st)
            end
            for i in 1:L
                h=st[i];a=acts[i];k=labels[i]
                h==0 && a!=0 && error("State zero must use gear zero.")
                min(shadow[i],H)==h || error("Shadow/capped projection identity failed.")
                reset=(U[i]>>11)<(h==0 ? zero_threshold : thresh[a+1])
                if t>burn
                    led.visits[k,h+1,a+1]+=1;led.resets[k,h+1,a+1]+=reset;led.shadow_sum[k]+=shadow[i]
                end
                st[i]=reset ? 0 : (h==0 ? 1 : min(h+1,H))
                shadow[i]=reset ? 0 : shadow[i]+1
            end
        end
    end
    rows=Dict{String,Any}[]
    for j in eachindex(names)
        state_counts!(ledgers[j].finish_counts,states[j],labels)
        r=finalize_ledger(f,ledgers[j],L,burn,T,names[j]);r["replication"]=rep;r["initial_law"]=initial
        r["transition_seed"]=string(tseed);r["initial_seed"]=string(iseed);push!(rows,r)
    end
    Dict{String,Any}("passed"=>true,"family"=>f.id,"source_group"=>f.source_group,"L"=>L,"replication"=>rep,"initial_law"=>initial,
        "burn_in"=>burn,"horizon"=>T,"transition_seed"=>string(tseed),"initial_seed"=>string(iseed),
        "policy_names"=>collect(String.(names)),"policy_results"=>rows,"hard_feasibility_checks"=>checked,
        "diagnostic_elapsed_seconds"=>(time_ns()-started)/1e9,"common_random_numbers"=>true,"random_streams_private"=>true,
        "stationarity_not_asserted"=>true,"no_optional_stopping_or_seed_replacement"=>true)
end
function stats(v)
    length(v)>=2 || error("At least two independent replication observations needed.")
    (mean=mean(v),se=std(v;corrected=true)/sqrt(length(v)))
end
const METRICS=("cost_per_project","total_cost","total_bound_gap","relative_bound_gap","raw_bound_gap_per_project","residual_gap_per_project","charged_action_slack_per_project",
    "priced_unused_capacity_per_project","charged_bias_boundary_per_project","martingale_per_project","bias_adjusted_gap_per_project",
    "resource_per_project","unused_budget_per_project","nonmaximal_fraction","attainable_resource_gap_per_project",
    "cap_fraction_per_project","top_tail_fraction_per_project","shadow_minus_capped_cost_per_project","online_allocation_seconds_diagnostic")
function summarize(records::Vector{Dict{String,Any}})
    good=filter(r->get(r,"passed",false),records);groups=Dict{Tuple{String,Int,String,Int,Int},Vector{Dict{String,Any}}}()
    for r in good
        key=(String(r["family"]),Int(r["L"]),String(r["initial_law"]),Int(r["burn_in"]),Int(r["horizon"]))
        push!(get!(groups,key,Dict{String,Any}[]),r)
    end
    summaries=Dict{String,Any}[];pairs=Dict{String,Any}[]
    for key in sort!(collect(keys(groups)))
        rr=sort(groups[key];by=r->r["replication"])
        length(rr)>=2 || continue
        # Caller checks expected cardinality; incomplete groups are NEVER treated
        # as complete prescribed estimates, even when two replications exist.
        expected=Int(rr[1]["planned_replications"])
        length(rr)==expected || continue
        length(unique(r["replication"] for r in rr))==expected || error("Duplicate replication record.")
        lookup=[Dict(x["policy"]=>x for x in r["policy_results"]) for r in rr]
        for p in MA.POLICIES
            name=String(p);base=Dict{String,Any}("family"=>key[1],"L"=>key[2],"initial_law"=>key[3],"burn_in"=>key[4],"horizon"=>key[5],"replications"=>expected,"policy"=>name,
                "bound_per_project"=>lookup[1][name]["bound_per_project"],"lambda_star"=>lookup[1][name]["lambda_star"],"scope"=>"precision pilot; SE across independent replications, not across periods")
            for metric in METRICS
                values=Float64[x[name][metric] for x in lookup];s=stats(values)
                base[metric*"_mean"]=s.mean;base[metric*"_se"]=s.se
            end
            push!(summaries,base)
        end
        names=collect(String.(MA.POLICIES))
        for j in 2:length(names),i in 1:j-1
            # Difference second minus first; negative favors the second policy.
            values=Float64[MM.parseq(x[names[j]]["cost_per_project_exact"])-MM.parseq(x[names[i]]["cost_per_project_exact"]) for x in lookup];s=stats(values)
            push!(pairs,Dict{String,Any}("family"=>key[1],"L"=>key[2],"initial_law"=>key[3],"burn_in"=>key[4],"horizon"=>key[5],"replications"=>expected,
                "comparison"=>names[j]*"_minus_"*names[i],"mean_cost_difference_per_project"=>s.mean,"paired_monte_carlo_se"=>s.se))
        end
    end
    summaries,pairs
end
end # module
