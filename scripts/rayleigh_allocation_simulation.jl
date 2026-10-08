# Included inside RayleighAllocation. Simulations validate the convention/estimator
# against exact stationary costs. They do NOT solve the knapsack each period:
# previously validated tiny-state action tables are used. No claim about online
# allocation latency or large-system scaling is derived from this simulation.

function finite_window(P::Matrix{Q},g::Q,b::Vector{Q},burn::Int,T::Int)
    n=size(P,1);initial=zeros(Float64,n);initial[1]=1.0
    Pt=transpose(Float64.(P));start=(Pt^burn)*initial;finish=(Pt^T)*start
    err=max(abs(sum(start)-1),abs(sum(finish)-1))
    require0(err<=1e-9 && minimum(start)>=-1e-12 && minimum(finish)>=-1e-12,"finite-window distribution arithmetic")
    mean=Float64(g)+(dot(start,Float64.(b))-dot(finish,Float64.(b)))/T
    (mean=mean,start=start,finish=finish,mass_error=err)
end

function mean_se(v::AbstractVector{<:Real})
    length(v)>=2 || error("At least two independent replications required.")
    (mean=mean(v),se=std(v;corrected=true)/sqrt(length(v)))
end

function simulate_system(j::JointData,dual::DualData,evaluations,config;seed::UInt64,seconds_limit::Real=Inf,policy_names::Tuple=ALL_POLICIES)
    started=time_ns();m=j.single.model;H=m.H;L=length(j.weights);n=length(j.states)
    R=Int(config["replications"]);burn=Int(config["burn_in"]);T=Int(config["horizon"])
    R>=2 && burn>=0 && T>0 || error("Invalid Monte Carlo dimensions.")
    names=collect(policy_names);K=length(names)
    K>0 && length(unique(names))==K && all(name in ALL_POLICIES && haskey(evaluations,name) for name in names) || error("Unknown or duplicated simulation policy.")
    # Scale only for EXACT budget checks, not for rounding resource costs.
    denominator0=foldl(lcm,[denominator(x) for x in vcat(m.c,[j.budget])];init=BigInt(1))
    units_q=m.c.*denominator0;budget_q=j.budget*denominator0
    require0(all(denominator(x)==1 && 0<=numerator(x)<=typemax(Int) for x in units_q) &&
        denominator(budget_q)==1 && 0<=numerator(budget_q)<=typemax(Int),"exact integer budget encoding")
    units=Int[numerator(x) for x in units_q];budget=Int(numerator(budget_q));unit_scale=Float64(denominator0)
    spend=zeros(Int,n,K);reset=zeros(Float64,n,L,K)
    nonmaximal=zeros(Float64,n,K);belowmax=zeros(Float64,n,K);attainable_gap=zeros(Float64,n,K)
    for (k,name) in enumerate(names),i in 1:n
        e=evaluations[name];actions=j.actions[i][e.pi[i]]
        total=sum(BigInt(units[a+1]) for a in actions)
        require0(total<=budget && total<=typemax(Int),"simulation table budget")
        spend[i,k]=Int(total)
        nonmaximal[i,k]=Float64(e.capacity.nonmaximal[i]);belowmax[i,k]=Float64(e.capacity.belowmax[i])
        attainable_gap[i,k]=Float64(e.capacity.gaps[i])
        name in (:LAGRANGIAN_NI,:LAGRANGIAN_FC) && require0(nonmaximal[i,k]==0,"non-idling simulation table")
        name==:LAGRANGIAN_FC && require0(attainable_gap[i,k]==0,"maximum-expenditure simulation table")
        for ell in 1:L
            reset[i,ell,k]=Float64(j.states[i][ell]==0 ? m.wR : m.p[actions[ell]+1])
        end
    end
    costs=Float64.(j.costs);cap=[any(==(H),x) ? 1.0 : 0.0 for x in j.states]
    lbias=Float64[sum(dual.bias[ell][h+1] for (ell,h) in enumerate(x)) for x in j.states]
    observations=Dict{String,Any}[];costmatrix=zeros(Float64,R,K);resmatrix=similar(costmatrix)
    checked=0;transitions=0
    for rep in 1:R
        (time_ns()-started)/1e9<=seconds_limit || error("Simulation cooperative budget exceeded; no optional stopping acceptance.")
        # Replication streams independent; one uniform per project/time is shared
        # by every policy. This couples outcomes without changing any marginal law.
        repseed=seed+UInt64(104729)*UInt64(rep);rng=Xoshiro(repseed)
        states=ones(Int,K);U=zeros(Float64,L);sums=zeros(Float64,K);resource=zeros(Float64,K)
        caps=zeros(Float64,K);starts=zeros(Float64,K)
        idle=zeros(Float64,K);below=zeros(Float64,K);gaps=zeros(Float64,K)
        for t in 1:burn+T
            rand!(rng,U)
            for k in 1:K
                s=states[k];spend[s,k]<=budget || error("Realized budget violation.")
                checked+=1
                if t==burn+1;starts[k]=lbias[s];end
                if t>burn
                    sums[k]+=costs[s];resource[k]+=Float64(spend[s,k])/unit_scale;caps[k]+=cap[s]
                    idle[k]+=nonmaximal[s,k];below[k]+=belowmax[s,k];gaps[k]+=attainable_gap[s,k]
                end
                # Lexicographic joint-state enumeration: last project varies fastest.
                dest=0
                for ell in 1:L
                    h=j.states[s][ell]
                    next=U[ell]<reset[s,ell,k] ? 0 : (h==0 ? 1 : min(h+1,H))
                    dest=dest*(H+1)+next;transitions+=1
                end
                states[k]=dest+1
            end
        end
        for (k,name) in enumerate(names)
            value=sums[k]/T;res=resource[k]/T;costmatrix[rep,k]=value;resmatrix[rep,k]=res
            correction=(starts[k]-lbias[states[k]])/T
            push!(observations,Dict("replication"=>rep,"seed"=>string(repseed),"policy"=>String(name),
                "cost_mean"=>value,"resource_mean"=>res,"cap_any_fraction"=>caps[k]/T,
                "nonmaximal_fraction"=>idle[k]/T,"below_maximum_resource_fraction"=>below[k]/T,
                "mean_attainable_resource_gap"=>gaps[k]/T,"mean_unused_budget"=>Float64(j.budget)-res,
                "charged_bias_boundary_correction"=>correction,
                "bias_adjusted_cost_minus_bound"=>value-correction-Float64(dual.bound),
                "start_charged_bias"=>starts[k],"end_charged_bias"=>lbias[states[k]],
                "end_joint_state"=>j.states[states[k]],"hard_budget_violations"=>0))
        end
    end
    summaries=Dict{String,Any}[]
    for (k,name) in enumerate(names)
        e=evaluations[name];f=finite_window(e.P,e.cost,e.bias,burn,T)
        fr=finite_window(e.P,e.resource,e.resource_bias,burn,T)
        fc=finite_window(e.P,e.cap_any,e.cap_bias,burn,T)
        fi=finite_window(e.P,e.capacity.idling_probability,e.capacity.idling_bias,burn,T)
        fg=finite_window(e.P,e.capacity.mean_attainable_gap,e.capacity.gap_bias,burn,T)
        fm=finite_window(e.P,e.capacity.belowmax_probability,e.capacity.belowmax_bias,burn,T)
        expected_correction=(dot(f.start,lbias)-dot(f.finish,lbias))/T
        require0(f.mean-expected_correction>=Float64(dual.bound)-1e-9,"finite-window expected Lagrangian inequality")
        r=mean_se(view(costmatrix,:,k));rr=mean_se(view(resmatrix,:,k))
        localrows=[x for x in observations if x["policy"]==String(name)]
        adjusted=mean_se([x["bias_adjusted_cost_minus_bound"] for x in localrows])
        idle_stats=mean_se([x["nonmaximal_fraction"] for x in localrows])
        gap_stats=mean_se([x["mean_attainable_resource_gap"] for x in localrows])
        push!(summaries,Dict("policy"=>String(name),"empirical_cost_mean"=>r.mean,"monte_carlo_se"=>r.se,
            "exact_stationary_cost"=>qs(e.cost),"finite_window_expected_cost"=>f.mean,
            "finite_window_stationary_bias"=>f.mean-Float64(e.cost),
            "empirical_resource_mean"=>rr.mean,"resource_monte_carlo_se"=>rr.se,
            "empirical_nonmaximal_probability"=>idle_stats.mean,"nonmaximal_monte_carlo_se"=>idle_stats.se,
            "empirical_mean_attainable_resource_gap"=>gap_stats.mean,"attainable_gap_monte_carlo_se"=>gap_stats.se,
            "empirical_below_maximum_resource_probability"=>mean([x["below_maximum_resource_fraction"] for x in localrows]),
            "exact_nonmaximal_probability"=>qs(e.capacity.idling_probability),
            "exact_below_maximum_resource_probability"=>qs(e.capacity.belowmax_probability),
            "exact_mean_attainable_resource_gap"=>qs(e.capacity.mean_attainable_gap),
            "finite_window_expected_nonmaximal_probability"=>fi.mean,
            "finite_window_expected_attainable_gap"=>fg.mean,
            "finite_window_expected_below_maximum_resource_probability"=>fm.mean,
            "finite_window_expected_resource"=>fr.mean,"finite_window_expected_cap_any"=>fc.mean,
            "exact_stationary_cap_any"=>qs(e.cap_any),"empirical_cap_any"=>mean([x["cap_any_fraction"] for x in localrows]),
            "empirical_bias_adjusted_gap"=>adjusted.mean,"bias_adjusted_gap_se"=>adjusted.se,
            "finite_window_expected_bias_adjusted_gap"=>f.mean-expected_correction-Float64(dual.bound),
            "floating_distribution_mass_error"=>max(f.mass_error,fr.mass_error,fc.mass_error,fi.mass_error,fg.mass_error,fm.mass_error),
            "scope"=>"MC standard errors describe independent finite-window replications; not initialization or cap-truncation error bounds"))
    end
    differences=Dict{String,Any}[]
    pairs=[(name,:WHITTLE_K) for name in names if name!=:WHITTLE_K && :WHITTLE_K in names]
    for pair in ((:LAGRANGIAN_NI,:LAGRANGIAN_K),(:LAGRANGIAN_FC,:LAGRANGIAN_K),(:LAGRANGIAN_FC,:LAGRANGIAN_NI))
        all(name in names for name in pair) && push!(pairs,pair)
    end
    for (a,b) in pairs
        ka=findfirst(==(a),names);kb=findfirst(==(b),names)
        v=costmatrix[:,ka].-costmatrix[:,kb];r=mean_se(v)
        push!(differences,Dict("comparison"=>String(a)*" minus "*String(b),"mean_difference"=>r.mean,
            "paired_monte_carlo_se"=>r.se,"exact_stationary_difference"=>qs(evaluations[a].cost-evaluations[b].cost)))
    end
    span=maximum(dualbias for dualbias in Q[sum(dual.bias[ell][h+1] for (ell,h) in enumerate(x)) for x in j.states])-
        minimum(dualbias for dualbias in Q[sum(dual.bias[ell][h+1] for (ell,h) in enumerate(x)) for x in j.states])
    Dict{String,Any}("executed"=>true,"passed"=>true,"replications"=>R,"burn_in"=>burn,"horizon"=>T,
        "policies"=>String.(names),"seed_base"=>string(seed),"common_random_numbers"=>true,"initial_state"=>zeros(Int,L),
        "period_allocation_feasibility_checks"=>checked,"project_transition_updates"=>transitions,"uniform_random_values"=>R*(burn+T)*L,
        "violations"=>0,"resource_denominator"=>string(denominator0),"budget_integer_units"=>budget,
        "uniform_expected_T_window_bound"=>qs(dual.bound-span/T),
        "finite_horizon_bound_scope"=>"bound on expected cost, NOT on every sample path; MC gaps may be negative and are never clipped",
        "replication_observations"=>observations,"summaries"=>summaries,"paired_policy_differences"=>differences,
        "diagnostic_elapsed_seconds"=>(time_ns()-started)/1e9,
        "online_solver_called_each_period"=>false,
        "simulation_scope"=>"precomputed exact tiny-state action tables; validates transitions/cost conventions/estimation, not large-system online allocation speed")
end
