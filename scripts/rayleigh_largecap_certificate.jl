# Included in RayleighLargeCap. Independent per-state scalar recursions, NOT the
# block evaluator. O(A*H^2) adjacent checks over AH boundary policies; O(H+AH)
# retained storage. No threshold-family or all-policy enumeration. All proof work
# is outside numerical timers. See docs/RAYLEIGH_LARGECAP_CERTIFICATE.md.

function primitive_lemma(m::AoIIModel{Q})
    H=m.H;A=length(m.p)-1
    H>=1 && 0<1-m.wR<1 && all(0<p<1 for p in m.p) && m.c[1]==0 || error("Invalid primitive topology.")
    slopes=Q[(m.c[a+1]-m.c[a])/(m.p[a+1]-m.p[a]) for a in 1:A]
    all(m.p[a+1]>m.p[a] && m.c[a+1]>m.c[a] for a in 1:A) && issorted(slopes) || error("Ordered menu/slopes required.")
    gamma=maximum(m.c[a]/m.p[a] for a in 1:A+1)
    margin=slopes[1]-gamma
    margin>0 || error("Primitive all-policy marginal-resource bound fails; no certificate.")
    (slopes=slopes,gamma=gamma,margin=margin)
end
function scalar_enclosures!(t,w,e,phi,gamma,m,actions,p,q,c,u)
    H=m.H;a=actions[H]+1
    t[H]=RB.BONE/p[a];w[H]=E(H)/p[a];e[H]=c[a]/p[a]
    for h in H-1:-1:1
        a=actions[h]+1;t[h]=RB.BONE+q[a]*t[h+1]
        w[h]=E(h)+q[a]*w[h+1];e[h]=c[a]+q[a]*e[h+1]
    end
    den=RB.BONE+u*t[1];hb=u*w[1]/den;cb=u*e[1]/den
    for h in 1:H;phi[h]=w[h]-hb*t[h];gamma[h]=e[h]-cb*t[h];end
    hb,cb
end

"""Certify an arbitrary supplied nested path independently, at every defining
switch charge. Each root is defined EXACTLY by the current policy and selected
adjacent pair; intervals enclose it but are never substituted as the root.

All other action inequalities are established through ordered adjacent slopes.
At a common successor the numerator collapses algebraically; exact cap ties
are not tested by rounding or sampled lambda values. Remaining uncertain scalar
inequalities use a bounded exact current-policy fallback, not an assumed tie.
"""
function certify_chain(m::AoIIModel{Q},order,c;progress=(k,n)->nothing)
    RB.ieee_check();H=m.H;A=length(m.p)-1;K=A*H
    length(order)==K || error("Incomplete candidate path; no certificate.")
    lemma=primitive_lemma(m);slopes=lemma.slopes;rs=E.(slopes)
    signs=[sign(slopes[j]-slopes[a]) for j in 1:A,a in 1:A]
    p=E.(m.p);q=E.(1 .-m.p);cc=E.(m.c);u=E(1-m.wR)
    actions=fill(A,H);t=Vector{E}(undef,H);w=similar(t);e=similar(t);phi=similar(t);gam=similar(t)
    limits=c["certificate"];records=Dict{String,Any}[];started=time_ns()
    adjacent_checks=0;symbolic_zeros=0;symbolic_signs=0;interval_checks=0;exact_checks=0;fallbacks=0
    minimum_strict=nothing
    for (k,pair) in enumerate(order)
        RB.ieee_check()
        (time_ns()-started)/1e9<=limits["seconds_per_case"] || error("Certificate cooperative time limit; prefix retained in progress only.")
        h,a=Int(pair[1]),Int(pair[2])
        1<=h<=H && 1<=a<=A && actions[h]==a && (h==1 || actions[h-1]<a) ||
            error("Nonadjacent/nonthreshold/repeated candidate step $k: ($h,$a).")
        scalar_enclosures!(t,w,e,phi,gam,m,actions,p,q,cc,u)
        successor=min(h+1,H);fp=phi[successor];dp=rs[a]-gam[successor]
        exact_e=nothing
        function exact_policy!()
            if exact_e===nothing
                fallbacks<limits["maximum_exact_policy_fallbacks"] && H<=limits["exact_max_H"] && A<=limits["exact_max_A"] ||
                    error("Unresolved certificate outside its saved exact-policy budget at step $k.")
                all(max(ndigits(abs(numerator(x));base=2),ndigits(denominator(x);base=2))<=limits["exact_input_bits"]
                    for x in vcat(m.p,m.c,[1-m.wR])) || error("Certificate source bit budget exceeded.")
                fallbacks+=1;exact_e=excursion(m,actions)
            end
            exact_e
        end
        if fp.lo<=0 || dp.lo<=0
            ex=exact_policy!();fp=E(ex.phi[successor]);dp=E(slopes[a]-ex.gamma[successor])
        end
        fp.lo>0 && dp.lo>0 || error("Positive defining root not enclosed.")
        root=fp/dp
        if !RB.accurate(root,limits["root_atol"],limits["root_rtol"])
            ex=exact_policy!();root=E(ex.phi[successor]/(slopes[a]-ex.gamma[successor]))
        end
        RB.accurate(root,limits["root_atol"],limits["root_rtol"]) || error("Certificate root enclosure accuracy unresolved.")
        for age in 1:H
            b=actions[age];s=min(age+1,H)
            # b-1 needs N_b <= 0; b+1 needs N_(b+1) >= 0. Ordered
            # reset/resource slopes and root>0 imply ALL nonadjacent inequalities.
            for (j,direction) in ((b,-1),(b+1,1))
                1<=j<=A || continue
                adjacent_checks+=1
                if s==successor
                    sg=direction*signs[j,a]
                    sg>=0 || error("Exact same-successor Bellman violation at step $k, age $age, increment $j.")
                    if sg==0;symbolic_zeros+=1;else;symbolic_signs+=1;end
                    continue
                end
                z=fp*(rs[j]-gam[s])-phi[s]*dp
                low=direction==1 ? z.lo : -z.hi
                high=direction==1 ? z.hi : -z.lo
                high<0 && error("Strict certified Bellman violation at step $k, age $age, increment $j.")
                if low>=0
                    interval_checks+=1
                    low>0 && (minimum_strict=minimum_strict===nothing ? low : min(minimum_strict,low))
                else
                    ex=exact_policy!()
                    value=direction*(ex.phi[successor]*(slopes[j]-ex.gamma[s])-ex.phi[s]*(slopes[a]-ex.gamma[successor]))
                    value>=0 || error("Exact Bellman violation at step $k, age $age, increment $j.")
                    exact_checks+=1
                end
            end
        end
        # The switch action's equality is an algebraic identity at the root.
        # It gives the same optimal gain and normalized bias before and after
        # switching. The proof then orders consecutive roots without comparing
        # rounded numbers or declaring overlapping intervals to be ties.
        push!(records,Dict{String,Any}("step"=>k,"age"=>h,"upper_gear"=>a,
            "root_value"=>root.value,"root_lo"=>root.lo,"root_hi"=>root.hi,
            "root_definition"=>"phi_S[min(h+1,H)]/(r_a-gamma_S[min(h+1,H)])"))
        actions[h]-=1
        (k==1 || k%128==0 || k==K) && progress(k,K)
    end
    all(iszero,actions) || error("Path does not end all-passive.")
    Dict{String,Any}("passed"=>true,"dai_verified"=>true,"interval_indexability_verified"=>true,
      "certificate_type"=>"primitive resource lemma plus whole-charge Bellman boundary-chain certificate",
      "full_PCL_family_certificate"=>false,"all_policy_resource_positivity"=>true,
      "threshold_family_enumerated"=>false,"all_policy_enumeration"=>false,
      "all_real_charges"=>true,"nondecreasing_exact_switches"=>"proved inductively; not sorted rounded values",
      "ordered_resource_slopes_exact"=>string.(slopes),"Gamma_c_exact"=>string(lemma.gamma),
      "resource_margin_exact"=>string(lemma.margin),"path"=>records,"complete_assignments"=>K,
      "policy_intervals"=>K+1,"explicit_adjacent_inequalities"=>adjacent_checks,
      "implied_all_action_inequalities_at_boundaries"=>K*H*A,
      "symbolic_same_successor_equalities"=>symbolic_zeros,"symbolic_same_successor_strict_checks"=>symbolic_signs,
      "interval_sign_checks"=>interval_checks,"exact_inequality_checks"=>exact_checks,
      "exact_policy_fallbacks"=>fallbacks,
      "minimum_strict_interval_margin"=>(minimum_strict===nothing ? "none" : minimum_strict),
      "lower_tail"=>"all-active policy: every alternative resource slope is negative",
      "upper_tail"=>"all-passive policy: every nonzero gear has positive resource slope",
      "duration_seconds"=>(time_ns()-started)/1e9,
      "scope"=>"fixed instance's optimal nested chain/DAI; NOT a whole-family PCLI2 or uniform-in-cap theorem")
end
function certify_model(spec,c;progress=(k,n)->nothing,save_candidate=s->nothing)
    m=specification_model(spec)
    # Untrusted candidate generated outside timing. The checker never uses the
    # candidate's marginal values to decide Bellman signs. A failure is retained;
    # no alternate model, menu or seed is searched for automatically.
    candidate_started=time_ns()
    r=lc_prepare(SparseContext(m,RT.block_options(c)),Val(:BLOCK));lc_run!(r);s=lc_snapshot(r)
    candidate_seconds=(time_ns()-candidate_started)/1e9
    save_candidate(s)
    s.complete || error("Candidate path unavailable: $(r.status): $(r.message)")
    core=certify_chain(m,[(h-1,a) for (h,a) in s.order],c;progress=progress)
    core["specification"]=Dict(spec);core["model"]=RT.model_record(m)
    core["float64_model_exactly_identical"]=true
    core["candidate_method"]="accepted autonomous BLOCK, untrusted until independent certificate"
    core["candidate_counts"]=s.counts
    core["candidate_generation_seconds"]=candidate_seconds
    core["certificate_timing_scope"]="candidate generation and independent verification recorded separately; both outside algorithm timers"
    snapshot_check(s,core,c) || error("Candidate's displayed indices miss the independent certified roots.")
    # Bounds for the chance to reach the cap in an excursion starting at age 1.
    # This is neither a stationary-occupancy estimate nor a truncation-cost bound.
    low=RB.geometric(E(1-m.p[end]),m.H-1).power
    high=RB.geometric(E(1-m.p[1]),m.H-1).power
    core["excursion_reach_cap_bounds"]=Dict("lower"=>low.lo,"upper"=>high.hi,
        "formula"=>"(1-p_A)^(H-1) <= P_1^policy(reach H before reset) <= (1-p_0)^(H-1)",
        "scope"=>"all admissible policies; not a stationary or truncation-error bound")
    core
end
