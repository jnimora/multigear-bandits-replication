# Scalable alternative to the exact dense inverse-residual certificate.
# Float64 LU is still dense, shared by all methods, and charged to loop time.
# Exact residual arithmetic visits nonzeros without allocating a rational N^2
# matrix. No model-specific path, transition formula, or index is used.
#
# Discounted: ||v-x||inf <= ||b-(I-beta P)x||inf/(1-beta ||P||inf).
# Average: alpha=sum_j min_i P_ij > 0 implies span(Pz)<=(1-alpha)span(z).
# For stochastic stored rows, r=b-(I-P)x-rho*1 gives
# span(v-x)<=span(r)/alpha. An adjacent transition difference d with sum(d)=0
# therefore satisfies |d'(v-x)| <= ||d||1 span(r)/(2 alpha).
# This bound is invariant to bias normalization and does not require an inverse.
# All row sums, alpha, residuals, and marginal/ratio bounds are exact rationals
# of the STORED floating primitives. Nonstochastic average rows are refused.
function _contraction_certified_selection(model::FiniteMultiGearModel{Float64},actions,
        frontier;criterion::Symbol=:discounted,beta=nothing,omega=nothing)
    criterion in (:discounted,:average) || throw(ArgumentError("Unknown criterion."))
    n=nstates(model);p=policy_data(model,actions);Q=_ResidualQ
    pairs=Tuple{Int,Int}[(Int(t[1]),Int(t[2])) for t in frontier]
    length(unique(pairs))==length(pairs) || throw(ArgumentError("Repeated frontier pair."))
    for (i,a) in pairs
        1<=i<=n && model.controllable[i] && a==actions[i] && 1<=a<=maxgear(model) ||
            throw(ArgumentError("Frontier is not a current adjacent comparison."))
    end
    factor=criterion==:average ? 1.0 : Float64(beta)
    criterion==:discounted && !(isfinite(factor) && 0<factor<1) &&
        throw(ArgumentError("Invalid discount."))
    b=Q(factor);d=n+(criterion==:average)
    report=Dict{String,Any}("scope"=>"current-frontier selection for exact stored-Float64 system; not path or PCL certification",
        "revision"=>"contraction-residual-v1","criterion"=>string(criterion),"dimension"=>d,
        "frontier_pairs"=>[[i,a] for (i,a) in pairs],"factorizations"=>0,"inverse_rhs_columns"=>0,
        "arithmetic"=>"Float64 LU; exact rational residuals and contraction bounds; no rational dense matrix or inverse",
        "status"=>"unresolved_comparison")
    unresolved(reason)=(report["reason"]=reason;(position=0,value=NaN,status=:unresolved_comparison,report=report))
    # Column traversal reads the dense policy matrix contiguously. Zero entries
    # do not trigger rational conversions or products, including in residuals.
    masses=fill(zero(Q),n);alpha=zero(Q)
    for j in 1:n
        smallest=Inf
        for i in 1:n
            z=p.P[i,j];smallest=min(smallest,z)
            z==0 || (masses[i]+=Q(z))
        end
        smallest==0 || (alpha+=Q(smallest))
    end
    if criterion==:average
        all(==(one(Q)),masses) || return unresolved("Stored average policy rows are not exactly stochastic.")
        alpha>0 || return unresolved("No strictly positive common transition mass.")
        report["bound_kind"]="exact common-mass span contraction"
        report["common_mass_exact"]=string(alpha)
    else
        b*maximum(masses)<1 || return unresolved("Discounted row-sum contraction cannot be certified.")
        report["bound_kind"]="exact discounted row-sum contraction"
        report["inverse_norm_upper_exact"]=string(1/(1-b*maximum(masses)))
    end
    M=zeros(d,d);B=zeros(d,2)
    for j in 1:n,i in 1:n;M[i,j]=(i==j)-factor*p.P[i,j];end
    B[1:n,1]=p.h;B[1:n,2]=p.c
    if criterion==:average
        w=omega===nothing ? vcat(1.0,zeros(n-1)) : Float64.(omega)
        length(w)==n && all(isfinite,w) && abs(sum(w)-1)<=model.row_atol ||
            throw(ArgumentError("Invalid normalization."))
        M[1:n,d].=1;M[d,1:n]=w
    end
    report["factorizations"]=1
    X=lu!(M;check=true)\B
    all(isfinite,X) || return unresolved("Nonfinite fresh policy solution.")
    x=Q.(X);residual=Q.(B[1:n,:])-x[1:n,:]
    if criterion==:average
        for t in 1:2,i in 1:n;residual[i,t]-=x[d,t];end
    end
    for j in 1:n,i in 1:n
        z=p.P[i,j];z==0 && continue
        q=b*Q(z)
        for t in 1:2;residual[i,t]+=q*x[j,t];end
    end
    errors=Q[]
    for t in 1:2
        if criterion==:average
            push!(errors,(maximum(@view residual[:,t])-minimum(@view residual[:,t]))/alpha)
        else
            push!(errors,maximum(abs,@view residual[:,t])/(1-b*maximum(masses)))
        end
    end
    report[criterion==:average ? "bias_error_span_upper_exact" : "value_error_inf_upper_exact"]=string.(errors)
    report["residual_inf_exact"]=[string(maximum(abs,@view residual[:,t])) for t in 1:2]
    details=Dict{String,Any}[];intervals=Tuple{Q,Q}[];centers=Q[];positions=Int[]
    for (position,(i,a)) in enumerate(pairs)
        f=Q(model.h[i,a])-Q(model.h[i,a+1]);g=Q(model.c[i,a+1])-Q(model.c[i,a])
        dn=zero(Q);ds=zero(Q)
        for j in 1:n
            hi=model.P[i,j,a+1];lo=model.P[i,j,a];hi==lo && continue
            delta=b*(Q(hi)-Q(lo));dn+=abs(delta);ds+=delta
            f-=delta*x[j,1];g+=delta*x[j,2]
        end
        criterion==:average && ds!=0 && return unresolved("Adjacent transition difference does not sum exactly to zero.")
        multiplier=criterion==:average ? dn/2 : dn
        flo=f-multiplier*errors[1];fhi=f+multiplier*errors[1]
        glo=g-multiplier*errors[2];ghi=g+multiplier*errors[2]
        row=Dict{String,Any}("state"=>i,"gear"=>a,"f_center"=>Float64(f),"g_center"=>Float64(g),
            "f_lower_exact"=>string(flo),"f_upper_exact"=>string(fhi),
            "g_lower_exact"=>string(glo),"g_upper_exact"=>string(ghi))
        push!(details,row);report["frontier_bounds"]=details
        if ghi<=0
            row["eligibility"]="excluded_nonpositive_work";continue
        elseif glo<=0
            return unresolved("Resource-marginal interval contains zero.")
        end
        lo,hi=_residual_ratio_interval(flo,fhi,glo,ghi)
        row["ratio_lower_exact"]=string(lo);row["ratio_upper_exact"]=string(hi)
        row["ratio_center"]=Float64(f/g);row["eligibility"]="positive_work"
        push!(intervals,(lo,hi));push!(centers,f/g);push!(positions,position)
    end
    report["positive_candidates"]=length(positions)
    report["excluded_nonpositive_candidates"]=length(pairs)-length(positions)
    if isempty(positions)
        report["status"]="no_positive_candidate"
        return (position=0,value=NaN,status=:no_positive_candidate,report=report)
    end
    best=_residual_unique_minimum(intervals)
    best==0 && return unresolved("Ratio enclosures overlap; exact ties are not inferred.")
    value=Float64(centers[best]);isfinite(value) || return unresolved("Nonfinite ratio.")
    selected=positions[best];report["selected_pair"]=[pairs[selected]...]
    report["status"]="residual_certified"
    if length(intervals)>1
        gap=minimum(intervals[j][1]-intervals[best][2] for j in eachindex(intervals) if j!=best)
        report["separation_lower_exact"]=string(gap);report["separation_lower_approx"]=Float64(gap)
    end
    (position=selected,value=value,status=:residual_certified,report=report)
end
