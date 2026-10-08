# Rare selection safeguard for the EXACT stored-Float64 linear system.
# No input renormalization, rational LU, global precision/rounding change, or RNG.
# Approximate factorization is Float64. Only residuals/bounds use Rational{BigInt}.
const _ResidualQ = Rational{BigInt}
_residual_q(x::Real) = _ResidualQ(x)

# Given exact rational A and an arbitrary Float64 approximation X, enclose A^-1 B.
# Discounted case uses ||(I-beta*P)^-1||inf <= 1/(1-beta*||P||inf).
# General square case: E=I-Z*A; if ||E||inf<1, ||A^-1||inf <= ||Z||inf/(1-||E||inf).
function _residual_inverse_bound(A::Matrix{_ResidualQ}, Z::Matrix{Float64})
    d=size(A,1)
    size(A)==(d,d) && size(Z)==(d,d) || throw(DimensionMismatch("Inverse-bound sizes."))
    all(isfinite,Z) || return nothing
    z=_residual_q.(Z); q=zero(_ResidualQ); zn=zero(_ResidualQ)
    for i in 1:d
        rowsum=zero(_ResidualQ); zsum=zero(_ResidualQ)
        for j in 1:d
            e=_residual_q(i==j)
            for k in 1:d; e-=z[i,k]*A[k,j]; end
            rowsum+=abs(e); zsum+=abs(z[i,j])
        end
        q=max(q,rowsum); zn=max(zn,zsum)
    end
    q<1 || return nothing
    return (bound=zn/(1-q),contraction=q,kind="exact left-inverse residual norm")
end

function _residual_value_bounds(A::Matrix{_ResidualQ},B::Matrix{_ResidualQ},X,
                                inverse_bound::_ResidualQ)
    d=size(A,1)
    size(A)==(d,d) && size(B,1)==d && size(X)==size(B) ||
        throw(DimensionMismatch("Residual-bound sizes."))
    inverse_bound>0 || throw(ArgumentError("Positive inverse-norm bound required."))
    all(isfinite,X) || throw(ArgumentError("Nonfinite approximate solution."))
    x=_residual_q.(X); residual=fill(zero(_ResidualQ),size(B,2))
    for t in axes(B,2), i in 1:d
        r=B[i,t]
        for j in 1:d; r-=A[i,j]*x[j,t]; end
        residual[t]=max(residual[t],abs(r))
    end
    return (x=x,residual=residual,error=inverse_bound.*residual)
end

# Four corners also handle negative cost marginals; denominators must be positive.
function _residual_ratio_interval(flo,fhi,glo,ghi)
    flo<=fhi && zero(glo)<glo<=ghi || throw(ArgumentError("Invalid marginal interval."))
    corners=(flo/glo,flo/ghi,fhi/glo,fhi/ghi)
    return minimum(corners),maximum(corners)
end
function _residual_unique_minimum(intervals)
    isempty(intervals) && return 0
    best=argmin([x[2] for x in intervals])
    all(j->j==best || intervals[best][2]<intervals[j][1],eachindex(intervals)) || return 0
    return best
end

"""
    residual_certified_selection(model, actions, frontier; criterion, beta, omega)

Certify a UNIQUE minimum among positive-work pairs in the CURRENT frontier using
exact rational residual/error bounds for the stored floating-input system. This
is not a certificate of prior steps, indexability, or a fully exact index value.
Returns :residual_certified only when one complete ratio interval is strictly
below every other interval. Exact/overlapping ties remain unresolved.
Pairs whose work enclosure is wholly nonpositive are excluded. A work enclosure
that overlaps zero without proving nonpositivity remains unresolved. Returned
positions refer to the original supplied frontier, including excluded pairs.
The single fresh Float64 LU is exceptional work. With bound_method=:inverse
(the historical default), average mode also solves for a full approximate inverse.
With bound_method=:contraction, exact residuals use no dense rational matrix or
inverse: discounted row-sum contraction or average common-mass span contraction
must be certified. Average mode then requires exactly stochastic stored policy
rows and exactly zero-sum adjacent transition differences. Unsupported bounds
leave the comparison unresolved. No factorization is done in rationals.
"""
function residual_certified_selection(model::FiniteMultiGearModel{Float64},actions,
        frontier;criterion::Symbol=:discounted,beta=nothing,omega=nothing,bound_method::Symbol=:inverse)
    bound_method in (:inverse,:contraction) || throw(ArgumentError("Unknown residual bound method."))
    bound_method==:contraction && return _contraction_certified_selection(model,actions,frontier;
        criterion=criterion,beta=beta,omega=omega)
    n=nstates(model); p=policy_data(model,actions)
    criterion in (:discounted,:average) || throw(ArgumentError("Unknown criterion."))
    pairs=Tuple{Int,Int}[(Int(t[1]),Int(t[2])) for t in frontier]
    length(unique(pairs))==length(pairs) || throw(ArgumentError("Repeated frontier pair."))
    for (i,a) in pairs
        1<=i<=n && model.controllable[i] && a==actions[i] && 1<=a<=maxgear(model) ||
            throw(ArgumentError("Frontier is not an admissible current adjacent comparison."))
    end
    d=n+(criterion==:average)
    factor=criterion==:discounted ? Float64(beta) : 1.0
    criterion==:discounted && !(isfinite(factor) && 0<factor<1) &&
        throw(ArgumentError("Invalid discount."))
    w=if criterion==:average
        omega===nothing ? vcat(1.0,zeros(n-1)) : Float64.(omega)
    else
        Float64[]
    end
    criterion==:average && !(length(w)==n && all(isfinite,w) &&
        abs(sum(w)-1)<=model.row_atol) && throw(ArgumentError("Invalid normalization."))
    qfactor=_residual_q(factor)
    A=fill(zero(_ResidualQ),d,d);B=fill(zero(_ResidualQ),d,2)
    for i in 1:n
        B[i,1]=_residual_q(p.h[i]);B[i,2]=_residual_q(p.c[i])
        for j in 1:n
            A[i,j]=_residual_q(i==j)-qfactor*_residual_q(p.P[i,j])
        end
        if criterion==:average; A[i,d]=one(_ResidualQ);A[d,i]=_residual_q(w[i]);end
    end
    report=Dict{String,Any}("scope"=>"current-frontier selection for exact stored-Float64 linear system; not DAI certification",
        "criterion"=>string(criterion),"dimension"=>d,"frontier_pairs"=>[[i,a] for (i,a) in pairs],
        "arithmetic"=>"Float64 LU and exact Rational{BigInt} residual/bound arithmetic; no rational LU",
        "factorizations"=>0,"inverse_rhs_columns"=>0,"status"=>"unresolved_comparison")
    # All conversion/factorization errors leave the accepted algorithm state unchanged.
    report["factorizations"]=1
    fac=lu(Float64.(A);check=true); X=fac\Float64.(B)
    all(isfinite,X) || return (position=0,value=NaN,status=:unresolved_comparison,report=report)
    inverse_info=nothing
    if criterion==:discounted
        contraction=maximum(qfactor*sum(_residual_q(p.P[i,j]) for j in 1:n) for i in 1:n)
        contraction<1 && (inverse_info=(bound=1/(1-contraction),contraction=contraction,
            kind="exact discounted row-sum contraction"))
    end
    if inverse_info===nothing
        report["inverse_rhs_columns"]=d
        Z=fac\Matrix{Float64}(I,d,d)
        inverse_info=_residual_inverse_bound(A,Z)
    end
    if inverse_info===nothing
        report["reason"]="Could not certify an inverse-norm bound."
        return (position=0,value=NaN,status=:unresolved_comparison,report=report)
    end
    bound=_residual_value_bounds(A,B,X,inverse_info.bound)
    report["inverse_bound_kind"]=inverse_info.kind
    report["inverse_norm_upper_exact"]=string(inverse_info.bound)
    report["contraction_upper_exact"]=string(inverse_info.contraction)
    report["residual_inf_exact"]=string.(bound.residual)
    report["value_error_inf_upper_exact"]=string.(bound.error)
    details=Dict{String,Any}[]; intervals=Tuple{_ResidualQ,_ResidualQ}[];centers=_ResidualQ[]
    positions=Int[]
    for (position,(i,a)) in enumerate(pairs)
        f0=_residual_q(model.h[i,a])-_residual_q(model.h[i,a+1])
        g0=_residual_q(model.c[i,a+1])-_residual_q(model.c[i,a]);dn=zero(_ResidualQ)
        for j in 1:n
            delta=qfactor*(_residual_q(model.P[i,j,a+1])-_residual_q(model.P[i,j,a]))
            f0-=delta*bound.x[j,1];g0+=delta*bound.x[j,2];dn+=abs(delta)
        end
        flo=f0-dn*bound.error[1];fhi=f0+dn*bound.error[1]
        glo=g0-dn*bound.error[2];ghi=g0+dn*bound.error[2]
        row=Dict{String,Any}("state"=>i,"gear"=>a,"f_center"=>Float64(f0),"g_center"=>Float64(g0),
            "f_lower_exact"=>string(flo),"f_upper_exact"=>string(fhi),
            "g_lower_exact"=>string(glo),"g_upper_exact"=>string(ghi))
        push!(details,row)
        if ghi<=0
            row["eligibility"]="excluded_nonpositive_work"
            continue
        elseif glo<=0
            report["reason"]="Resource-marginal interval contains zero.";report["frontier_bounds"]=details
            return (position=0,value=NaN,status=:unresolved_comparison,report=report)
        end
        lo,hi=_residual_ratio_interval(flo,fhi,glo,ghi)
        row["ratio_lower_exact"]=string(lo);row["ratio_upper_exact"]=string(hi)
        row["ratio_center"]=Float64(f0/g0)
        row["eligibility"]="positive_work"
        push!(intervals,(lo,hi));push!(centers,f0/g0);push!(positions,position)
    end
    report["frontier_bounds"]=details
    report["positive_candidates"]=length(positions)
    report["excluded_nonpositive_candidates"]=length(pairs)-length(positions)
    if isempty(positions)
        report["status"]="no_positive_candidate"
        return (position=0,value=NaN,status=:no_positive_candidate,report=report)
    end
    best=_residual_unique_minimum(intervals)
    if best==0
        report["reason"]="Ratio enclosures overlap; exact ties are not inferred."
        return (position=0,value=NaN,status=:unresolved_comparison,report=report)
    end
    value=Float64(centers[best])
    isfinite(value) || return (position=0,value=NaN,status=:unresolved_comparison,report=report)
    selected=positions[best]
    report["status"]="residual_certified";report["selected_pair"]=[pairs[selected][1],pairs[selected][2]]
    if length(intervals)>1
        gap=minimum(intervals[j][1]-intervals[best][2] for j in eachindex(intervals) if j!=best)
        report["separation_lower_exact"]=string(gap);report["separation_lower_approx"]=Float64(gap)
    end
    return (position=selected,value=value,status=:residual_certified,report=report)
end
