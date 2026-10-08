# Preparation for the shared checked reduced-FP pivot kernel.
# Each trace/workspace is owned by one task. No global thread or RNG changes.

const PreparationStage = NamedTuple{(:stage,:seconds,:allocated_bytes,:gc_seconds,
    :compile_seconds,:recompile_seconds),Tuple{Symbol,Float64,Int,Float64,Float64,Float64}}
const PreparationResidual = NamedTuple{(:stage,:columns,:maximum_backward_error),
    Tuple{Symbol,Int,Float64}}

"""Optional preparation instrumentation. Use in separate diagnostic executions,
not in the principal initialization/loop/end-to-end timings. Stage times are
nonoverlapping but exclude wrapper overhead; they need not sum to wall time.
"""
mutable struct PreparationTrace
    stages::Vector{PreparationStage}
    residuals::Vector{PreparationResidual}
end
PreparationTrace() = PreparationTrace(PreparationStage[],PreparationResidual[])

@inline function _preparation_stage(f::F,::Nothing,name::Symbol) where {F}
    return f()
end
function _preparation_stage(f::F,trace::PreparationTrace,name::Symbol) where {F}
    measured=@timed f()
    push!(trace.stages,(stage=name,seconds=measured.time,allocated_bytes=Int(measured.bytes),
        gc_seconds=measured.gctime,compile_seconds=measured.compile_time,
        recompile_seconds=measured.recompile_time))
    return measured.value
end
@inline _preparation_record!(::Nothing,name,columns,error)=nothing
function _preparation_record!(trace::PreparationTrace,name,columns,error)
    push!(trace.residuals,(stage=name,columns=columns,maximum_backward_error=Float64(error)))
    return nothing
end

function _preparation_tolerance(tol)
    isfinite(tol) && tol>=0 || throw(ArgumentError("Invalid preparation backward tolerance."))
    return nothing
end

# Common assurance rule for EVERY freshly solved column, including all solved
# inverse columns, in C/R/M/optimized-FP. normwise backward error is NOT a
# forward-error bound or an indexability certificate. RHS may be explicit, or
# identity columns specified by row labels. No full identity is needed for FP.
function _preparation_column_check(M,X,rhs,tol;identity_rows=nothing)
    _preparation_tolerance(tol)
    size(M,1)==size(M,2)==size(X,1) || throw(DimensionMismatch("Preparation solve dimensions."))
    if identity_rows===nothing
        size(rhs)==size(X) || throw(DimensionMismatch("Preparation RHS dimensions."))
    else
        length(identity_rows)==size(X,2) || throw(DimensionMismatch("Identity column labels."))
        all(i->1<=i<=size(M,1),identity_rows) || throw(ArgumentError("Invalid identity row."))
    end
    all(isfinite,M) && all(isfinite,X) || error("Nonfinite preparation coefficient/solution.")
    R=similar(X)
    mul!(R,M,X)
    mn=opnorm(M,Inf); isfinite(mn) || error("Preparation norm overflow.")
    worst=0.0
    for col in axes(X,2)
        xn=0.0; bn=identity_rows===nothing ? 0.0 : 1.0; rn=0.0
        for i in axes(X,1)
            b=identity_rows===nothing ? rhs[i,col] : (i==identity_rows[col] ? 1.0 : 0.0)
            isfinite(b) || error("Nonfinite preparation right-hand side.")
            rn=max(rn,abs(R[i,col]-b)); xn=max(xn,abs(X[i,col]))
            identity_rows===nothing && (bn=max(bn,abs(b)))
        end
        den=mn*xn+bn
        isfinite(den) && isfinite(rn) || error("Preparation residual scale overflow.")
        err=den==0 ? (rn==0 ? 0.0 : Inf) : rn/den
        worst=max(worst,err)
    end
    isfinite(worst) && worst<=tol || error("Preparation backward-residual screen failed: $worst > $tol.")
    return worst
end

# Assemble directly from validated primitives. Average admissibility is already
# guaranteed by the shared positive-row PerformanceInput; no graph-copy checks
# are hidden here. This is the N(+1) COEFFICIENT matrix, not a tableau/inverse.
function _preparation_system(model,actions,criterion,b,omega)
    n=nstates(model); d=n+(criterion==:average)
    M=zeros(Float64,d,d); B=zeros(Float64,d,2)
    factor=criterion==:discounted ? b : 1.0
    @inbounds for j in 1:n, i in 1:n
        M[i,j]=(i==j ? 1.0 : 0.0)-factor*model.P[i,j,actions[i]+1]
    end
    @inbounds for i in 1:n
        slot=actions[i]+1; B[i,1]=model.h[i,slot]; B[i,2]=model.c[i,slot]
    end
    if criterion==:average
        @inbounds for i in 1:n; M[i,d]=1.0; M[d,i]=omega[i]; end
    end
    return M,B
end

# Only solve and check the required inverse/value columns. In particular, the
# all-zero reference policy does NOT construct any tableau or aggregate metrics.
function _preparation_selected_solve(model,actions,criterion,b,omega,labels,tol,trace,prefix)
    M,B=_preparation_stage(trace,Symbol(prefix,"_assembly")) do
        _preparation_system(model,actions,criterion,b,omega)
    end
    d=size(M,1); m=length(labels)
    rhs=_preparation_stage(trace,Symbol(prefix,"_rhs")) do
        R=zeros(Float64,d,m+2)
        for j in 1:m; R[labels[j],j]=1.0; end
        copyto!(@view(R[:,m+1:m+2]),B)
        R
    end
    fac=_preparation_stage(trace,Symbol(prefix,"_factorization")) do
        lu(M;check=true)
    end
    solution=_preparation_stage(trace,Symbol(prefix,"_solve")) do
        S=copy(rhs); ldiv!(fac,S); S
    end
    error=_preparation_stage(trace,Symbol(prefix,"_residual")) do
        _preparation_column_check(M,solution,rhs,tol)
    end
    _preparation_record!(trace,Symbol(prefix,"_all_solved_columns"),m+2,error)
    return solution
end


# Error-free subtraction for finite transition probabilities. Comparing both
# components avoids treating equal rounded differences as exact identities.
@inline function _preparation_difference(x::Float64,y::Float64)
    hi=x-y;bvirt=x-hi;avirt=hi+bvirt
    lo=(x-avirt)+(bvirt-y)
    hi,lo
end
function _preparation_reference_columns(model,states)
    C=length(states);A=maxgear(model);rows=C*(A-1)
    rows==0 && return Int[],Int[]
    same=trues(C,A-1)
    for a in 3:A,j in 1:nstates(model)
        @inbounds for s in 1:C
            if same[s,a-1]
                i=states[s]
                same[s,a-1]=_preparation_difference(model.P[i,j,a+1],model.P[i,j,a])==
                            _preparation_difference(model.P[i,j,3],model.P[i,j,2])
            end
        end
    end
    columns=Vector{Int}(undef,rows);representatives=Int[]
    for a in 2:A,s in 1:C
        r=s+(a-2)*C
        if a>2 && same[s,a-1]
            columns[r]=columns[s]
        else
            push!(representatives,r);columns[r]=length(representatives)
        end
    end
    columns,representatives
end

# Fixed reference differences are packed by original state, NOT physical pivot
# slot. A bounded d-by-block scratch matrix replaces the scalar triple loop.
# The appended average-coordinate difference is zero; the rate remains in W/V.
function _preparation_reference_transform(model,criterion,b,states,solution,block_columns)
    n=nstates(model); d=size(solution,1); C=length(states); A=maxgear(model)
    rows=C*(A-1)
    reference_columns,representatives=_preparation_reference_columns(model,states)
    unique_rows=length(representatives)
    ell=zeros(Float64,C,unique_rows); fh=zeros(Float64,rows); gh=zeros(Float64,rows)
    rows==0 && return ell,fh,gh,reference_columns
    width=min(block_columns,unique_rows)
    D=zeros(Float64,d,width); transformed=zeros(Float64,C+2,width)
    continuation_f=zeros(unique_rows);continuation_g=zeros(unique_rows)
    factor=criterion==:discounted ? b : 1.0
    for first in 1:width:unique_rows
        bsize=min(width,unique_rows-first+1)
        for col in 1:bsize
            r=representatives[first+col-1]; ci=mod1(r,C); a=2+div(r-1,C); i=states[ci]
            @inbounds for j in 1:n
                D[j,col]=factor*(model.P[i,j,a+1]-model.P[i,j,a])
            end
            d>n && (D[d,col]=0.0)
        end
        mul!(@view(transformed[:,1:bsize]),transpose(solution),@view(D[:,1:bsize]))
        for col in 1:bsize
            r=representatives[first+col-1]; ci=mod1(r,C); a=2+div(r-1,C); i=states[ci]
            @inbounds for j in 1:C; ell[j,first+col-1]=transformed[j,col]; end
            continuation_f[first+col-1]=transformed[C+1,col]
            continuation_g[first+col-1]=transformed[C+2,col]
            fh[r]=model.h[i,a]-model.h[i,a+1]-transformed[C+1,col]
            gh[r]=model.c[i,a+1]-model.c[i,a]+transformed[C+2,col]
        end
    end
    # Shared transition rows still have independent holding/resource terms.
    # Reuse only the transformed continuation contribution.
    for r in 1:rows
        original=representatives[reference_columns[r]]
        r==original && continue
        ci=mod1(r,C);a=2+div(r-1,C);i=states[ci]
        column=reference_columns[r]
        fh[r]=(model.h[i,a]-model.h[i,a+1])-continuation_f[column]
        gh[r]=(model.c[i,a+1]-model.c[i,a])+continuation_g[column]
    end
    all(isfinite,ell) && all(isfinite,fh) && all(isfinite,gh) ||
        error("Nonfinite transformed reference comparisons.")
    return ell,fh,gh,reference_columns
end

# Obtain the reduced current tableau by a right solve, rather than solving
# inverse columns and multiplying B0*W. Every transposed-solve and value column
# still receives the same normwise backward-residual screen. The two current
# value columns retain the original aggregate/scaling formulas for general c.
function _preparation_current_solve(model,actions,criterion,b,omega,labels,tol,trace)
    M,B=_preparation_stage(trace,:current_assembly) do
        _preparation_system(model,actions,criterion,b,omega)
    end
    n=nstates(model); d=size(M,1); m=length(labels)
    rhs=_preparation_stage(trace,:current_rhs) do
        R=zeros(Float64,d,m)
        factor=criterion==:discounted ? b : 1.0
        @inbounds for ss in 1:32:m,jj in 1:32:n
            for s in ss:min(ss+31,m),j in jj:min(jj+31,n)
                i=labels[s];R[j,s]=(i==j ? 1.0 : 0.0)-factor*model.P[i,j,1]
            end
        end
        if criterion==:average
            @inbounds for s in 1:m;R[d,s]=1.0;end
        end
        R
    end
    fac=_preparation_stage(trace,:current_factorization) do
        lu(M;check=true)
    end
    rows,values=_preparation_stage(trace,:current_solve) do
        Y=copy(rhs);ldiv!(transpose(fac),Y)
        V=copy(B);ldiv!(fac,V)
        Y,V
    end
    row_error,value_error=_preparation_stage(trace,:current_residual) do
        (_preparation_column_check(transpose(M),rows,rhs,tol),
         _preparation_column_check(M,values,B,tol))
    end
    _preparation_record!(trace,:current_transposed_columns,m,row_error)
    _preparation_record!(trace,:current_value_columns,2,value_error)
    return _preparation_stage(trace,:current_tableau) do
        X=Matrix{Float64}(undef,m,m)
        # Block the transpose/gather so both source and destination tiles stay
        # in cache, including noncontiguous controllable labels and suffixes.
        @inbounds for cc in 1:32:m, ss in 1:32:m
            for c in cc:min(cc+31,m),s in ss:min(ss+31,m)
                X[s,c]=rows[labels[c],s]
            end
        end
        phi=zeros(m);psi=zeros(m);scale=zeros(m)
        factor=criterion==:discounted ? b : 1.0
        @inbounds for s in 1:m
            i=labels[s];a=actions[i]
            phi[s]=model.h[i,1]-model.h[i,a+1]
            psi[s]=model.c[i,a+1]-model.c[i,1];scale[s]=abs(psi[s])
        end
        # Same per-state addition order, but stream transition columns and
        # reuse each value coefficient across all active labels.
        P=model.P
        @inbounds for j in 1:n
            v=values[j,1];u=values[j,2]
            for s in 1:m
                i=labels[s];a=actions[i]
                delta=factor*(P[i,j,a+1]-P[i,j,1])
                phi[s]-=delta*v;psi[s]+=delta*u;scale[s]+=abs(delta*u)
            end
        end
        all(isfinite,X) && all(isfinite,phi) && all(isfinite,psi) && all(isfinite,scale) ||
            error("Nonfinite optimized tableau/aggregate metrics.")
        X,phi,psi,scale
    end
end

"""
    initialize_reduced_fp_optimized(input::PerformanceInput; ...)

Optimized Float64 preparation for the SAME ReducedFPWorkspace and pivot kernel.
The general original initialize_reduced_fp remains available and unchanged.
Average use requires dense-positive rows or an explicitly checked support topology; other sparse/transient/periodic
models remain supported by the original general initializer, not this pilot.
All solved columns pass the common backward-residual screen. block_columns
bounds temporary transformation storage; it does not change required ell storage.
trace is for separate instrumented runs. actions permits tested suffix starts;
keep_recovery is optional and remains false in the index-only comparison.
"""
function initialize_reduced_fp_optimized(input::PerformanceInput;
        criterion::Symbol=:discounted,beta=nothing,omega=nothing,
        family::AbstractPolicyFamily=UnrestrictedFamily(),keep_recovery::Bool=false,
        actions=nothing,backward_tol::Real=1e-10,block_columns::Integer=64,
        trace::Union{Nothing,PreparationTrace}=nothing)
    _preparation_tolerance(backward_tol)
    1<=block_columns<=typemax(Int) || throw(ArgumentError("block_columns must be positive."))
    model=input.model; b,w=_rankone_parameters(model,criterion,beta,omega)
    criterion==:average && !_performance_average_admitted(input) && throw(ArgumentError(
        "Optimized average preparation requires dense-positive rows or explicitly checked support topology; use the general initializer otherwise."))
    _check_family(model,family)
    n=nstates(model); A=maxgear(model); states=findall(model.controllable); C=length(states); K=C*A
    act=actions===nothing ? [model.controllable[i] ? A : 0 for i in 1:n] : Int.(actions)
    length(act)==n && all(i->act[i] in admissible_gears(model,i),1:n) ||
        throw(ArgumentError("Invalid optimized-preparation policy."))
    policy_in_family(family,act) || throw(ArgumentError("Policy outside family."))
    labels=[i for i in states if act[i]>0]; m=length(labels)
    x,ph,ps,sc=_preparation_current_solve(model,act,criterion,b,w,labels,backward_tol,trace)
    counts=FPWorkCounts(); counts.policy_factorizations=1
    ell=zeros(0,0); reference_columns=Int[]; fh=Float64[]; gh=Float64[]; RW=nothing; RV=nothing
    if A>=2 || keep_recovery
        low=zeros(Int,n)
        refsolution=_preparation_selected_solve(model,low,criterion,b,w,states,backward_tol,trace,:reference)
        ell,fh,gh,reference_columns=_preparation_stage(trace,:reference_transform) do
            _preparation_reference_transform(model,criterion,b,states,refsolution,Int(block_columns))
        end
        counts.reference_factorizations=1
        if keep_recovery
            RW=Matrix(@view(refsolution[:,1:C])); RV=Matrix(@view(refsolution[:,C+1:C+2]))
        end
    end
    ws=_preparation_stage(trace,:workspace_buffers) do
        # The usual all-high start already has full-capacity owned buffers.
        # Reuse them instead of allocating/copying a second C-by-C tableau.
        X=m==C ? x : zeros(Float64,C,C)
        phi=m==C ? ph : zeros(C);psi=m==C ? ps : zeros(C);pscale=m==C ? sc : zeros(C)
        if m!=C
            copyto!(@view(X[1:m,1:m]),x)
            copyto!(@view(phi[1:m]),ph);copyto!(@view(psi[1:m]),ps);copyto!(@view(pscale[1:m]),sc)
        end
        refs=zeros(Int,n); slots=zeros(Int,n); state_at=zeros(Int,C); pred=zeros(Int,n)
        for (j,i) in enumerate(states); refs[i]=j; end
        for (j,i) in enumerate(labels); state_at[j]=i; slots[i]=j; end
        if family isa OrderedThresholdFamily
            for k in 2:length(family.states); pred[family.states[k]]=family.states[k-1]; end
        end
        mpi=Matrix{Union{Missing,Float64}}(undef,C,A); fill!(mpi,missing)
        lex=Vector{Union{Nothing,ExactRational}}(undef,K); fill!(lex,nothing)
        ReducedFPWorkspace(model,criterion,b,w,family,states,refs,state_at,slots,act,pred,m,
            X,zeros(C,C),phi,psi,zeros(C),zeros(C),zeros(C),zeros(C),zeros(C),zeros(C),
            pscale,zeros(C),zeros(C),zeros(C),ell,reference_columns,false,fh,gh,RW,RV,zeros(C),zeros(C),zeros(C),
            zeros(Int,C),0,fill(false,n),zeros(Int,n),zeros(Int,n),zeros(Int,n),zeros(Int,n),fill(false,n),
            mpi,zeros(Int,K),zeros(Int,K),zeros(K),zeros(K),zeros(K),zeros(K),zeros(Int,K),
            fill(:unassigned,K),lex,0,counts)
    end
    _preparation_stage(trace,:current_comparisons) do
        _fp_recompute_current!(ws)
    end
    all(isfinite,ws.f) && all(isfinite,ws.g) && all(isfinite,ws.g_scale) ||
        error("Nonfinite optimized initial comparisons.")
    return ws
end

"""Uninstrumented old/new workspace comparison plus independent fresh-solve
FP audits. Used outside timing. This is not an indexability certificate."""
function audit_fp_initialization(input::PerformanceInput;criterion=:discounted,beta=nothing,
        omega=nothing,family=UnrestrictedFamily(),actions=nothing,keep_recovery=false,
        block_columns=64,tol=1e-7)
    isfinite(tol) && tol>0 || throw(ArgumentError("Invalid initialization audit tolerance."))
    old=initialize_reduced_fp(input.model;criterion=criterion,beta=beta,omega=omega,family=family,
        actions=actions,keep_recovery=keep_recovery)
    new=initialize_reduced_fp_optimized(input;criterion=criterion,beta=beta,omega=omega,family=family,
        actions=actions,keep_recovery=keep_recovery,block_columns=block_columns)
    errors=Dict{String,Float64}()
    errors["ell"]=_performance_scaled_error(old.ell[:,old.reference_columns],new.ell[:,new.reference_columns])
    for field in (:X,:phi,:psi,:psi_scale,:f,:g,:g_scale,:fhat,:ghat)
        errors[string(field)]=_performance_scaled_error(getfield(old,field),getfield(new,field))
    end
    if keep_recovery
        errors["recovery_W"]=_performance_scaled_error(old.recovery_W,new.recovery_W)
        errors["recovery_values"]=_performance_scaled_error(old.recovery_values,new.recovery_values)
    end
    mapping=old.states==new.states && old.reference_slot==new.reference_slot &&
        old.state_at==new.state_at && old.slot_of==new.slot_of && old.actions==new.actions && old.m==new.m
    checks=(audit_reduced_fp(old;atol=tol/10,rtol=tol),audit_reduced_fp(new;atol=tol/10,rtol=tol))
    passed=mapping && all(e->isfinite(e) && e<=tol,values(errors)) && all(a->a.passed,checks)
    return (passed=passed,field_errors=errors,mapping_agreement=mapping,
        old_audit_passed=checks[1].passed,new_audit_passed=checks[2].passed,
        old_policy_factorizations=old.counts.policy_factorizations,
        new_policy_factorizations=new.counts.policy_factorizations,
        old_reference_factorizations=old.counts.reference_factorizations,
        new_reference_factorizations=new.counts.reference_factorizations)
end

"""Numerical monotonicity SCREEN on the downshift assignment sequence.
Completion, path monotonicity and DAI verification are distinct. A resolved
negative gap is not by itself proof of nonindexability of the bandit."""
function performance_mpi_screen(snapshot;atol=1e-9,rtol=1e-7)
    isfinite(atol) && isfinite(rtol) && min(atol,rtol)>=0 || throw(ArgumentError("Invalid MP screen tolerances."))
    decreasing=Int[]; mingap=Inf; finite=all(isfinite,snapshot.ratio)
    for k in 2:length(snapshot.ratio)
        a,b=snapshot.ratio[k-1],snapshot.ratio[k]; gap=b-a
        mingap=min(mingap,gap)
        gap < -(atol+rtol*max(1.0,abs(a),abs(b))) && push!(decreasing,k)
    end
    status=!finite ? :nonfinite : !isempty(decreasing) ? :resolved_decrease :
        snapshot.complete ? :numerical_nondecreasing : :incomplete_no_resolved_decrease
    return (status=status,decreasing_positions=decreasing,
        minimum_gap=length(snapshot.ratio)<2 ? 0.0 : mingap,dai_verified=false)
end
