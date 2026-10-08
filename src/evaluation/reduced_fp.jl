# Reduced FP uses fixed-capacity, double-buffered storage. All mutating workspaces
# belong to ONE job. Models/reference data must not be mutated concurrently.

mutable struct FPWorkCounts
    policy_factorizations::Int
    reference_factorizations::Int
    nonfinal_pivots::Int
    final_pivots::Int
    trial_attempts::Int
    retained_refreshes::Int
    new_comparisons::Int
    exact_comparison_solves::Int
    fallback_direct_solves::Int
    audit_solves::Int
    rebuilds::Int
    rebuild_attempts::Int
    unichain_screens::Int
    reference_dot_elements::Int
    materialization_elements::Int
end
FPWorkCounts() = FPWorkCounts(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

struct FPPivotReport{T<:AbstractFloat}
    accepted::Bool
    reason::Symbol
    selected::Tuple{Int,Int}
    denominator::T
    denominator_scale::T
    old_dimension::Int
    new_dimension::Int
end

"""
Preallocated reduced FP workspace. `X[1:m,1:m]` is logically active; physical
capacity never shrinks. `state_at` and `slot_of` map slots to ORIGINAL states.
`ell[:,reference_columns[r]]` holds a fixed mathematical reference row as a
contiguous column. Optimized preparation shares exactly repeated transition
differences without conflating their holding or resource terms.
Only gears >=2 have reference rows. The binary index-only path performs NO
reference-policy factorization and retains NO full-state recovery arrays.

Public fields are inspection-only. Rejected pivots preserve accepted numeric
state/actions/maps; scratch buffers and diagnostic counters may change. Ordinary
Float64 kernels are designed for zero allocation AFTER compilation, but that
property must be measured on the executing Julia version.
"""
mutable struct ReducedFPWorkspace{T<:AbstractFloat,F<:AbstractPolicyFamily}
    model::FiniteMultiGearModel{T}
    criterion::Symbol
    beta::Union{Nothing,T}
    omega::Union{Nothing,Vector{T}}
    family::F
    states::Vector{Int}
    reference_slot::Vector{Int}
    state_at::Vector{Int}
    slot_of::Vector{Int}
    actions::Vector{Int}
    predecessor::Vector{Int}
    m::Int
    X::Matrix{T}
    trial_X::Matrix{T}
    phi::Vector{T}
    psi::Vector{T}
    trial_phi::Vector{T}
    trial_psi::Vector{T}
    f::Vector{T}
    g::Vector{T}
    trial_f::Vector{T}
    trial_g::Vector{T}
    psi_scale::Vector{T}
    trial_psi_scale::Vector{T}
    g_scale::Vector{T}
    trial_g_scale::Vector{T}
    ell::Matrix{T}
    reference_columns::Vector{Int}
    frontier_only::Bool
    fhat::Vector{T}
    ghat::Vector{T}
    recovery_W::Union{Nothing,Matrix{T}}
    recovery_values::Union{Nothing,Matrix{T}}
    v::Vector{T}
    z::Vector{T}
    row::Vector{T}
    frontier_slots::Vector{Int}
    nfrontier::Int
    graph_seen::Vector{Bool}
    graph_nodes::Vector{Int}
    graph_next::Vector{Int}
    graph_order::Vector{Int}
    graph_component::Vector{Int}
    graph_closed::Vector{Bool}
    mpi::Matrix{Union{Missing,T}}
    log_state::Vector{Int}
    log_gear::Vector{Int}
    log_f::Vector{T}
    log_g::Vector{T}
    log_m::Vector{T}
    log_eta::Vector{T}
    log_dimension::Vector{Int}
    log_evidence::Vector{Symbol}
    log_exact::Vector{Union{Nothing,ExactRational}}
    k::Int
    counts::FPWorkCounts
end

# Construct the fixed reference COEFFICIENT matrix, not its inverse.
function _fp_reference_system(model::FiniteMultiGearModel{T},criterion,b,w) where {T}
    n=nstates(model); d=n+(criterion == :average)
    B0=zeros(T,d,d)
    factor=criterion == :discounted ? b : one(T)
    for j in 1:n, i in 1:n
        B0[i,j]=(i==j ? one(T) : zero(T))-factor*model.P[i,j,1]
    end
    if criterion == :average
        for i in 1:n; B0[i,n+1]=one(T); B0[n+1,i]=w[i]; end
    end
    return B0
end

# One factorization, solves for active inverse COLUMNS and two value columns.
# Used ONLY by initialization, explicit recovery/audits, or exceptional rebuilds.
function _fp_fresh(model::FiniteMultiGearModel{T},actions,criterion,b,w,labels;
                   backward_tol::Real=1e-10) where {T}
    p,M,B,_ = _rankone_system(model,actions,criterion,b,w)
    n=nstates(model); d=size(M,1); m=length(labels)
    rhs=zeros(T,d,m+2)
    for t in 1:m; rhs[labels[t],t]=one(T); end
    rhs[:,m+1:m+2]=B
    sol=lu(M;check=true)\rhs
    all(isfinite,sol) || error("Nonfinite reduced-FP initialization solve.")
    # Forward error is NOT certified by this backward-residual screen.
    residual=M*sol-rhs; mn=opnorm(M,Inf)
    for j in axes(sol,2)
        den=mn*norm(@view(sol[:,j]),Inf)+norm(@view(rhs[:,j]),Inf)
        r=norm(@view(residual[:,j]),Inf)
        isfinite(den) && isfinite(r) || error("Nonfinite reduced-FP solve diagnostics.")
        (den==0 ? r==0 : r/den<=backward_tol) || error("Reduced-FP solve residual exceeds its tolerance.")
    end
    values=sol[:,m+1:m+2]
    X=(_fp_reference_system(model,criterion,b,w)*sol[:,1:m])[labels,:]
    phi=zeros(T,m); psi=zeros(T,m); scales=zeros(T,m)
    factor=criterion == :discounted ? b : one(T)
    for t in 1:m
        i=labels[t]; a=actions[i]
        phi[t]=model.h[i,1]-model.h[i,a+1]
        psi[t]=model.c[i,a+1]-model.c[i,1]
        scales[t]=abs(psi[t])
        for j in 1:n
            dp=factor*(model.P[i,j,a+1]-model.P[i,j,1])
            phi[t]-=dp*values[j,1]; psi[t]+=dp*values[j,2]
            scales[t]+=abs(dp*values[j,2])
        end
    end
    all(isfinite,X) && all(isfinite,phi) && all(isfinite,psi) && all(isfinite,scales) ||
        error("Nonfinite reduced tableau or aggregate metrics.")
    return X,phi,psi,scales,values,sol[:,1:m]
end

@inline _fp_rowid(w,i,a) = w.reference_slot[i]+(a-2)*length(w.states)

function _fp_reference_data(model::FiniteMultiGearModel{T},criterion,b,w,states;
                            backward_tol::Real=1e-10) where {T}
    low=zeros(Int,nstates(model))
    _,_,_,_,values,W=_fp_fresh(model,low,criterion,b,w,states;backward_tol=backward_tol)
    C=length(states); A=maxgear(model); n=nstates(model)
    ell=zeros(T,C,C*(A-1)); fh=zeros(T,C*(A-1)); gh=similar(fh)
    factor=criterion == :discounted ? b : one(T)
    for a in 2:A, (ci,i) in enumerate(states)
        r=ci+(a-2)*C
        fh[r]=model.h[i,a]-model.h[i,a+1]
        gh[r]=model.c[i,a+1]-model.c[i,a]
        for j in 1:n
            dp=factor*(model.P[i,j,a+1]-model.P[i,j,a])
            fh[r]-=dp*values[j,1]; gh[r]+=dp*values[j,2]
            for c in 1:C; ell[c,r]+=dp*W[j,c]; end
        end
    end
    return ell,fh,gh,W,values
end

"""
    initialize_reduced_fp(model; criterion=:discounted, beta=nothing, omega=nothing,
        family=UnrestrictedFamily(), keep_recovery=false, actions=nothing,
        backward_tol=1e-10)

Prepare a reduced tableau and reusable buffers. The normal start is the highest
gear at every controllable state. `actions` is for kernel/rebuild tests and does
not turn a suffix computation into a full DAI computation. Initializing the
average formulation requires a unichain current AND reference policy; transient
states and periodic classes are permitted. This does not verify all policies.
"""
function initialize_reduced_fp(model::FiniteMultiGearModel{T};criterion::Symbol=:discounted,
        beta=nothing,omega=nothing,family::AbstractPolicyFamily=UnrestrictedFamily(),
        keep_recovery::Bool=false,actions=nothing,backward_tol::Real=1e-10) where {T}
    b,w=_rankone_parameters(model,criterion,beta,omega)
    isfinite(backward_tol) && backward_tol>=0 || throw(ArgumentError("Invalid backward tolerance."))
    _check_family(model,family)
    n=nstates(model); A=maxgear(model); states=findall(model.controllable); C=length(states); K=C*A
    act=actions===nothing ? [model.controllable[i] ? A : 0 for i in 1:n] : Int.(actions)
    policy_data(model,act); policy_in_family(family,act) || throw(ArgumentError("Policy outside family."))
    if criterion == :average
        # No reference solve is needed in the binary index-only branch, but its
        # required unichain assumption is still checked from the graph.
        length(_closed_classes(policy_data(model,zeros(Int,n)).P))==1 ||
            throw(ArgumentError("Average reduced FP requires a unichain reference policy."))
    end
    labels=[i for i in states if act[i]>0]; m=length(labels)
    x,ph,ps,sc,_,_=_fp_fresh(model,act,criterion,b,w,labels;backward_tol=backward_tol)
    counts=FPWorkCounts(); counts.policy_factorizations=1
    ell=zeros(T,0,0); fh=T[]; gh=T[]; RW=nothing; RV=nothing
    if A>=2 || keep_recovery
        ell,fh,gh,rw,rv=_fp_reference_data(model,criterion,b,w,states;backward_tol=backward_tol)
        counts.reference_factorizations=1
        if keep_recovery; RW=rw; RV=rv; end
    end
    X=zeros(T,C,C); X[1:m,1:m]=x
    phi=zeros(T,C); psi=zeros(T,C); pscale=zeros(T,C)
    phi[1:m]=ph; psi[1:m]=ps; pscale[1:m]=sc
    refs=zeros(Int,n); slots=zeros(Int,n); state_at=zeros(Int,C); pred=zeros(Int,n)
    for (j,i) in enumerate(states); refs[i]=j; end
    for (j,i) in enumerate(labels); state_at[j]=i; slots[i]=j; end
    if family isa OrderedThresholdFamily
        for k in 2:length(family.states); pred[family.states[k]]=family.states[k-1]; end
    end
    mpi=Matrix{Union{Missing,T}}(undef,C,A); fill!(mpi,missing)
    lex=Vector{Union{Nothing,ExactRational}}(undef,K); fill!(lex,nothing)
    ws=ReducedFPWorkspace(model,criterion,b,w,family,states,refs,state_at,slots,act,pred,m,
        X,zeros(T,C,C),phi,psi,zeros(T,C),zeros(T,C),zeros(T,C),zeros(T,C),zeros(T,C),zeros(T,C),
        pscale,zeros(T,C),zeros(T,C),zeros(T,C),ell,collect(1:size(ell,2)),false,fh,gh,RW,RV,zeros(T,C),zeros(T,C),zeros(T,C),
        zeros(Int,C),0,fill(false,n),zeros(Int,n),zeros(Int,n),zeros(Int,n),zeros(Int,n),fill(false,n),
        mpi,zeros(Int,K),zeros(Int,K),zeros(T,K),zeros(T,K),zeros(T,K),zeros(T,K),zeros(Int,K),
        fill(:unassigned,K),lex,0,counts)
    _fp_recompute_current!(ws)
    all(isfinite,ws.f) && all(isfinite,ws.g) && all(isfinite,ws.g_scale) ||
        error("Nonfinite initial current comparisons.")
    return ws
end

function _fp_recompute_current!(w::ReducedFPWorkspace{T}) where {T}
    for s in 1:w.m
        i=w.state_at[s]; a=w.actions[i]
        if a==1
            w.f[s]=w.phi[s]; w.g[s]=w.psi[s]; w.g_scale[s]=w.psi_scale[s]
        else
            r=_fp_rowid(w,i,a); f=w.fhat[r]; g=w.ghat[r]; sc=abs(g)
            for t in 1:w.m
                l=w.ell[w.reference_slot[w.state_at[t]],w.reference_columns[r]]
                f+=l*w.phi[t]; g+=l*w.psi[t]; sc+=abs(l)*w.psi_scale[t]
            end
            w.f[s]=f; w.g[s]=g; w.g_scale[s]=sc
        end
    end
    return nothing
end

# Allocation-free graph screen on the proposed successor, using stored positive
# probabilities exactly as the baseline. A state remains in the Markov chain
# after its reduced coordinate disappears. No transition thresholding.
@inline function _fp_edge(w,i,j,changed)
    a=w.actions[i]-(i==changed ? 1 : 0)
    return w.model.P[i,j,a+1]>0
end
function _fp_unichain_successor!(w,changed)
    n=nstates(w.model); fill!(w.graph_seen,false); fill!(w.graph_component,0)
    finish=0
    for root in 1:n
        w.graph_seen[root] && continue
        top=1; w.graph_nodes[top]=root; w.graph_next[top]=1; w.graph_seen[root]=true
        while top>0
            u=w.graph_nodes[top]; j=w.graph_next[top]
            while j<=n && (!(_fp_edge(w,u,j,changed)) || w.graph_seen[j]); j+=1; end
            if j>n
                finish+=1; w.graph_order[finish]=u; top-=1
            else
                w.graph_next[top]=j+1; top+=1; w.graph_nodes[top]=j
                w.graph_next[top]=1; w.graph_seen[j]=true
            end
        end
    end
    nc=0
    for k in n:-1:1
        root=w.graph_order[k]; w.graph_component[root]!=0 && continue
        nc+=1; top=1; w.graph_nodes[1]=root; w.graph_component[root]=nc
        while top>0
            u=w.graph_nodes[top]; top-=1
            for j in 1:n
                if w.graph_component[j]==0 && _fp_edge(w,j,u,changed)
                    w.graph_component[j]=nc; top+=1; w.graph_nodes[top]=j
                end
            end
        end
    end
    fill!(w.graph_closed,true)
    for j in 1:n, i in 1:n
        if w.graph_component[i]!=w.graph_component[j] && _fp_edge(w,i,j,changed)
            w.graph_closed[w.graph_component[i]]=false
        end
    end
    nclosed=0
    for k in 1:nc; nclosed+=w.graph_closed[k]; end
    return nclosed==1
end

# Fuse survivor packing with the numerical update. No physical permutation of
# the accepted tableau and no separate copy of its survivor block are needed.
@inline _fp_source(s,p,m,final) = final && s==p ? m : s

# Matrix updates use contiguous SIMD loops, with finite checks fused into the
# write. Trial storage remains separate, so rejection is fully transactional.
# No @fastmath: nonfinite values must be detected and ordinary rounding retained.
function _checked_rankone!(dest,src,v,z,eta,m)
    valid=true
    @inbounds for c in 1:m
        zc=z[c]/eta
        @simd for r in 1:m
            value=src[r,c]-v[r]*zc
            dest[r,c]=value;valid &= isfinite(value)
        end
    end
    return valid
end

function _fp_checked_final!(dest,src,v,z,eta,m,p)
    rdim=m-1
    p==m && return _checked_rankone!(dest,src,v,z,eta,rdim)
    valid=true
    @inbounds for c in 1:rdim
        oc=c==p ? m : c;zc=z[oc]/eta
        # One exceptional packed row; the other rows use direct contiguous
        # addresses, without a source-slot branch at every matrix element.
        @simd for r in 1:p-1
            value=src[r,oc]-v[r]*zc
            dest[r,c]=value;valid &= isfinite(value)
        end
        value=src[m,oc]-v[m]*zc
        dest[p,c]=value;valid &= isfinite(value)
        @simd for r in p+1:rdim
            value=src[r,oc]-v[r]*zc
            dest[r,c]=value;valid &= isfinite(value)
        end
    end
    return valid
end

"""
    pivot_reduced_fp!(workspace, original_state; pivot_atol=0, pivot_rtol=1e-12,
                      check_unichain=true)

One transactional numerical downshift, NOT greedy selection or DAI verification.
Both nonfinal and final updates use pre-pivot data; final deletion replaces the
removed physical slot by the last active slot. Writes go straight into reusable
trial storage. Accepted/trial buffers are exchanged only after finite checks.
Failure preserves accepted arrays/maps/actions; scratch/counters may change.
The default current-comparison cache includes temporarily infeasible pairs.
A performance workspace may select frontier_only storage: only feasible current
comparisons are then valid, and reentering comparisons are freshly materialized.
"""
function pivot_reduced_fp!(w::ReducedFPWorkspace{T},i::Integer;
        pivot_atol::Real=0,pivot_rtol::Real=1e-12,check_unichain::Bool=true) where {T}
    1<=i<=nstates(w.model) || throw(BoundsError(w.actions,i))
    p=w.slot_of[i]; p>0 || throw(ArgumentError("State has no active coordinate."))
    a=w.actions[i]; a>0 || throw(ArgumentError("No downshift at state."))
    pa,pr=T(pivot_atol),T(pivot_rtol)
    isfinite(pa) && isfinite(pr) && pa>=0 && pr>=0 || throw(ArgumentError("Invalid pivot tolerances."))
    m=w.m; final=a==1; rdim=m-(final ? 1 : 0)
    w.counts.trial_attempts+=1
    f=w.f[p]; g=w.g[p]
    scale=one(T)
    if final
        eta=w.X[p,p]; scale+=abs(eta-one(T))
        @inbounds for s in 1:m; w.v[s]=w.X[s,p]; w.z[s]=w.X[p,s]; end
    else
        r=_fp_rowid(w,i,a)
        @inbounds for s in 1:m
            w.v[s]=w.X[s,p]; w.row[s]=w.ell[w.reference_slot[w.state_at[s]],w.reference_columns[r]]
        end
        mul!(@view(w.z[1:m]),transpose(@view(w.X[1:m,1:m])),@view(w.row[1:m]))
        eta=one(T)+w.z[p]
        for s in 1:m; scale+=abs(w.row[s]*w.v[s]); end
    end
    if w.criterion==:average && check_unichain
        w.counts.unichain_screens+=1
        _fp_unichain_successor!(w,Int(i)) ||
            return FPPivotReport(false,:unsupported_average_policy,(Int(i),a),eta,scale,m,m)
    end
    if !isfinite(eta) || !isfinite(scale) || !(eta>pa+pr*scale) || !isfinite(f) || !isfinite(g)
        return FPPivotReport(false,:unsafe_denominator,(Int(i),a),eta,scale,m,m)
    end
    ok=final ? _fp_checked_final!(w.trial_X,w.X,w.v,w.z,eta,m,p) :
               _checked_rankone!(w.trial_X,w.X,w.v,w.z,eta,m)
    # The graph traversal has finished. Reuse its node scratch for one lookup
    # from each active slot to its fixed reference label, shared by all metrics.
    refs=w.graph_nodes
    @inbounds for t in 1:m;refs[t]=w.reference_slot[w.state_at[t]];end
    # The selected row is no longer needed after z has been formed. Pack v
    # once by fixed reference label, so retained comparisons can use BLAS dot
    # products without repeatedly gathering the same active-coordinate map.
    packed = !isempty(w.fhat) && 4m >= length(w.states)
    if packed
        fill!(w.row,zero(T))
        @inbounds for t in 1:m; w.row[refs[t]]=w.v[t]; end
    end
    fdiv=f/eta; gdiv=g/eta
    ok &= isfinite(fdiv) && isfinite(gdiv)
    @inbounds for s in 1:rdim
        os=_fp_source(s,p,m,final)
        ph=w.phi[os]-fdiv*w.v[os]; ps=w.psi[os]-gdiv*w.v[os]
        sc=w.psi_scale[os]+abs(gdiv*w.v[os])
        w.trial_phi[s]=ph; w.trial_psi[s]=ps; w.trial_psi_scale[s]=sc
        ok &= isfinite(ph) && isfinite(ps) && isfinite(sc)
    end
    refreshed=0; exposed=0; dot_elements=0; materialized_elements=0
    @inbounds for s in 1:rdim
        os=_fp_source(s,p,m,final); state=w.state_at[os]
        gear=w.actions[state]-(state==i ? 1 : 0)
        if w.frontier_only && !_fp_successor_feasible(w,w.family,state,i)
            # Explicitly invalid cache entries cannot be used for selection.
            # Accepted state stays unchanged until every trial check passes.
            w.trial_f[s]=T(NaN);w.trial_g[s]=T(NaN);w.trial_g_scale[s]=T(NaN)
            continue
        end
        if gear==1
            ff=w.trial_phi[s]; gg=w.trial_psi[s]; gs=w.trial_psi_scale[s]
        elseif state==i || (w.frontier_only && !_fp_feasible(w,w.family,state))
            # A newly exposed comparison, evaluated from the NEW aggregates.
            r=_fp_rowid(w,state,gear); ff=w.fhat[r]; gg=w.ghat[r]; gs=abs(gg)
            for t in 1:rdim
                ot=_fp_source(t,p,m,final)
                l=w.ell[refs[ot],w.reference_columns[r]]
                ff+=l*w.trial_phi[t]; gg+=l*w.trial_psi[t]
                gs+=abs(l)*w.trial_psi_scale[t]
            end
            exposed+=1;materialized_elements+=rdim
        else
            r=_fp_rowid(w,state,gear); numerator=zero(T)
            column=w.reference_columns[r]
            if packed
                numerator=dot(@view(w.ell[:,column]),w.row)
            else
                # At a small active dimension, avoid reading the inactive
                # part of the fixed reference column. No fast-math flags.
                @simd for t in 1:m
                    numerator+=w.ell[refs[t],column]*w.v[t]
                end
            end
            chi=numerator/eta
            ff=w.f[os]-chi*f; gg=w.g[os]-chi*g
            gs=w.g_scale[os]+abs(chi*g)
            refreshed+=1;dot_elements+=packed ? length(w.states) : m
        end
        w.trial_f[s]=ff; w.trial_g[s]=gg; w.trial_g_scale[s]=gs
        ok &= isfinite(ff) && isfinite(gg) && isfinite(gs)
    end
    ok || return FPPivotReport(false,:nonfinite_trial,(Int(i),a),eta,scale,m,m)
    # Commit: references, not matrix contents, are exchanged.
    w.X,w.trial_X=w.trial_X,w.X
    w.phi,w.trial_phi=w.trial_phi,w.phi; w.psi,w.trial_psi=w.trial_psi,w.psi
    w.f,w.trial_f=w.trial_f,w.f; w.g,w.trial_g=w.trial_g,w.g
    w.psi_scale,w.trial_psi_scale=w.trial_psi_scale,w.psi_scale
    w.g_scale,w.trial_g_scale=w.trial_g_scale,w.g_scale
    if final
        moved=w.state_at[m]
        if p!=m; w.state_at[p]=moved; w.slot_of[moved]=p; end
        w.state_at[m]=0; w.slot_of[i]=0; w.m=rdim
        w.counts.final_pivots+=1
    else
        w.counts.nonfinal_pivots+=1
    end
    w.actions[i]-=1
    w.counts.retained_refreshes+=refreshed; w.counts.new_comparisons+=exposed
    w.counts.reference_dot_elements+=dot_elements
    w.counts.materialization_elements+=materialized_elements
    return FPPivotReport(true,:accepted,(Int(i),a),eta,scale,m,rdim)
end

"""Rebuild the CURRENT accepted tableau from one fresh factorization; exceptional work."""
function rebuild_reduced_fp!(w::ReducedFPWorkspace;backward_tol::Real=1e-10)
    w.counts.rebuild_attempts+=1
    labels=w.state_at[1:w.m]
    x,ph,ps,sc,_,_=_fp_fresh(w.model,w.actions,w.criterion,w.beta,w.omega,labels;
                            backward_tol=backward_tol)
    # _fp_fresh validated the solve. Stage every derived cache value before commit.
    for j in 1:w.m, i in 1:w.m; w.trial_X[i,j]=x[i,j]; end
    for s in 1:w.m
        w.trial_phi[s]=ph[s]; w.trial_psi[s]=ps[s]; w.trial_psi_scale[s]=sc[s]
    end
    for s in 1:w.m
        i=w.state_at[s]; a=w.actions[i]
        if a==1
            ff=ph[s]; gg=ps[s]; gs=sc[s]
        else
            r=_fp_rowid(w,i,a); ff=w.fhat[r]; gg=w.ghat[r]; gs=abs(gg)
            for t in 1:w.m
                l=w.ell[w.reference_slot[w.state_at[t]],w.reference_columns[r]]
                ff+=l*ph[t]; gg+=l*ps[t]; gs+=abs(l)*sc[t]
            end
        end
        all(isfinite,(ff,gg,gs)) || error("Nonfinite rebuilt current comparison; accepted state unchanged.")
        w.trial_f[s]=ff; w.trial_g[s]=gg; w.trial_g_scale[s]=gs
    end
    w.X,w.trial_X=w.trial_X,w.X
    w.phi,w.trial_phi=w.trial_phi,w.phi; w.psi,w.trial_psi=w.trial_psi,w.psi
    w.f,w.trial_f=w.trial_f,w.f; w.g,w.trial_g=w.trial_g,w.g
    w.psi_scale,w.trial_psi_scale=w.trial_psi_scale,w.psi_scale
    w.g_scale,w.trial_g_scale=w.trial_g_scale,w.g_scale
    w.counts.rebuilds+=1
    return nothing
end
