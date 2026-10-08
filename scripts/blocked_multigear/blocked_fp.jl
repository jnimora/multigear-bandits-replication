module BlockedMultiGearFP
using LinearAlgebra
if !isdefined(parentmodule(@__MODULE__), :MultiGearBandits)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "../../src/MultiGearBandits.jl"))
end
const M=getfield(parentmodule(@__MODULE__), :MultiGearBandits)

# Experimental index-only workspace. core.X is a deferred base, NOT the current
# tableau when q>0. Use materialized_core for independent audits. Model, family,
# preparation and every selection/monotonicity safeguard are shared with M.
Base.@kwdef mutable struct Workspace{R}
    run::R
    U::Matrix{Float64}
    V::Matrix{Float64}
    small::Matrix{Float64}
    smallvec::Vector{Float64}
    q::Int=0
    blocksize::Int
    absolute_bound::Float64
    reference_width::Int
    L::Matrix{Float64}
    Y::Matrix{Float64}
    trial_Y::Matrix{Float64}
    coefficients::Vector{Float64}
    reference_ids::Vector{Int}
    reference_lookup::Vector{Int}
    reference_order::Vector{Int}
    reference_position::Vector{Int}
    ages::Vector{Int}
    clock::Int=0
    flushes::Int=0
    cached_rows::Int=0
    direct_rows::Int=0
    panel_fills::Int=0
    flushed_rank::Int=0
    flush_work::Float64=0.0
end

function wrap(run;blocksize::Int=32,reference_width::Int=0,reference_panels::Int=2,reference_order::Symbol=:physical)
    run.algorithm==:FP && run.log.k==0 && run.status==:ready ||
        throw(ArgumentError("Wrap a fresh FP performance run"))
    1<=blocksize<=4096 && 0<=reference_width<=4096 && 1<=reference_panels<=64 ||
        throw(ArgumentError("Invalid block/cache dimensions"))
    w=run.core;n=length(w.states);b=min(blocksize,max(n,1))
    bound=maximum(abs,w.X;init=0.0)
    isfinite(bound) && bound<=floatmax(Float64)/16 ||
        throw(ArgumentError("Initial tableau exceeds the finite safety envelope"))
    width=min(reference_width,size(w.ell,2));panels=width==0 ? 0 : min(reference_panels,cld(size(w.ell,2),width))
    capacity=width*panels
    reference_order in (:state,:physical) || throw(ArgumentError("Unknown reference order"))
    order=Int[];position=zeros(Int,size(w.ell,2))
    if reference_order==:state
        # Group gear rows by state without using the realized policy/index path.
        # Exact shared reference columns are inserted once, never approximately merged.
        for i in w.states,a in 2:M.maxgear(w.model)
            id=w.reference_columns[M._fp_rowid(w,i,a)]
            position[id]>0 && continue
            push!(order,id);position[id]=length(order)
        end
    else
        append!(order,eachindex(position));position.=order
    end
    length(order)==length(position) || error("Incomplete reference permutation")
    Workspace(run=run,U=zeros(n,b),V=zeros(n,b),small=zeros(b,width),smallvec=zeros(b),
        blocksize=b,absolute_bound=bound,reference_width=width,
        L=zeros(n,capacity),Y=zeros(n,capacity),trial_Y=zeros(n,capacity),
        coefficients=zeros(capacity),reference_ids=zeros(Int,capacity),
        reference_lookup=zeros(Int,size(w.ell,2)),reference_order=order,reference_position=position,ages=zeros(Int,panels))
end
function prepare(input;blocksize=32,reference_width=0,reference_panels=2,reference_order=:physical,kwargs...)
    wrap(M.prepare_performance_run(input,:FP;kwargs...);blocksize,reference_width,reference_panels,reference_order)
end

# A panel is chosen solely by the fixed reference permutation and cache width.
# No realized/future policy path, transition topology, or index order is used.
function projected_row!(b,logical_row)
    w=b.run.core;m=w.m;q=b.q;column=w.reference_columns[logical_row]
    if b.reference_width==0
        @inbounds for t in 1:m;w.row[t]=w.ell[w.reference_slot[w.state_at[t]],column];end
        mul!(@view(w.z[1:m]),transpose(@view(w.X[1:m,1:m])),@view(w.row[1:m]))
        if q>0
            mul!(@view(b.smallvec[1:q]),transpose(@view(b.U[1:m,1:q])),@view(w.row[1:m]))
            mul!(@view(w.z[1:m]),@view(b.V[1:m,1:q]),@view(b.smallvec[1:q]),-1.0,1.0)
        end
        b.direct_rows+=1
    else
        slot=b.reference_lookup[column]
        b.clock+=1
        if slot==0
            panel=argmin(b.ages);start=(panel-1)*b.reference_width+1
            position=b.reference_position[column]
            lo=div(position-1,b.reference_width)*b.reference_width+1
            count=min(b.reference_width,size(w.ell,2)-lo+1)
            @inbounds for j in start:start+b.reference_width-1
                id=b.reference_ids[j];id>0 && (b.reference_lookup[id]=0)
                b.reference_ids[j]=0
                for t in 1:m;b.L[t,j]=0.0;b.Y[t,j]=0.0;end
            end
            @inbounds for k in 1:count
                id=b.reference_order[lo+k-1];j=start+k-1
                for t in 1:m;b.L[t,j]=w.ell[w.reference_slot[w.state_at[t]],id];end
            end
            stop=start+count-1
            mul!(@view(b.Y[1:m,start:stop]),transpose(@view(w.X[1:m,1:m])),@view(b.L[1:m,start:stop]))
            if q>0
                mul!(@view(b.small[1:q,1:count]),transpose(@view(b.U[1:m,1:q])),@view(b.L[1:m,start:stop]))
                mul!(@view(b.Y[1:m,start:stop]),@view(b.V[1:m,1:q]),@view(b.small[1:q,1:count]),-1.0,1.0)
            end
            all(isfinite,@view(b.Y[1:m,start:stop])) || return false
            @inbounds for k in 1:count
                id=b.reference_order[lo+k-1];j=start+k-1;b.reference_ids[j]=id;b.reference_lookup[id]=j
            end
            slot=start+position-lo;b.panel_fills+=1
        end
        b.ages[div(slot-1,b.reference_width)+1]=b.clock
        @inbounds for t in 1:m;w.z[t]=b.Y[t,slot];end
        b.cached_rows+=1
    end
    all(isfinite,@view(w.z[1:m]))
end

function pivot_vectors!(b,p,final)
    w=b.run.core;m=w.m;q=b.q
    @inbounds for t in 1:m;w.v[t]=w.X[t,p];end
    # Stream each factor column contiguously; preserve each entry's correction
    # order. This is the same memory-access pattern as the binary prototype.
    @inbounds for l in 1:q
        vp=b.V[p,l]
        @simd for t in 1:m;w.v[t]-=b.U[t,l]*vp;end
    end
    if final
        @inbounds for t in 1:m;w.z[t]=w.X[p,t];end
        @inbounds for l in 1:q
            up=b.U[p,l]
            @simd for t in 1:m;w.z[t]-=b.V[t,l]*up;end
        end
        eta=w.v[p];scale=1.0+abs(eta-1.0)
    else
        logical_row=M._fp_rowid(w,w.state_at[p],w.actions[w.state_at[p]])
        projected_row!(b,logical_row) || return (NaN,NaN)
        eta=1.0+w.z[p];scale=1.0
        column=w.reference_columns[logical_row]
        @inbounds for t in 1:m
            scale+=abs(w.ell[w.reference_slot[w.state_at[t]],column]*w.v[t])
        end
    end
    eta,scale
end

# Stage EVERY scalar/vector/cache check before changing accepted state.
# Full tableau finiteness is checked at flushes. Between flushes a conservative
# absolute envelope bounds all partial products/sums well below Float64 overflow;
# a case outside this envelope is rejected, not silently allowed through.
function stage_block!(b,eta)
    w=b.run.core;m=w.m;q=b.q
    vmax=0.0;zmax=0.0
    @inbounds for t in 1:m
        z=w.z[t]/eta;v=w.v[t]
        isfinite(z) && isfinite(v) || return false
        b.U[t,q+1]=v;b.V[t,q+1]=z
        vmax=max(vmax,abs(v));zmax=max(zmax,abs(z))
    end
    bound=b.absolute_bound+vmax*zmax
    isfinite(bound) && bound<=floatmax(Float64)/16 || return false
    if !isempty(b.coefficients)
        mul!(b.coefficients,transpose(@view(b.L[1:m,:])),@view(w.v[1:m]))
        @inbounds for j in eachindex(b.reference_ids)
            b.reference_ids[j]>0 || continue
            coeff=b.coefficients[j]/eta
            isfinite(coeff) || return false
            @simd for t in 1:m
                b.trial_Y[t,j]=b.Y[t,j]-coeff*w.z[t]
            end
            all(isfinite,@view(b.trial_Y[1:m,j])) || return false
        end
    end
    if q+1==b.blocksize
        @inbounds for j in 1:m,t in 1:m;w.trial_X[t,j]=w.X[t,j];end
        mul!(@view(w.trial_X[1:m,1:m]),@view(b.U[1:m,1:q+1]),
             transpose(@view(b.V[1:m,1:q+1])),-1.0,1.0)
        all(isfinite,@view(w.trial_X[1:m,1:m])) || return false
    end
    true
end

function commit_block!(b,p,final)
    w=b.run.core;m=w.m;q=b.q+1
    if q==b.blocksize
        w.X,w.trial_X=w.trial_X,w.X
        b.absolute_bound=maximum(abs,@view(w.X[1:m,1:m]);init=0.0)
        b.flushes+=1;b.flushed_rank+=q;b.flush_work+=2.0*m*m*q;q=0
    else
        vmax=maximum(abs,@view(b.U[1:m,q]);init=0.0)
        zmax=maximum(abs,@view(b.V[1:m,q]);init=0.0)
        b.absolute_bound+=vmax*zmax
    end
    b.Y,b.trial_Y=b.trial_Y,b.Y
    if final && p!=m
        @inbounds for j in 1:m;w.X[p,j],w.X[m,j]=w.X[m,j],w.X[p,j];end
        @inbounds for t in 1:m;w.X[t,p],w.X[t,m]=w.X[t,m],w.X[t,p];end
        @inbounds for l in 1:q
            b.U[p,l],b.U[m,l]=b.U[m,l],b.U[p,l]
            b.V[p,l],b.V[m,l]=b.V[m,l],b.V[p,l]
        end
        @inbounds for j in eachindex(b.reference_ids)
            b.L[p,j]=b.L[m,j];b.Y[p,j]=b.Y[m,j]
        end
    end
    b.q=q
    nothing
end

function pivot!(b::Workspace,i::Integer;
        pivot_atol::Real=0,pivot_rtol::Real=1e-12,check_unichain::Bool=true)
    w=b.run.core;T=Float64
    1<=i<=M.nstates(w.model) || throw(BoundsError(w.actions,i))
    p=w.slot_of[i]; p>0 || throw(ArgumentError("State has no active coordinate."))
    a=w.actions[i]; a>0 || throw(ArgumentError("No downshift at state."))
    pa,pr=T(pivot_atol),T(pivot_rtol)
    isfinite(pa) && isfinite(pr) && pa>=0 && pr>=0 || throw(ArgumentError("Invalid pivot tolerances."))
    m=w.m; final=a==1; rdim=m-(final ? 1 : 0)
    w.counts.trial_attempts+=1
    f=w.f[p]; g=w.g[p]
    eta,scale=pivot_vectors!(b,p,final)
    if w.criterion==:average && check_unichain
        w.counts.unichain_screens+=1
        M._fp_unichain_successor!(w,Int(i)) ||
            return M.FPPivotReport(false,:unsupported_average_policy,(Int(i),a),eta,scale,m,m)
    end
    if !isfinite(eta) || !isfinite(scale) || !(eta>pa+pr*scale) || !isfinite(f) || !isfinite(g)
        return M.FPPivotReport(false,:unsafe_denominator,(Int(i),a),eta,scale,m,m)
    end
    ok=true
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
        os=M._fp_source(s,p,m,final)
        ph=w.phi[os]-fdiv*w.v[os]; ps=w.psi[os]-gdiv*w.v[os]
        sc=w.psi_scale[os]+abs(gdiv*w.v[os])
        w.trial_phi[s]=ph; w.trial_psi[s]=ps; w.trial_psi_scale[s]=sc
        ok &= isfinite(ph) && isfinite(ps) && isfinite(sc)
    end
    refreshed=0; exposed=0; dot_elements=0; materialized_elements=0
    @inbounds for s in 1:rdim
        os=M._fp_source(s,p,m,final); state=w.state_at[os]
        gear=w.actions[state]-(state==i ? 1 : 0)
        if w.frontier_only && !M._fp_successor_feasible(w,w.family,state,i)
            # Explicitly invalid cache entries cannot be used for selection.
            # Accepted state stays unchanged until every trial check passes.
            w.trial_f[s]=T(NaN);w.trial_g[s]=T(NaN);w.trial_g_scale[s]=T(NaN)
            continue
        end
        if gear==1
            ff=w.trial_phi[s]; gg=w.trial_psi[s]; gs=w.trial_psi_scale[s]
        elseif state==i || (w.frontier_only && !M._fp_feasible(w,w.family,state))
            # A newly exposed comparison, evaluated from the NEW aggregates.
            r=M._fp_rowid(w,state,gear); ff=w.fhat[r]; gg=w.ghat[r]; gs=abs(gg)
            for t in 1:rdim
                ot=M._fp_source(t,p,m,final)
                l=w.ell[refs[ot],w.reference_columns[r]]
                ff+=l*w.trial_phi[t]; gg+=l*w.trial_psi[t]
                gs+=abs(l)*w.trial_psi_scale[t]
            end
            exposed+=1;materialized_elements+=rdim
        else
            r=M._fp_rowid(w,state,gear); numerator=zero(T)
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
    ok || return M.FPPivotReport(false,:nonfinite_trial,(Int(i),a),eta,scale,m,m)
    stage_block!(b,eta) || return M.FPPivotReport(false,:nonfinite_trial,(Int(i),a),eta,scale,m,m)
    commit_block!(b,p,final)
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
    return M.FPPivotReport(true,:accepted,(Int(i),a),eta,scale,m,rdim)
end

# Callable update keeps the parent implementation's complete selection,
# fallback, in-loop monotonicity and log-commit path unchanged.
function (b::Workspace)(w,i,o)
    report=pivot!(b,i;pivot_atol=o.pivot_atol,pivot_rtol=o.pivot_rtol,check_unichain=false)
    report.accepted || return (report.reason,report.denominator)
    M._fp_frontier!(w)
    (:accepted,report.denominator)
end
step!(b::Workspace)=M._step_performance_with_update!(b.run,b)
function run!(b::Workspace)
    while b.run.status in (:ready,:running,:step_limit);step!(b);end
    b
end
snapshot(b::Workspace)=M.performance_snapshot(b.run)

# Shallow diagnostic view: only the tableau and audit counter need owned copies.
# Model and accepted vectors are read-only during audit. No materialization or
# audit is hidden in the timed loop or allowed to change the future pivot path.
function materialized_core(b::Workspace)
    w=b.run.core;args=[getfield(w,k) for k in fieldnames(typeof(w))]
    shadow=typeof(w)(args...);shadow.X=copy(w.X);shadow.counts=deepcopy(w.counts)
    m=w.m;q=b.q
    if q>0
        mul!(@view(shadow.X[1:m,1:m]),@view(b.U[1:m,1:q]),transpose(@view(b.V[1:m,1:q])),-1.0,1.0)
    end
    shadow
end
audit(b::Workspace;kwargs...)=M.audit_reduced_fp(materialized_core(b);kwargs...)
end
