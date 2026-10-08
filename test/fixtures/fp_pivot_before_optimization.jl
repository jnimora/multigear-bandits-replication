# Frozen checked pivot before the 2026-10-02 efficiency pass. Test oracle only.
# Included inside MultiGearBandits; uses the same workspace and report types.
function _pivot_reduced_fp_before_optimization!(w::ReducedFPWorkspace{T},i::Integer;
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
            w.v[s]=w.X[s,p]; w.row[s]=w.ell[w.reference_slot[w.state_at[s]],r]
        end
        @inbounds for c in 1:m
            z=zero(T)
            for s in 1:m; z+=w.row[s]*w.X[s,c]; end
            w.z[c]=z
        end
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
    # Column-major traversal; no outer product and no X[keep,keep] allocation.
    ok=true
    @inbounds for c in 1:rdim
        oc=_fp_source(c,p,m,final); zc=w.z[oc]/eta
        @inbounds for s in 1:rdim
            os=_fp_source(s,p,m,final)
            x=w.X[os,oc]-w.v[os]*zc
            w.trial_X[s,c]=x; ok &= isfinite(x)
        end
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
    refreshed=0; exposed=0
    @inbounds for s in 1:rdim
        os=_fp_source(s,p,m,final); state=w.state_at[os]
        gear=w.actions[state]-(state==i ? 1 : 0)
        if gear==1
            ff=w.trial_phi[s]; gg=w.trial_psi[s]; gs=w.trial_psi_scale[s]
        elseif state==i
            # A newly exposed comparison, evaluated from the NEW aggregates.
            r=_fp_rowid(w,state,gear); ff=w.fhat[r]; gg=w.ghat[r]; gs=abs(gg)
            for t in 1:rdim
                ot=_fp_source(t,p,m,final)
                l=w.ell[w.reference_slot[w.state_at[ot]],r]
                ff+=l*w.trial_phi[t]; gg+=l*w.trial_psi[t]
                gs+=abs(l)*w.trial_psi_scale[t]
            end
            exposed+=1
        else
            r=_fp_rowid(w,state,gear); numerator=zero(T)
            for t in 1:m
                numerator+=w.ell[w.reference_slot[w.state_at[t]],r]*w.v[t]
            end
            chi=numerator/eta
            ff=w.f[os]-chi*f; gg=w.g[os]-chi*g
            gs=w.g_scale[os]+abs(chi*g)
            refreshed+=1
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
    return FPPivotReport(true,:accepted,(Int(i),a),eta,scale,m,rdim)
end
