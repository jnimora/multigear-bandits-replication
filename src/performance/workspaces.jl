# Performance counterparts of the audited reference algorithms.
# Julia >= 1.13: dense lu!(F, M) reuses both factor and pivot storage.
# No @fastmath, shared mutable workspace, changing BLAS setting, or global RNG.

Base.@kwdef struct PerformanceOptions
    comparison_atol::Float64 = 1e-11
    comparison_rtol::Float64 = 1e-9
    monotonicity_atol::Float64 = 1e-10
    monotonicity_rtol::Float64 = 1e-9
    pivot_atol::Float64 = 0.0
    pivot_rtol::Float64 = 1e-12
    exact_fallback::Bool = true
    exact_max_states::Int = 8
    terminal_update::Bool = false
    fp_frontier_cache::Bool = true
    # Opt-in; old archived configurations retain their original behavior.
    residual_fallback::Bool = false
    residual_max_states::Int = 256
    residual_bound::Symbol = :inverse
end

function _performance_check_options(o::PerformanceOptions)
    _check_monotonicity_tolerances(o.monotonicity_atol,o.monotonicity_rtol)
    all(isfinite, (o.comparison_atol,o.comparison_rtol,o.pivot_atol,o.pivot_rtol)) ||
        throw(ArgumentError("Nonfinite performance tolerances."))
    min(o.comparison_atol,o.comparison_rtol,o.pivot_atol,o.pivot_rtol)>=0 ||
        throw(ArgumentError("Performance tolerances must be nonnegative."))
    o.exact_max_states>=1 || throw(ArgumentError("Invalid exact-fallback limit."))
    o.residual_max_states>=1 || throw(ArgumentError("Invalid residual-fallback limit."))
    o.residual_bound in (:inverse,:contraction) || throw(ArgumentError("Invalid residual bound method."))
    return nothing
end

"""
Validated shared INPUT, not an algorithm workspace. Creation/model generation
are common preparation and are reported separately, never charged only to one
algorithm. The dense pilot's average mode requires EVERY admissible transition
entry to be strictly positive. This sufficient condition makes every policy
irreducible. Explicit average_topology=:reset_or_increment also admits only the checked
reset/progression support with a single uncontrollable age-zero state. Explicit average_topology=:common_reachable_state checks that all states reach
a shared root through edges positive under every admissible action. Other models are rejected,
not classified as nonindexable. No DAI claim follows from this input check.
"""
struct PerformanceInput
    model::FiniteMultiGearModel{Float64}
    all_policies_positive::Bool
    average_topology::Symbol
    function PerformanceInput(model::FiniteMultiGearModel{Float64};average_topology::Symbol=:dense_positive)
        positive=true
        for i in 1:nstates(model), a in admissible_gears(model,i), j in 1:nstates(model)
            positive &= model.P[i,j,a+1]>0
        end
        average_topology in (:dense_positive,:reset_or_increment,:common_reachable_state) ||
            throw(ArgumentError("Unknown average performance topology admission."))
        average_topology==:reset_or_increment && _check_reset_or_increment_topology(model)
        average_topology==:common_reachable_state && _check_common_reachable_state(model)
        new(model,positive,average_topology)
    end
end

# An explicit support proof, not a user-asserted positivity flag. From state 1
# (age zero), every positive age is reachable along positive progression edges;
# from any positive age, state 1 is reachable in one step under every action.
# A positive self-loop at state 1 also makes every policy aperiodic. This does
# NOT establish indexability. No transition entry is modified or normalized here.
function _check_reset_or_increment_topology(m::FiniteMultiGearModel{Float64})
    n=nstates(m);n>=2 || throw(ArgumentError("Reset/increment topology needs age zero and a positive age."))
    !m.controllable[1] && all(m.controllable[2:end]) ||
        throw(ArgumentError("Reset/increment admission requires only state 1 to be uncontrollable."))
    for i in 1:n, a in admissible_gears(m,i)
        jnext=i==1 ? 2 : min(i+1,n)
        abs(sum(@view(m.P[i,:,a+1]))-1.0)<=min(m.row_atol,1e-12) ||
            throw(ArgumentError("Reset/increment admission requires stochastic rows at the standard tolerance."))
        m.P[i,1,a+1]>0 && m.P[i,jnext,a+1]>0 ||
            throw(ArgumentError("Missing strictly positive reset/progression edge."))
        for j in 1:n
            if j!=1 && j!=jnext
                m.P[i,j,a+1]==0 || throw(ArgumentError("Unexpected edge in reset/increment topology."))
            end
        end
    end
    nothing
end
@inline _performance_average_admitted(input::PerformanceInput) =
    input.all_policies_positive || input.average_topology in (:reset_or_increment,:common_reachable_state)

Base.@kwdef mutable struct PerformanceCounts
    policy_factorizations::Int = 0
    reference_factorizations::Int = 0
    rankone_updates::Int = 0
    direct_frontier_pairs::Int = 0
    retained_pair_refreshes::Int = 0
    new_pair_exposures::Int = 0
    selected_pair_preparations::Int = 0
    exact_comparison_solves::Int = 0
    fallback_direct_solves::Int = 0
end

# Algorithm is a type parameter (:C, :R, or :M), so ordinary kernels specialize.
# Fields are private implementation state; each independent job owns its copy.
Base.@kwdef mutable struct FullPerformanceWorkspace{Alg,F<:AbstractPolicyFamily,L}
    model::FiniteMultiGearModel{Float64}
    criterion::Symbol
    beta::Union{Nothing,Float64}
    omega::Union{Nothing,Vector{Float64}}
    factor::Float64
    family::F
    actions::Vector{Int}
    states::Vector{Int}
    predecessor::Vector{Int}
    frontier::Vector{Int}
    nfrontier::Int = 0
    system::Matrix{Float64}
    rhs::Matrix{Float64}
    values::Matrix{Float64}
    trial_values::Matrix{Float64}
    lu_workspace::L
    inverse::Matrix{Float64}
    trial_inverse::Matrix{Float64}
    delta::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    old_row::Vector{Float64}
    f::Vector{Float64}
    g::Vector{Float64}
    trial_f::Vector{Float64}
    trial_g::Vector{Float64}
    cache_gear::Vector{Int}
    trial_cache_gear::Vector{Int}
    cache_rows::Matrix{Float64}
    cache_offsets::Vector{Int}
    counts::PerformanceCounts = PerformanceCounts()
end

@inline _performance_method(::FullPerformanceWorkspace{A}) where {A} = A
@inline _performance_feasible(w,::UnrestrictedFamily,i)=true
@inline function _performance_feasible(w,::OrderedThresholdFamily,i)
    p=w.predecessor[i]
    return p==0 || w.actions[p]<w.actions[i]
end
function _performance_feasible(w,f::ExplicitPolicyFamily,i)
    for p in f.policies
        same=true
        for j in eachindex(w.actions)
            if p[j]!=w.actions[j]-(j==i ? 1 : 0)
                same=false; break
            end
        end
        same && return true
    end
    return false
end

# Gather into PREALLOCATED column storage; no model row slice is created.
function _performance_delta!(dest,model,i,a,factor)
    n=nstates(model)
    @inbounds for j in 1:n
        dest[j]=factor*(model.P[i,j,a+1]-model.P[i,j,a])
    end
    length(dest)>n && (dest[n+1]=0.0)
    return nothing
end
function _performance_metrics(model,i,a,factor,values)
    f=model.h[i,a]-model.h[i,a+1]
    g=model.c[i,a+1]-model.c[i,a]
    @inbounds for j in 1:nstates(model)
        d=factor*(model.P[i,j,a+1]-model.P[i,j,a])
        f-=d*values[j,1]; g+=d*values[j,2]
    end
    return f,g
end
# In an ordered-threshold family there is at most one feasible boundary per
# positive gear. A gear-indexed cache therefore needs d*min(A,N) entries,
# rather than d*N, without changing any arithmetic or update order.
@inline _performance_cache_column(w,::AbstractPolicyFamily,i,a)=i
@inline _performance_cache_column(w,::OrderedThresholdFamily,i,a)=
    maxgear(w.model)<nstates(w.model) ? a : i
@inline function _performance_cache_column(w,f::RowThresholdFamily,i,a)
    b=f.row_of[i]
    w.cache_offsets[b]+(maxgear(w.model)<length(f.rows[b]) ? a : f.position[i])
end

# A product of chains has at most one feasible boundary per row and gear.
# Short rows use their state positions instead, avoiding over-allocation when
# A exceeds the row length. Columns from different rows cannot alias.
_performance_cache_layout(model,f::AbstractPolicyFamily)=(nstates(model),Int[])
_performance_cache_layout(model,f::OrderedThresholdFamily)=(min(maxgear(model),nstates(model)),Int[])
function _performance_cache_layout(model,f::RowThresholdFamily)
    offsets=zeros(Int,length(f.rows));columns=0
    for b in eachindex(f.rows)
        offsets[b]=columns;columns+=min(maxgear(model),length(f.rows[b]))
    end
    columns,offsets
end
function _performance_cache_row!(w,i,a)
    column=_performance_cache_column(w,w.family,i,a)
    @inbounds for j in 1:nstates(w.model)
        w.cache_rows[j,column]=w.factor*(w.model.P[i,j,a+1]-w.model.P[i,j,a])
    end
    size(w.cache_rows,1)>nstates(w.model) && (w.cache_rows[end,column]=0.0)
    return nothing
end

# Broad frontiers are evaluated by transition column, reusing contiguous
# column-major data instead of repeatedly traversing strided transition rows.
# Each state's summation still proceeds in the original state-column order.
function _performance_direct_frontier!(w)
    q=w.nfrontier
    if q<32
        for k in 1:q
            i=w.frontier[k]
            w.f[i],w.g[i]=_performance_metrics(w.model,i,w.actions[i],w.factor,w.values)
        end
    else
        f=w.f;g=w.g;P=w.model.P;acts=w.actions;front=w.frontier;factor=w.factor
        @inbounds for k in 1:q
            i=front[k];a=acts[i]
            f[i]=w.model.h[i,a]-w.model.h[i,a+1]
            g[i]=w.model.c[i,a+1]-w.model.c[i,a]
        end
        @inbounds for j in 1:nstates(w.model)
            v=w.values[j,1];u=w.values[j,2]
            for k in 1:q
                i=front[k];a=acts[i]
                delta=factor*(P[i,j,a+1]-P[i,j,a])
                f[i]-=delta*v;g[i]+=delta*u
            end
        end
    end
    w.counts.direct_frontier_pairs+=q
    nothing
end

function _performance_frontier!(w::FullPerformanceWorkspace{Alg}) where {Alg}
    q=0
    for i in w.states
        a=w.actions[i]
        if a>0 && _performance_feasible(w,w.family,i)
            q+=1; w.frontier[q]=i
            if Alg==:M && w.cache_gear[i]!=a
                w.f[i],w.g[i]=_performance_metrics(w.model,i,a,w.factor,w.values)
                w.counts.direct_frontier_pairs+=1
                _performance_cache_row!(w,i,a)
                w.cache_gear[i]=a; w.counts.new_pair_exposures+=1
            end
        elseif Alg==:M
            w.cache_gear[i]=0
        end
    end
    w.nfrontier=q
    Alg==:M || _performance_direct_frontier!(w)
    return nothing
end

function _performance_value_error(M,B,X)
    d=size(M,1); mn=0.0
    for i in 1:d
        s=0.0
        @inbounds for j in 1:d; s+=abs(M[i,j]); end
        mn=max(mn,s)
    end
    worst=0.0
    for col in 1:2
        xn=0.0; bn=0.0; rn=0.0
        for i in 1:d
            r=-B[i,col]
            @inbounds for j in 1:d; r+=M[i,j]*X[j,col]; end
            rn=max(rn,abs(r)); xn=max(xn,abs(X[i,col])); bn=max(bn,abs(B[i,col]))
        end
        den=mn*xn+bn
        err=den==0 ? (rn==0 ? 0.0 : Inf) : rn/den
        worst=max(worst,err)
    end
    return worst
end

function _initialize_full_performance(input::PerformanceInput,::Val{Alg};
        criterion=:discounted,beta=nothing,omega=nothing,
        family::AbstractPolicyFamily=UnrestrictedFamily(),preparation_trace=nothing,
        backward_tol::Real=1e-10) where {Alg}
    Alg in (:C,:R,:M) || throw(ArgumentError("Unknown full-state algorithm."))
    VERSION>=v"1.13" || error("The reusable dense-LU performance path requires Julia 1.13 or later.")
    model=input.model; b,om=_rankone_parameters(model,criterion,beta,omega)
    criterion==:average && !_performance_average_admitted(input) && throw(ArgumentError(
        "Average performance requires dense-positive rows or explicit checked reset/increment topology; use the reference routines otherwise."))
    _check_family(model,family)
    n=nstates(model); d=n+(criterion==:average); A=maxgear(model)
    actions=[model.controllable[i] ? A : 0 for i in 1:n]
    states=findall(model.controllable); pred=zeros(Int,n)
    if family isa OrderedThresholdFamily
        for j in 2:length(family.states); pred[family.states[j]]=family.states[j-1]; end
    end
    factor=criterion==:discounted ? b : 1.0
    _preparation_tolerance(backward_tol)
    M,B=_preparation_stage(preparation_trace,:current_assembly) do
        _preparation_system(model,actions,criterion,b,om)
    end
    F=_preparation_stage(preparation_trace,:current_factorization) do
        lu(M;check=true)
    end
    values=_preparation_stage(preparation_trace,:current_value_solve) do
        V=copy(B); ldiv!(F,V); V
    end
    value_error=_preparation_stage(preparation_trace,:current_value_residual) do
        _preparation_column_check(M,values,B,backward_tol)
    end
    _preparation_record!(preparation_trace,:current_value_columns,2,value_error)
    Z=if Alg==:C
        zeros(0,0)
    else
        inverse=_preparation_stage(preparation_trace,:current_inverse_solve) do
            V=Matrix{Float64}(I,d,d); ldiv!(F,V); V
        end
        inverse_error=_preparation_stage(preparation_trace,:current_inverse_residual) do
            _preparation_column_check(M,inverse,nothing,backward_tol;identity_rows=1:d)
        end
        _preparation_record!(preparation_trace,:current_inverse_columns,d,inverse_error)
        inverse
    end
    stored_factor=Alg==:C ? F : nothing
    w=_preparation_stage(preparation_trace,:workspace_buffers) do
        cache_columns,cache_offsets=Alg==:M ? _performance_cache_layout(model,family) : (0,Int[])
        FullPerformanceWorkspace{Alg,typeof(family),typeof(stored_factor)}(
        model=model,criterion=criterion,beta=b,omega=om,factor=factor,family=family,
        actions=actions,states=states,predecessor=pred,frontier=zeros(Int,length(states)),
        system=Alg==:C ? M : zeros(0,0),rhs=B,values=values,trial_values=zeros(d,2),lu_workspace=stored_factor,
        inverse=Z,trial_inverse=zeros(size(Z)),delta=zeros(d),y=zeros(d),z=zeros(d),old_row=Alg==:C ? zeros(d) : Float64[],
        f=zeros(n),g=zeros(n),trial_f=zeros(n),trial_g=zeros(n),cache_gear=zeros(Int,n),
        trial_cache_gear=zeros(Int,n),cache_rows=Alg==:M ? zeros(d,cache_columns) : zeros(0,0),cache_offsets=cache_offsets)
    end
    w.counts.policy_factorizations=1
    _preparation_stage(preparation_trace,:initial_frontier_metrics) do
        _performance_frontier!(w)
    end
    return w
end

# Update only the changed coefficient/RHS row from primitives; no full system
# reconstruction. The old row is retained for rejection/exception rollback.
function _performance_stage_row!(w,i)
    n=nstates(w.model); d=size(w.system,1); a=w.actions[i] # successor storage slot
    @inbounds for j in 1:d; w.old_row[j]=w.system[i,j]; end
    oldh,oldc=w.rhs[i,1],w.rhs[i,2]
    @inbounds for j in 1:n
        w.system[i,j]=(i==j ? 1.0 : 0.0)-w.factor*w.model.P[i,j,a]
    end
    w.rhs[i,1]=w.model.h[i,a]; w.rhs[i,2]=w.model.c[i,a]
    return oldh,oldc
end
function _performance_restore_row!(w,i,h,c)
    @inbounds for j in axes(w.system,2); w.system[i,j]=w.old_row[j]; end
    w.rhs[i,1]=h; w.rhs[i,2]=c
    return nothing
end

function _performance_update!(w::FullPerformanceWorkspace{:C},i,o)
    oldh,oldc=_performance_stage_row!(w,i)
    accepted=false
    try
        # FRESH numeric factorization, but reuse existing factor/pivot storage.
        w.counts.policy_factorizations+=1
        fac=lu!(w.lu_workspace,w.system;check=true)
        copyto!(w.trial_values,w.rhs); ldiv!(fac,w.trial_values)
        all(isfinite,w.trial_values) || return (:nonfinite_update,NaN)
        accepted=true
    catch err
        (err isa SingularException || err isa LinearAlgebra.LAPACKException) || rethrow()
        return (:factorization_failure,NaN)
    finally
        accepted || _performance_restore_row!(w,i,oldh,oldc)
    end
    w.values,w.trial_values=w.trial_values,w.values
    w.actions[i]-=1; _performance_frontier!(w)
    return (:accepted,NaN)
end

# M-DS: cache precisely the successor frontier. Retained row storage is never
# copied; newly exposed/reentering rows are populated only after acceptance.
function _performance_trial_cache!(w,selected,fselected,gselected,eta)
    w.actions[selected]-=1
    try
        for i in w.states
            a=w.actions[i]
            if a>0 && _performance_feasible(w,w.family,i)
                w.trial_cache_gear[i]=a
                if w.cache_gear[i]==a
                    t=0.0;column=_performance_cache_column(w,w.family,i,a)
                    @inbounds for j in eachindex(w.y); t+=w.cache_rows[j,column]*w.y[j]; end
                    t/=eta
                    w.trial_f[i]=w.f[i]-t*fselected
                    w.trial_g[i]=w.g[i]-t*gselected
                    w.counts.retained_pair_refreshes+=1
                else
                    w.trial_f[i],w.trial_g[i]=_performance_metrics(w.model,i,a,w.factor,w.trial_values)
                    w.counts.direct_frontier_pairs+=1; w.counts.new_pair_exposures+=1
                end
                isfinite(w.trial_f[i]) && isfinite(w.trial_g[i]) || return false
            else
                w.trial_cache_gear[i]=0
            end
        end
    finally
        w.actions[selected]+=1
    end
    return true
end

function _performance_update!(w::FullPerformanceWorkspace{Alg},i,o) where {Alg}
    Alg in (:R,:M) || error("Internal algorithm dispatch error.")
    a=w.actions[i]; d=length(w.delta)
    _performance_delta!(w.delta,w.model,i,a,w.factor)
    @inbounds for j in 1:d; w.y[j]=w.inverse[j,i]; end
    # Old pivot column is a reusable COPY, not a view into a mutated matrix.
    mul!(w.z,transpose(w.inverse),w.delta)
    eta=1.0; scale=1.0; fs=w.model.h[i,a]-w.model.h[i,a+1]
    gs=w.model.c[i,a+1]-w.model.c[i,a]
    @inbounds for j in 1:d
        eta+=w.delta[j]*w.y[j]; scale+=abs(w.delta[j]*w.y[j])
        fs-=w.delta[j]*w.values[j,1]; gs+=w.delta[j]*w.values[j,2]
    end
    w.counts.selected_pair_preparations+=1
    all(isfinite,(eta,scale,fs,gs)) || return (:nonfinite_update,eta)
    eta>o.pivot_atol+o.pivot_rtol*scale || return (:unresolved_pivot,eta)
    inverse_finite=_checked_rankone!(w.trial_inverse,w.inverse,w.y,w.z,eta,d)
    @inbounds for j in 1:d
        w.trial_values[j,1]=w.values[j,1]+(fs/eta)*w.y[j]
        w.trial_values[j,2]=w.values[j,2]-(gs/eta)*w.y[j]
    end
    # Exactly zero primitive RHS => exactly zero solution; no tolerance clipping.
    for col in 1:2
        zero_rhs=true
        for j in 1:nstates(w.model)
            v=j==i ? (col==1 ? w.model.h[i,a] : w.model.c[i,a]) : w.rhs[j,col]
            if v!=0; zero_rhs=false; break; end
        end
        if zero_rhs
            @inbounds for j in 1:d; w.trial_values[j,col]=0.0; end
        end
    end
    inverse_finite && all(isfinite,w.trial_values) || return (:nonfinite_update,eta)
    if Alg==:M
        _performance_trial_cache!(w,i,fs,gs,eta) || return (:nonfinite_cache,eta)
    end
    # Commit only after all trial numerical checks succeeded.
    w.rhs[i,1]=w.model.h[i,a]; w.rhs[i,2]=w.model.c[i,a]
    w.inverse,w.trial_inverse=w.trial_inverse,w.inverse
    w.values,w.trial_values=w.trial_values,w.values
    w.actions[i]-=1
    if Alg==:M
        for j in w.states
            newa=w.trial_cache_gear[j]
            if newa>0 && w.cache_gear[j]!=newa; _performance_cache_row!(w,j,newa); end
        end
        w.f,w.trial_f=w.trial_f,w.f; w.g,w.trial_g=w.trial_g,w.g
        w.cache_gear,w.trial_cache_gear=w.trial_cache_gear,w.cache_gear
    end
    w.counts.rankone_updates+=1
    _performance_frontier!(w)
    return (:accepted,eta)
end


# Conservative support-only unichain check. Retain edges present under EVERY
# admissible action. If every state reaches one root on this common graph,
# every stationary policy has exactly one closed class. No probabilities or
# numerical tolerances are changed, and indexability is not asserted.
function _check_common_reachable_state(m::FiniteMultiGearModel)
    n=nstates(m);common=trues(n,n)
    for i in 1:n, a in admissible_gears(m,i), j in 1:n
        common[i,j] &= m.P[i,j,a+1]>0
    end
    for root in 1:n
        seen=falses(n);seen[root]=true;queue=[root];k=1
        while k<=length(queue)
            j=queue[k];k+=1
            for i in 1:n
                if !seen[i] && common[i,j];seen[i]=true;push!(queue,i);end
            end
        end
        all(seen) && return root
    end
    throw(ArgumentError("No common reachable state on action-independent positive support."))
end
@inline function _performance_feasible(w,f::RowThresholdFamily,i)
    p=f.predecessor[i]
    p==0 || w.actions[p]<w.actions[i]
end
