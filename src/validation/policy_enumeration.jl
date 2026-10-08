# Independent oracle: no calls to policy_data, evaluate_discounted,
# evaluate_average, adjacent_marginals, or the downshift implementation.
# It assembles from primitives and uses exact Gauss-Jordan elimination.

"""A closed real charge interval. nothing means -Inf/+Inf; empty marks the empty set."""
struct ChargeInterval
    lower::Union{Nothing,ExactRational}
    upper::Union{Nothing,ExactRational}
    empty::Bool
end
ChargeInterval() = ChargeInterval(nothing,nothing,false)
Base.:(==)(a::ChargeInterval,b::ChargeInterval) =
    a.empty == b.empty && a.lower == b.lower && a.upper == b.upper
Base.isequal(a::ChargeInterval,b::ChargeInterval) = a == b
Base.hash(a::ChargeInterval,h::UInt) = hash((a.lower,a.upper,a.empty),h)
function Base.in(x::Real,I::ChargeInterval)
    I.empty && return false
    y = _exact_number(x)
    return (I.lower === nothing || I.lower <= y) && (I.upper === nothing || y <= I.upper)
end
function Base.show(io::IO,I::ChargeInterval)
    I.empty && return print(io,"empty")
    print(io,I.lower === nothing ? "(-Inf" : "[$(I.lower)",", ",
          I.upper === nothing ? "Inf)" : "$(I.upper)]")
end

struct ExactPolicyRecord
    actions::Vector{Int}
    cost_vector::Vector{ExactRational}  # F or normalized phi
    resource_vector::Vector{ExactRational} # G or normalized gamma
    cost_rate::Union{Nothing,ExactRational}
    resource_rate::Union{Nothing,ExactRational}
    residual_intercept::Matrix{ExactRational}
    residual_slope::Matrix{ExactRational}
    gap_intercept::Matrix{ExactRational} # N_c by A; zero-based gears are NOT columns here
    gap_slope::Matrix{ExactRational}     # -g
    interval::ChargeInterval
end

struct BellmanEnumeration
    model::ExactMultiGearModel
    criterion::Symbol
    beta::Union{Nothing,ExactRational}
    omega::Vector{ExactRational}
    states::Vector{Int}
    records::Vector{ExactPolicyRecord}
    total_policies::BigInt
    complete::Bool
    status::Symbol
    message::String
    coverage::Symbol
    sign_indexability::Symbol
    dai::Union{Nothing,Matrix{ExactRational}}
    witness::Any
end

# Independent exact solver. Intended only for small validation models.
function _oracle_solve(M::Matrix{ExactRational},B::Matrix{ExactRational})
    n = size(M,1)
    size(M,2) == n && size(B,1) == n || throw(DimensionMismatch("Invalid exact system."))
    Z = hcat(copy(M),copy(B))
    for k in 1:n
        offset = findfirst(x -> !iszero(x),@view Z[k:n,k])
        offset === nothing && throw(ArgumentError("Singular exact policy system."))
        pivotrow = k + offset - 1
        if pivotrow != k
            tmp = copy(Z[k,:]); Z[k,:] = Z[pivotrow,:]; Z[pivotrow,:] = tmp
        end
        pivot = Z[k,k]
        Z[k,:] ./= pivot
        for i in 1:n
            if i != k && !iszero(Z[i,k])
                factor = Z[i,k]
                Z[i,:] .-= factor .* Z[k,:]
            end
        end
    end
    X = Z[:,(n+1):end]
    M*X == B || error("Internal exact-solve identity failed.")
    return X
end

# Boolean transitive closure is deliberately different from the production
# evaluator's graph algorithm. Every positive probability is an edge.
function _oracle_reachability(edges::AbstractMatrix{Bool})
    reach = Matrix{Bool}(edges)
    n = size(reach,1)
    for i in 1:n; reach[i,i] = true; end
    for k in 1:n, i in 1:n, j in 1:n
        reach[i,j] = reach[i,j] || (reach[i,k] && reach[k,j])
    end
    return reach
end
function _oracle_closed_class_count(P)
    reach = _oracle_reachability(P .> 0)
    n = size(P,1)
    unseen = trues(n)
    count_closed = 0
    for i in 1:n
        unseen[i] || continue
        group = [j for j in 1:n if reach[i,j] && reach[j,i]]
        unseen[group] .= false
        closed = all(P[u,v] == 0 for u in group for v in 1:n if !(v in group))
        count_closed += closed
    end
    return count_closed
end
function _oracle_communicating(m)
    edges = falses(nstates(m),nstates(m))
    for i in 1:nstates(m), a in admissible_gears(m,i), j in 1:nstates(m)
        edges[i,j] |= m.P[i,j,a+1] > 0
    end
    return all(_oracle_reachability(edges))
end

function _oracle_interval(alpha,rho,m)
    lower = nothing
    upper = nothing
    for i in 1:nstates(m), a in admissible_gears(m,i)
        b,d = alpha[i,a+1],rho[i,a+1]
        if d > 0
            bound = -b/d
            lower = lower === nothing ? bound : max(lower,bound)
        elseif d < 0
            bound = -b/d
            upper = upper === nothing ? bound : min(upper,bound)
        elseif b < 0
            return ChargeInterval(nothing,nothing,true)
        end
    end
    empty = lower !== nothing && upper !== nothing && lower > upper
    return ChargeInterval(lower,upper,empty)
end

function _oracle_record(m,actions,criterion,beta,omega)
    n,A = nstates(m),maxgear(m)
    P = zeros(ExactRational,n,n)
    H,C = zeros(ExactRational,n),zeros(ExactRational,n)
    for i in 1:n
        a = actions[i]+1
        H[i],C[i] = m.h[i,a],m.c[i,a]
        for j in 1:n; P[i,j] = m.P[i,j,a]; end
    end
    hbar = cbar = nothing
    if criterion == :discounted
        X = _oracle_solve(Matrix{ExactRational}(I,n,n)-beta*P,hcat(H,C))
        v,w = X[:,1],X[:,2]
        factor = beta
    else
        _oracle_closed_class_count(P) == 1 || return nothing
        # First solve stationarity, then centered Poisson equations. This is
        # independent of the augmented-system LU in the production evaluator.
        B = Matrix(transpose(P))-Matrix{ExactRational}(I,n,n)
        B[n,:] .= 1
        rhs = zeros(ExactRational,n,1); rhs[n,1] = 1
        nu = _oracle_solve(B,rhs)[:,1]
        all(nu .>= 0) && sum(nu) == 1 && transpose(P)*nu == nu ||
            error("Exact stationary-distribution check failed.")
        hbar,cbar = dot(nu,H),dot(nu,C)
        centered = Matrix{ExactRational}(I,n,n)-P+ones(ExactRational,n)*transpose(nu)
        X = _oracle_solve(centered,hcat(H .- hbar,C .- cbar))
        v,w = X[:,1],X[:,2]
        v .-= dot(omega,v); w .-= dot(omega,w)
        factor = one(ExactRational)
        v .- P*v .+ hbar == H || error("Exact cost Poisson check failed.")
        w .- P*w .+ cbar == C || error("Exact resource Poisson check failed.")
    end
    alpha,rho = zeros(ExactRational,n,A+1),zeros(ExactRational,n,A+1)
    for i in 1:n, a in admissible_gears(m,i)
        continuation_h = sum(m.P[i,j,a+1]*v[j] for j in 1:n)
        continuation_c = sum(m.P[i,j,a+1]*w[j] for j in 1:n)
        alpha[i,a+1] = m.h[i,a+1]+factor*continuation_h-v[i]
        rho[i,a+1] = m.c[i,a+1]+factor*continuation_c-w[i]
        if criterion == :average
            alpha[i,a+1] -= hbar
            rho[i,a+1] -= cbar
        end
    end
    all(alpha[i,actions[i]+1] == 0 && rho[i,actions[i]+1] == 0 for i in 1:n) ||
        error("Selected-action Bellman equality failed in exact arithmetic.")
    states = findall(m.controllable)
    gf,gs = zeros(ExactRational,length(states),A),zeros(ExactRational,length(states),A)
    for (r,i) in enumerate(states), a in 1:A
        gf[r,a] = alpha[i,a]-alpha[i,a+1]
        gs[r,a] = rho[i,a]-rho[i,a+1]
    end
    return ExactPolicyRecord(copy(actions),v,w,hbar,cbar,alpha,rho,gf,gs,
                             _oracle_interval(alpha,rho,m))
end

function _oracle_coverage(records)
    intervals = [r.interval for r in records if !r.interval.empty]
    isempty(intervals) && return false
    # No numerical merging of nearby endpoints.
    sort!(intervals; lt=(x,y) -> x.lower === nothing ? y.lower !== nothing :
                              (y.lower !== nothing && x.lower < y.lower))
    intervals[1].lower === nothing || return false
    upper = intervals[1].upper
    upper === nothing && return true
    for I in intervals[2:end]
        I.lower !== nothing && I.lower > upper && return false
        I.upper === nothing && return true
        upper = max(upper,I.upper)
    end
    return false
end

# Exact sign of an affine gap on a whole optimality interval, including tails.
function _oracle_gap_sign(alpha,slope,I,root)
    I.empty && return true
    for x in (I.lower,I.upper)
        x === nothing && continue
        sign(alpha+slope*x) == sign(root-x) || return false
    end
    if root in I
        alpha+slope*root == 0 || return false
    end
    if I.lower === nothing && I.upper === nothing
        return slope < 0
    elseif I.lower === nothing
        slope <= 0 || return false
        root == I.upper && !(slope < 0) && return false
    elseif I.upper === nothing
        slope <= 0 || return false
        root == I.lower && !(slope < 0) && return false
    end
    return true
end

function _oracle_sign_analysis(m,records)
    states,A = findall(m.controllable),maxgear(m)
    dai = zeros(ExactRational,length(states),A)
    optimal = [r for r in records if !r.interval.empty]
    for (r,i) in enumerate(states), a in 1:A
        roots = Set{ExactRational}()
        for p in optimal
            b,d,I = p.gap_intercept[r,a],p.gap_slope[r,a],p.interval
            if d == 0 && b == 0
                if I.lower !== nothing && I.upper !== nothing && I.lower == I.upper
                    push!(roots,I.lower)
                else
                    return :exact_violation,nothing,
                        (reason=:zero_gap_interval,pair=(i,a),policy=p.actions,interval=I)
                end
            elseif d != 0
                root = -b/d
                root in I && push!(roots,root)
            end
        end
        length(roots) == 1 || return :exact_violation,nothing,
            (reason=:nonunique_finite_crossing,pair=(i,a),roots=sort!(collect(roots)))
        root = only(roots)
        for p in optimal
            _oracle_gap_sign(p.gap_intercept[r,a],p.gap_slope[r,a],p.interval,root) ||
                return :exact_violation,nothing,
                    (reason=:incorrect_gap_sign,pair=(i,a),policy=p.actions,interval=p.interval,root=root)
        end
        dai[r,a] = root
    end
    for (r,i) in enumerate(states), a in 1:(A-1)
        dai[r,a+1] <= dai[r,a] || return :exact_violation,nothing,
            (reason=:unordered_critical_charges,pair=(i,a),lower_gear=dai[r,a],upper_gear=dai[r,a+1])
    end
    return :exact_pass,dai,nothing
end

"""
    enumerate_bellman(model; criterion=:discounted, beta=nothing, omega=nothing,
                      max_policies=10000, max_states=8, time_limit_s=60.0)

Enumerate ALL admissible deterministic stationary policies, independently of
any downshift family. Solve each policy exactly; intersect all-action Bellman
half-lines; check coverage of the ENTIRE real charge line; then derive a DAI
only if every adjacent optimal-Q gap has the required unique sign change and
the critical charges have the manuscript's within-state ordering.

For a floating model, exact binary primitive values must be stochastic. beta
is first converted to the floating model's type, as in direct_downshift. With
ExactMultiGearModel, integer/rational primitives and beta remain exact.

Average enumeration additionally requires communicating support and verifies
unichainness of EVERY enumerated policy; otherwise its status is unsupported,
not nonindexable. A time/policy/state limit gives an incomplete/unresolved
reference and no DAI. The deadline is checked between policy solves; it is not
an interruptible hard limit on one exact-arithmetic solve.
"""
function enumerate_bellman(model::Union{FiniteMultiGearModel,ExactMultiGearModel};
                           criterion::Symbol=:discounted,beta=nothing,omega=nothing,
                           max_policies::Integer=10000,max_states::Integer=8,
                           time_limit_s::Real=60.0)
    criterion in (:discounted,:average) || throw(ArgumentError("Unknown criterion."))
    max_policies >= 1 && max_states >= 1 || throw(ArgumentError("Limits must be positive."))
    isfinite(time_limit_s) && time_limit_s >= 0 || throw(ArgumentError("Invalid time limit."))
    m = model isa ExactMultiGearModel ? model : ExactMultiGearModel(model)
    b = nothing
    if criterion == :discounted
        beta === nothing && throw(ArgumentError("A discount factor is required."))
        b = model isa FiniteMultiGearModel ? _exact_number(eltype(model.h)(beta)) : _exact_number(beta)
        0 < b < 1 || throw(ArgumentError("beta must lie in (0,1)."))
        omega === nothing || throw(ArgumentError("omega applies only to the average criterion."))
    else
        beta === nothing || throw(ArgumentError("Do not supply beta for the average criterion."))
    end
    w = _exact_normalization(nstates(m),omega)
    states = findall(m.controllable)
    total = BigInt(maxgear(m)+1)^length(states)
    records = ExactPolicyRecord[]
    partial(status,msg,witness=nothing) = BellmanEnumeration(m,criterion,b,w,states,records,total,
        false,status,msg,:unresolved,:unresolved,nothing,witness)
    nstates(m) <= max_states || return partial(:state_limit,"Exact enumeration state limit exceeded.")
    total <= max_policies && total <= typemax(Int) || return partial(:policy_limit,"Exact enumeration policy limit exceeded.")
    if criterion == :average && !_oracle_communicating(m)
        return partial(:unsupported_average_model,"The controlled support is not communicating.")
    end
    started = time_ns()
    for number in 0:(Int(total)-1)
        (time_ns()-started)/1e9 >= time_limit_s &&
            return partial(:time_limit,"Enumeration deadline reached; no DAI is asserted.")
        actions = zeros(Int,nstates(m))
        digits = number
        for i in states
            actions[i] = rem(digits,maxgear(m)+1)
            digits = div(digits,maxgear(m)+1)
        end
        record = _oracle_record(m,actions,criterion,b,w)
        record === nothing && return partial(:unsupported_average_model,
            "A stationary policy is multichain; the average reference is unsupported.",
            (reason=:multichain_policy,policy=actions))
        push!(records,record)
    end
    coverage = _oracle_coverage(records) ? :exact_pass : :exact_violation
    if coverage != :exact_pass
        return BellmanEnumeration(m,criterion,b,w,states,records,total,true,:coverage_failure,
            "Exact optimal-policy intervals do not cover the charge line; inspect the implementation.",
            coverage,:unresolved,nothing,nothing)
    end
    sign_status,dai,witness = _oracle_sign_analysis(m,records)
    return BellmanEnumeration(m,criterion,b,w,states,records,total,true,:complete,
        "Enumeration completed using exact rational arithmetic.",coverage,sign_status,dai,witness)
end

"""All Bellman-optimal deterministic policies at a finite real charge."""
function optimal_policies(reference::BellmanEnumeration,lambda::Real)
    reference.complete && reference.coverage == :exact_pass || throw(ArgumentError(
        "The enumeration is incomplete or lacks complete Bellman coverage."))
    x = _exact_number(lambda)
    return [copy(r.actions) for r in reference.records if x in r.interval]
end
