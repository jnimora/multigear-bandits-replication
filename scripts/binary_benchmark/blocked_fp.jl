module BlockedBinaryFP

using LinearAlgebra
# Reuse the reference initializer and numerical admission rules unchanged.
if !isdefined(parentmodule(@__MODULE__), :BinaryBenchmark)
    Base.include(parentmodule(@__MODULE__), joinpath(get(ENV, "BINARY_BENCH_REPO",
        normpath(joinpath(@__DIR__, "../.."))), "scripts/binary_benchmark/core.jl"))
end
const BB = getfield(parentmodule(@__MODULE__), :BinaryBenchmark)

"""Recommended unit-work binary downshift FP with deferred Schur updates.

For the m remaining packed coordinates the current tableau is
    X_current = base.X[1:m,1:m] - U[1:m,1:q] * V[1:m,1:q]'.
The reward/work vectors are ALWAYS current, not deferred. Every pivot uses
the same positive-work selection and MPI monotonicity screen as reference FP.
Only the matrix update is batched; matrix multiplication is ordinary BLAS.
Defaults (blocksize=32, tail=64) match the validated production campaign.
Scope: unit active resource, zero passive resource, unrestricted candidates.
"""
mutable struct Workspace
    base::BB.FP
    U::Matrix{Float64}
    V::Matrix{Float64}
    row::Vector{Float64}
    q::Int
    blocksize::Int
    tail::Int
    flushes::Int
    deferred_pivots::Int
end

function wrap(w::BB.FP; blocksize::Int=32, tail::Int=64)
    w.up && throw(ArgumentError("Only binary downshift FP is supported"))
    w.k == 0 && w.status == :ready || throw(ArgumentError("Expected a fresh reference workspace"))
    blocksize > 0 || throw(ArgumentError("blocksize must be positive"))
    tail >= 0 || throw(ArgumentError("tail must be nonnegative"))
    n = length(w.f); b = min(blocksize, max(n-1, 1))
    Workspace(w, zeros(n,b), zeros(n,b), zeros(n), 0, b, tail, 0, 0)
end

prepare(x, beta; kwargs...) = wrap(BB.prepare_fp(x,beta,false); kwargs...)

function flush!(w::Workspace)
    m = w.base.m; q = w.q
    q == 0 && return w
    # Views retain the parent's leading dimension. No dense temporary or packing
    # copy is allocated by Julia; the BLAS implementation owns its packing.
    mul!(@view(w.base.X[1:m,1:m]), @view(w.U[1:m,1:q]),
         transpose(@view(w.V[1:m,1:q])), -1.0, 1.0)
    w.q = 0; w.flushes += 1
    w
end

function pivot!(w::Workspace, p::Int, c=BB.Checks())
    s = w.base; m = s.m; q = w.q
    if m <= w.tail
        flush!(w)
        return BB.pivot!(s,p,c)
    end
    eta = s.X[p,p]
    @inbounds for l in 1:q; eta -= w.U[p,l]*w.V[p,l]; end
    isfinite(eta) && eta > c.pivot_tol || return :unsafe_pivot
    # Permute both the stored base and every pending factor. The selected
    # coordinate moves to m, exactly as in the eager reference implementation.
    if p != m
        @inbounds for j in 1:m; s.X[p,j],s.X[m,j] = s.X[m,j],s.X[p,j]; end
        @inbounds for i in 1:m; s.X[i,p],s.X[i,m] = s.X[i,m],s.X[i,p]; end
        @inbounds for l in 1:q
            w.U[p,l],w.U[m,l] = w.U[m,l],w.U[p,l]
            w.V[p,l],w.V[m,l] = w.V[m,l],w.V[p,l]
        end
        s.f[p],s.f[m] = s.f[m],s.f[p]
        s.g[p],s.g[m] = s.g[m],s.g[p]
        s.ids[p],s.ids[m] = s.ids[m],s.ids[p]
    end
    r = m-1
    @inbounds for i in 1:r
        s.column[i] = s.X[i,m]; w.row[i] = s.X[m,i]
    end
    # Materialize just the selected row/column. Factors are stored by columns
    # to stream contiguous memory in each short-rank correction.
    @inbounds for l in 1:q
        vl = w.V[m,l]; ul = w.U[m,l]
        @simd for i in 1:r
            s.column[i] -= w.U[i,l]*vl
            w.row[i] -= w.V[i,l]*ul
        end
    end
    fd = s.f[m]/eta; gd = s.g[m]/eta
    @inbounds for i in 1:r
        col = s.column[i]; z = w.row[i]/eta
        isfinite(col) && isfinite(z) || return :nonfinite
        s.f[i] -= fd*col; s.g[i] -= gd*col
        w.U[i,q+1] = col; w.V[i,q+1] = z
    end
    s.m = r; w.q = q+1; w.deferred_pivots += 1
    w.q == w.blocksize && flush!(w)
    :ok
end

function run!(w::Workspace, c=BB.Checks())
    s = w.base; n = length(s.indices)
    while s.k < n
        p,v,st,rej,mw = BB.select_positive(s.f,s.g,s.ids,s.m,false,c)
        s.rejected += rej; s.minimum_work = min(s.minimum_work,mw)
        st == :ok || (s.status=st; return w)
        if s.k > 0
            old = s.sequence[s.k]; s.minimum_gap = min(s.minimum_gap,v-old)
            BB.mono_ok(old,v,false,c) || (s.status=:nonmonotone; return w)
        end
        k = s.k+1; id = s.ids[p]
        s.indices[id] = v; s.order[k] = id; s.sequence[k] = v; s.k = k
        if k < n
            st = pivot!(w,p,c)
            st == :ok || (s.status=st; return w)
        end
    end
    # As in reference FP, the final unneeded tableau update is skipped.
    s.status = :complete
    w
end

result(w::Workspace) = merge(BB.result(w.base), Dict(
    "blocksize"=>w.blocksize, "tail"=>w.tail,
    "flushes"=>w.flushes, "deferred_pivots"=>w.deferred_pivots))

end
