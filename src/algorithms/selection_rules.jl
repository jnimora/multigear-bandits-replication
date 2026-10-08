# Shared selection/acceptance semantics. Numerical bands are screens, not
# interval certificates. Resolved nonpositive candidates are excluded; an
# unresolved sign must be resolved by a safeguard or stop the computation.
const _SELECTION_CONTRACT = "positive_work_monotone_v1"
function _positive_frontier_screen(f,g,bands,atol,rtol,guard)
    ratios=fill(eltype(f)(NaN),length(f)); best=0; ambiguous=false
    for j in eachindex(f,g,bands)
        g[j] < -bands[j] && continue
        if g[j]<=bands[j]; ambiguous=true; continue; end
        value=f[j]/g[j]
        isfinite(value) || return (0,ratios,true)
        ratios[j]=value
        (best==0 || value<ratios[best]) && (best=j)
    end
    ambiguous && return (best,ratios,true)
    best==0 && return (0,ratios,false)
    for j in eachindex(ratios)
        if j!=best && isfinite(ratios[j]) && abs(ratios[j]-ratios[best])<=
                _screen_band(abs(ratios[j])+abs(ratios[best]),atol,rtol,guard)
            return (best,ratios,true)
        end
    end
    return (best,ratios,false)
end

# The input order resolves exact ties. No division by a zero/negative work
# marginal is performed, including in exceptional exact-comparison paths.
function _exact_positive_minimum(f,g)
    best=0
    for j in eachindex(f,g)
        g[j]>0 || continue
        (best==0 || f[j]*g[best]<f[best]*g[j]) && (best=j)
    end
    return best
end

function _check_monotonicity_tolerances(atol,rtol)
    isfinite(atol) && isfinite(rtol) && atol>=0 && rtol>=0 ||
        throw(ArgumentError("Invalid MPI monotonicity tolerances."))
    return nothing
end

@inline function _mpi_nondecreasing(previous,current,atol,rtol,
                                    previous_exact=nothing,current_exact=nothing)
    if previous_exact!==nothing && current_exact!==nothing
        return current_exact>=previous_exact
    end
    isfinite(previous) && isfinite(current) || return false
    return current-previous>=-(atol+rtol*max(one(current),abs(previous),abs(current)))
end
