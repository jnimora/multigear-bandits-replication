"""
Finite-state multi-gear model and direct policy-evaluation reference routines.

This module minimizes holding cost plus a resource charge: h + lambda*c.
It includes direct policy evaluation, a downshift adaptive-greedy baseline,
full-inverse rank-one and frontier-cached baselines, a preallocated reduced FP
implementation, and an independent exact small-model Bellman reference. An MPI computation
is distinct from DAI verification. Loading the module changes no thread
settings, installs no packages, and writes no files.
"""
module MultiGearBandits

using LinearAlgebra
using Random

export FiniteMultiGearModel, nstates, maxgear, ncontrollable, admissible_gears,
       PolicyData, policy_data, DiscountedEvaluation, AverageEvaluation,
       SolveDiagnostics, evaluate_discounted, evaluate_average,
       charged_value, charged_rate, charged_bias, policy_qvalues,
       MarginalMetrics, adjacent_marginals,
       AbstractPolicyFamily, UnrestrictedFamily, OrderedThresholdFamily, RowThresholdFamily,
       ExplicitPolicyFamily, policy_in_family, downshift_frontier,
       ExactMultiGearModel, ChargeInterval, ExactPolicyRecord, BellmanEnumeration,
       enumerate_bellman, optimal_policies, DownshiftStep, DirectDownshiftResult,
       direct_downshift, DownshiftValidation, validate_downshift,
       RankOnePolicyState, RankOneUpdateReport, initialize_rankone,
       rankone_evaluation, update_downshift!, RankOneAuditRecord, audit_rankone,
       RankOneDownshiftResult, rankone_downshift,
       FrontierMetricCache, MarginalCachePivot, MarginalCacheRecord,
       initialize_marginal_cache, prepare_marginal_pivot, refresh_marginal_cache,
       MarginalCacheAuditRecord, audit_marginal_cache, CachedDownshiftResult, cached_downshift,
       FPWorkCounts, FPPivotReport, ReducedFPWorkspace, initialize_reduced_fp,
       pivot_reduced_fp!, rebuild_reduced_fp!, FPAuditRecord, audit_reduced_fp,
       reduced_fp_snapshot, recover_reduced_values, run_reduced_fp!,
       ReducedFPResult, reduced_fp_downshift

include("models/finite_model.jl")
include("evaluation/direct_policy.jl")
include("evaluation/marginals.jl")

include("models/exact_model.jl")
include("models/policy_families.jl")
include("validation/policy_enumeration.jl")
include("algorithms/selection_rules.jl")
include("algorithms/direct_downshift.jl")
include("evaluation/rankone_policy.jl")
include("validation/rankone_audit.jl")
include("algorithms/rankone_downshift.jl")
include("evaluation/cached_marginals.jl")
include("validation/cache_audit.jl")
include("algorithms/cached_downshift.jl")
include("validation/downshift_check.jl")

include("evaluation/reduced_fp.jl")
include("validation/reduced_fp_audit.jl")
include("algorithms/reduced_fp_downshift.jl")
include("validation/reduced_fp_reference.jl")

# Comparable compact-output engineering-pilot execution paths.
include("performance/workspaces.jl")
include("performance/preparation.jl")
include("performance/residual_comparison.jl")
include("performance/contraction_comparison.jl")
include("performance/driver.jl")
include("performance/audit.jl")
include("performance/generators.jl")
export PerformanceOptions, PerformanceInput, PerformanceRun, prepare_performance_run,
       step_performance!, run_performance!, performance_snapshot, performance_counts,
       performance_agreement, audit_performance_state, audit_performance_run,
       performance_stream_seed, performance_primitives, performance_model,
       PreparationTrace, initialize_reduced_fp_optimized, audit_fp_initialization,
       performance_mpi_screen, residual_certified_selection

end # module
