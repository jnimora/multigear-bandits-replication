using Test,LinearAlgebra,Random,TOML
BLAS.set_num_threads(1)
include("../src/MultiGearBandits.jl")
using .MultiGearBandits
include("fixtures/reference_models.jl")
include("fixtures/downshift_models.jl")
for file in ("reference","downshift","rankone","cached","reduced_fp","performance",
             "selection_contract","preparation","efficiency","fp_reference_optimization",
             "residual_comparison","contraction_comparison")
    @testset "$file" begin
        include(file*"_tests.jl")
    end
end
# Separate modules keep standalone test constants and fixtures independent.
module BinaryChecks
include("blocked_binary_fp_tests.jl")
end
module BlockedChecks
include("blocked_multigear_tests.jl")
end
