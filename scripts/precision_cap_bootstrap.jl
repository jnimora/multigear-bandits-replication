# Idempotent include route for setup, repository tests, and the native runner.
if !isdefined(@__MODULE__,:MultiGearBandits)
    include(joinpath(@__DIR__,"..","src","MultiGearBandits.jl"))
end
using .MultiGearBandits
for (name,file) in ((:RayleighAoII,"rayleigh_aoii_support.jl"),(:RayleighBlock,"rayleigh_block_support.jl"),
    (:RayleighTiming,"rayleigh_timing_support.jl"),(:RayleighLargeCap,"rayleigh_largecap_support.jl"),
    (:RayleighAllocation,"rayleigh_allocation_support.jl"),(:ManyProjectAllocator,"manyproject_allocator.jl"),
    (:ManyProjectModel,"manyproject_model.jl"),(:ManyProjectSimulation,"manyproject_simulation.jl"),
    (:PrecisionCapScores,"precision_cap_scores.jl"),(:PrecisionCapModel,"precision_cap_model.jl"),
    (:PrecisionCapSimulation,"precision_cap_simulation.jl"),(:PrecisionCapStudy,"precision_cap_study.jl"))
    if !isdefined(@__MODULE__,name);include(joinpath(@__DIR__,file));end
end
