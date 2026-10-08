# Additive extension; the accepted single-project and allocation kernels are unchanged.
include(joinpath(@__DIR__,"precision_cap_bootstrap.jl"))
if !isdefined(@__MODULE__, :PrecisionCapContinuation)
    include(joinpath(@__DIR__,"precision_cap_continuation_support.jl"))
end
for (name,file) in ((:HeterogeneityModel,"heterogeneity_model.jl"),
                    (:HeterogeneitySimulation,"heterogeneity_simulation.jl"),
                    (:HeterogeneityStudy,"heterogeneity_study.jl"))
    if !isdefined(@__MODULE__,name); include(joinpath(@__DIR__,file)); end
end
