# Small illustrative calculation; not a replacement for a production case.
using TOML,LinearAlgebra
include("../scripts/heterogeneity_bootstrap.jl")
BLAS.set_num_threads(1)
const HM=HeterogeneityModel
c=TOML.parsefile(joinpath(@__DIR__,"../configs/rayleigh_heterogeneity.toml"))
model=HM.make_model(BigInt(1)//BigInt(8),8)
data,certificate=HM.prepare_class(model,c)
f,dual=HM.from_classes("example","example_stream",1,[data,data],
    [BigInt(1)//1,BigInt(1)//1],[25,25],BigInt(21)//128;max_population=50)
windows=[Dict("name"=>"example","burn"=>2,"T"=>8)]
r=HeterogeneitySimulation.run_windows(f,50,1,windows;master_seed=2026092502,
    limits=HeterogeneityStudy.limits(c),seconds=60.0)
@assert r["passed"] && length(r["records"])==1
println("Small exact-model allocation example passed; all six policies completed.")
