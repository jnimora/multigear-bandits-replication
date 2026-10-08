# Rebuild exact project solutions and simulate one frozen job without private caches.
using TOML,LinearAlgebra,SHA
include("../heterogeneity_bootstrap.jl")
include("../confirmation_support.jl")
BLAS.set_num_threads(1)
root=normpath(joinpath(@__DIR__,"../.."))
plan=TOML.parsefile(joinpath(root,"configs/heterogeneity_confirmation_native.toml"))
job=only(filter(j->j["id"]==ARGS[1],plan["jobs"]))
out=abspath(ARGS[2]);ispath(out) && error("Output exists; choose a new directory")
mkpath(out)
c=TOML.parsefile(joinpath(root,"configs/rayleigh_heterogeneity.toml"))
e=plan["design"]["execution_requirements"]
c["master_seed"]=plan["design"]["master_seed"];c["replications"]=32
c["resource_limits"]["trajectory_seconds"]=Float64(e["fixed_trajectory_wall_limit_seconds"])
c["resource_limits"]["preparation_seconds"]=Float64(e["whole_family_preparation_seconds"])
spec=only(filter(f->f["id"]==job["family"],plan["families"]))
classes=HeterogeneityModel.ClassData[]
for cl in spec["classes"]
    m=HeterogeneityModel.make_model(HeterogeneityModel.AL.parseq(cl["u"]),Int(cl["H"]))
    data,record=HeterogeneityModel.prepare_class(m,c)
    push!(classes,data)
end
f,dual=HeterogeneityConfirmation.family_from_classes(spec,classes,plan)
HeterogeneityConfirmation.verify_job(f,job,c["master_seed"])
function save_window(row)
    row["transition_seed"]==job["transition_seed"] || error("Transition stream changed")
    row["initial_seed"]==job["initial_seed"] || error("Initial stream changed")
    open(io->TOML.print(io,HeterogeneityStudy.plain(row);sorted=true),joinpath(out,"window_"*row["window"]*".toml"),"w")
end
result=HeterogeneitySimulation.run_windows(f,Int(job["L"]),Int(job["rep"]),job["windows"];
    initial=job["initial"],master_seed=c["master_seed"],limits=HeterogeneityStudy.limits(c),
    seconds=c["resource_limits"]["trajectory_seconds"],retain_records=false,
    collect_between_policies=true,on_window=save_window)
open(io->TOML.print(io,HeterogeneityStudy.plain(result);sorted=true),joinpath(out,"result.toml"),"w")
result["passed"] || error("Trajectory incomplete; do not include as accepted data")
