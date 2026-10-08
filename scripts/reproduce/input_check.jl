using TOML,SHA
kind=ARGS[1];job=TOML.parsefile(ARGS[2])
if kind=="binary"
    include("../binary_benchmark/core.jl")
    x=BinaryBenchmark.generate(job["N"],job["seed"],job["family"])
    for (name,field) in (("P0.f64",:P0),("P1.f64",:P1),("R0.f64",:R0),("R1.f64",:R1))
        bytes2hex(sha256(reinterpret(UInt8,vec(getfield(x,field)))))==job["input_sha256"][name] || error("Input hash mismatch: "*name)
    end
else
    include("../blocked_multigear/frozen_common.jl")
    x=reconstruct(job["primitives"],job["kind"])
    input_identity(job["primitives"],job["kind"],x.input)["passed"] || error("Input hash mismatch")
end
println("Verified input bytes: ",get(job,"case",get(job,"id","input")))
