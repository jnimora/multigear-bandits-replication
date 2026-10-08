# Run from the repository root: julia --project=. examples/quickstart.jl
using LinearAlgebra
include("../scripts/blocked_multigear/blocked_fp.jl")
using .MultiGearBandits
BLAS.set_num_threads(1)
# Two states, three action levels; replace these arrays and the family for a
# different model. h is the cost rate and c is the resource rate.
P=fill(0.5,2,2,3)
h=[3.0 1.0 0.0;6.0 2.0 0.0]
c=[0.0 1.0 2.0;0.0 1.0 2.0]
model=FiniteMultiGearModel(P,h,c)
input=PerformanceInput(model)
family=UnrestrictedFamily()
for criterion in (:discounted,:average)
    kw=criterion==:discounted ? (;criterion,beta=0.99) : (;criterion)
    w=BlockedMultiGearFP.prepare(input;family,kw...,blocksize=32)
    BlockedMultiGearFP.run!(w)
    s=BlockedMultiGearFP.snapshot(w)
    @assert s.complete
    println(criterion, ": ", s.mpi)
end
