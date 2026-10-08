function downshift_case(name::AbstractString; exact::Bool=false,float_type=Float64)
    data = TOML.parsefile(joinpath(@__DIR__,"downshift_cases.toml"))[name]
    weights = data["transition_weights"]
    ng,n = length(weights),length(weights[1])
    denominator = Int(data["denominator"])
    P = Array{Rational{Int},3}(undef,n,n,ng)
    h,c = Matrix{Rational{Int}}(undef,n,ng),Matrix{Rational{Int}}(undef,n,ng)
    for a in 1:ng, i in 1:n
        h[i,a],c[i,a] = Int(data["costs"][i][a])//1,Int(data["resources"][i][a])//1
        for j in 1:n; P[i,j,a] = Int(weights[a][i][j])//denominator; end
    end
    mask = Bool.(get(data,"controllable",trues(n)))
    return exact ? ExactMultiGearModel(P,h,c; controllable=mask) :
        FiniteMultiGearModel(P,h,c; controllable=mask,float_type=float_type)
end

function close_unequal_model()
    P0 = [0.5 0.5;0.5 0.5]
    return FiniteMultiGearModel(cat(P0,P0; dims=3),[1.0 0.0;(1.0-2.0^(-40)) 0.0],
                               [0.0 1.0;0.0 1.0])
end

# Random EXACTLY stochastic dyadic primitives; no floating row normalization.
function dyadic_validation_model(seed::Integer; n=3,A=2)
    rng = Xoshiro(seed)
    P = zeros(n,n,A+1)
    for a in 1:(A+1), i in 1:n
        cuts = sort!(randperm(rng,15)[1:(n-1)])
        weights = diff(vcat(0,cuts,16))
        P[i,:,a] = weights ./ 16
    end
    h = Float64.(rand(rng,0:12,n,A+1))
    c = zeros(n,A+1)
    for a in 2:(A+1)
        c[:,a] = c[:,a-1]+rand(rng,1:4,n)
    end
    return FiniteMultiGearModel(P,h,c)
end
