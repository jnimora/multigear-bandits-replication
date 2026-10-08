# Synthetic fixtures for validation, not scientific experiment instances.
function one_state_model(; float_type=Float64, row_atol=1e-12)
    return FiniteMultiGearModel(ones(Int,1,1,3), reshape([6,3,1],1,3),
        reshape([0,1,3],1,3); float_type=float_type, row_atol=row_atol)
end

function two_state_model(; float_type=Float64, row_atol=1e-12)
    P = Array{Rational{Int},3}(undef,2,2,3)
    P[:,:,1] = [3//4 1//4; 1//4 3//4]
    P[:,:,2] = [7//8 1//8; 1//2 1//2]
    P[:,:,3] = [15//16 1//16; 3//4 1//4]
    h = [2//1 3//2 1//1; 5//1 4//1 3//1]
    c = [0//1 1//1 2//1; 0//1 3//2 3//1]
    return FiniteMultiGearModel(P,h,c; float_type=float_type, row_atol=row_atol)
end

function transient_model()
    P0 = [0.0 1.0 0.0; 0.0 0.5 0.5; 0.0 0.25 0.75]
    return FiniteMultiGearModel(cat(P0,P0; dims=3), [7.0 NaN;2.0 1.0;4.0 3.0],
        [0.0 NaN;0.0 1.0;0.0 2.0]; controllable=Bool[false,true,true])
end

function periodic_model()
    P0 = [0.0 1.0;1.0 0.0]
    return FiniteMultiGearModel(cat(P0,P0; dims=3), [2.0 1.0;5.0 4.0],
                               [0.0 1.0;0.0 2.0])
end

function multichain_model()
    P0 = [1.0 0.0;0.0 1.0]
    return FiniteMultiGearModel(cat(P0,P0; dims=3), [2.0 1.0;5.0 4.0],
                               [0.0 1.0;0.0 2.0])
end

function nonpositive_marginal_model()
    # Higher gear increases immediate resource, but avoids future resource use.
    P0, P1 = [0.0 1.0;0.0 1.0], [1.0 0.0;0.0 1.0]
    return FiniteMultiGearModel(cat(P0,P1; dims=3), [2.0 1.0;2.0 1.0],
                               [0.0 1.0;2.0 3.0])
end

function random_test_model(seed::Integer; n=4, A=2)
    rng = Xoshiro(seed)
    P = rand(rng,n,n,A+1) .+ 0.125
    for a in 1:(A+1), i in 1:n
        P[i,:,a] ./= sum(@view P[i,:,a])
    end
    h = 3 .* randn(rng,n,A+1)
    c = zeros(n,A+1)
    c[:,1] = rand(rng,n)
    for a in 2:(A+1)
        c[:,a] = c[:,a-1] .+ 0.25 .+ rand(rng,n)
    end
    mask = trues(n)
    isodd(seed) && (mask[1] = false)
    model = FiniteMultiGearModel(P,h,c; controllable=mask)
    actions = [rand(rng,admissible_gears(model,i)) for i in 1:n]
    return model, actions
end
