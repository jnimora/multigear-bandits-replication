const ExactRational = Rational{BigInt}
_exact_number(x::Integer) = BigInt(x)//BigInt(1)
_exact_number(x::Rational) = BigInt(numerator(x))//BigInt(denominator(x))
function _exact_number(x::AbstractFloat)
    isfinite(x) || throw(ArgumentError("Exact conversion requires finite inputs."))
    # Zero tolerance preserves the stored binary number, not a nearby fraction.
    return rationalize(BigInt,x; tol=zero(x))
end

"""
    ExactMultiGearModel(P, h, c; controllable=trues(size(h,1)))
    ExactMultiGearModel(floating_model)

Exact rational primitives for SMALL validation problems, with the same layout
and gear conventions as FiniteMultiGearModel. Every admissible transition row
must sum to EXACTLY one. Inputs are never normalized or rounded to convenient
fractions. A floating model's entries are converted to their exact binary
rational values. Consequently a row accepted by its floating row-sum tolerance
can be rejected here. Construct from integer/rational primitives to retain an
intended non-dyadic rational model; that model need not equal its Float64 copy.
"""
struct ExactMultiGearModel
    P::Array{ExactRational,3}
    h::Matrix{ExactRational}
    c::Matrix{ExactRational}
    controllable::BitVector
    function ExactMultiGearModel(P::AbstractArray{<:Real,3},
                                h::AbstractMatrix{<:Real}, c::AbstractMatrix{<:Real};
                                controllable::AbstractVector{Bool}=trues(size(h,1)))
        Base.require_one_based_indexing(P,h,c,controllable)
        n, ng = size(h)
        n >= 1 && ng >= 2 || throw(ArgumentError("At least one state and two gears are required."))
        size(c) == (n,ng) && size(P) == (n,n,ng) || throw(DimensionMismatch("Invalid primitives."))
        length(controllable) == n && any(controllable) || throw(ArgumentError("Invalid controllable mask."))
        pp = Array{ExactRational,3}(undef,n,n,ng)
        hh, cc = Matrix{ExactRational}(undef,n,ng), Matrix{ExactRational}(undef,n,ng)
        for i in 1:n, slot in 1:ng
            source = controllable[i] ? slot : 1
            hh[i,slot], cc[i,slot] = _exact_number(h[i,source]), _exact_number(c[i,source])
            cc[i,slot] >= 0 || throw(ArgumentError("Negative resource."))
            for j in 1:n
                pp[i,j,slot] = _exact_number(P[i,j,source])
                pp[i,j,slot] >= 0 || throw(ArgumentError("Negative probability."))
            end
            sum(@view pp[i,:,slot]) == 1 || throw(ArgumentError(
                "State $i, gear $(source-1): stored probabilities do not sum EXACTLY to one. " *
                "No normalization was applied; the exact oracle cannot certify this input."))
            if controllable[i] && slot > 1
                cc[i,slot] > cc[i,slot-1] || throw(ArgumentError("Resources must strictly increase."))
            end
        end
        new(pp,hh,cc,BitVector(controllable))
    end
end
ExactMultiGearModel(m::FiniteMultiGearModel) =
    ExactMultiGearModel(m.P,m.h,m.c; controllable=m.controllable)
nstates(m::ExactMultiGearModel) = size(m.h,1)
maxgear(m::ExactMultiGearModel) = size(m.h,2)-1
ncontrollable(m::ExactMultiGearModel) = count(m.controllable)
function admissible_gears(m::ExactMultiGearModel,i::Integer)
    1 <= i <= nstates(m) || throw(BoundsError(m.controllable,i))
    return 0:(m.controllable[i] ? maxgear(m) : 0)
end

function _exact_normalization(n,omega)
    w = zeros(ExactRational,n)
    if omega === nothing
        w[1] = 1
    else
        length(omega) == n || throw(DimensionMismatch("Invalid normalization length."))
        w .= _exact_number.(omega)
        sum(w) == 1 || throw(ArgumentError("Exact normalization must sum EXACTLY to one."))
    end
    return w
end
