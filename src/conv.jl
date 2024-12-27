using FFTW: plan_rfft, plan_irfft
using LinearAlgebra: mul!

module ConvShape
    @enum T begin
        FULL
        SAME
        VALID
    end
    export FULL, SAME, VALID
end

function conv_range(n, m, shape::ConvShape.T)
    n′ = n + m - 1
    if shape === ConvShape.FULL
        return 1:n′
    elseif shape === ConvShape.SAME
        return cld(m + 1, 2) .+ (0:n-1)
    elseif shape === ConvShape.VALID
        return m:(n′-m+1)
    end
end

struct ConvPlan{R,C,VR<:AbstractVector{R},VC<:AbstractVector{C},FP,RP}
    dims::NTuple{3, Int}
    K̂::VC
    f::VR
    f̂::VC
    fwd::FP
    rev::RP
end

function ConvPlan{R}(n::Integer, m::Integer) where {R<:Real}
    L = nextprod((2, 3, 5, 7), Int(n) + Int(m) - 1)
    dims = (Int(n), Int(m), L)
    return ConvPlan{R}(dims)
end
function ConvPlan{R}(dims::NTuple{3,Int}) where {R<:Real}
    C = complex(R)
    n, m, L = dims
    K̂ = Vector{C}(undef, L ÷ 2 + 1)
    f = Vector{R}(undef, L)
    f̂ = Vector{C}(undef, L ÷ 2 + 1)
    fwd = plan_rfft(f)
    rev = plan_irfft(f̂, L)

    VR = typeof(f)
    VC = typeof(K̂)
    FP = typeof(fwd)
    RP = typeof(rev)
    return ConvPlan{R,C,VR,VC,FP,RP}(dims, K̂, f, f̂, fwd, rev)
end

function replan_conv!(plan::ConvPlan{T}, K::AbstractVector) where {T}
    n, m, L = plan.dims
    length(K) == m || throw(DimensionMismatch())

    # borrow plan.f to calculate the FFT of K
    @view(plan.f[1:m]) .= K
    @view(plan.f[m+1:end]) .= zero(T)
    mul!(plan.K̂, plan.fwd, plan.f)
    return plan
end

function replan_conv(plan::ConvPlan, K::AbstractVector)
    K̂ = similar(plan.K̂)
    f = similar(plan.f)
    f̂ = similar(plan.f̂)
    plan′ = typeof(plan)(plan.dims, K̂, f, f̂, plan.fwd, plan.rev)
    return replan_conv!(plan′, K)
end

function plan_conv(f::AbstractVector{U}, K::AbstractVector{V}) where {U<:Real, V<:Real}
    T = promote_type(float(U), float(V))
    n, m = length(f), length(K)
    plan = ConvPlan{T}(n, m)
    return replan_conv!(plan, K)
end

"""
    conv(f, K, shape::Union{Symbol, ConvShape} = ConvShape.FULL)

Convolves the vectors `f` and `K`, returning the result with one of the following
`shape`s:

- `:full` or `ConvShape.FULL`
- `:same` or `ConvShape.SAME`
- `:valid` or `ConvShape.VALID`
"""
function conv end

function conv(f, K, shape::Symbol)
    if shape === :full
        return conv(f, K, ConvShape.FULL)
    elseif shape === :same
        return conv(f, K, ConvShape.SAME)
    elseif shape === :valid
        return conv(f, K, ConvShape.VALID)
    else
        throw(ArgumentError("Invalid convolution shape, $shape"))
    end
end
conv(f, K) = conv(f, K, ConvShape.FULL)

function conv(f::AbstractVector{S}, K::AbstractVector{T}, shape::ConvShape.T) where {S, T}
    return conv(f, plan_conv(f, K), shape)
end
function conv(f::AbstractVector{S}, plan::ConvPlan{T}, shape::ConvShape.T) where {S, T}
    return conv(T.(f), plan, shape)
end
function conv(f::AbstractVector{T}, plan::ConvPlan{T}, shape::ConvShape.T) where {T}
    n, m, L = plan.dims
    length(f) == n || throw(DimensionMismatch())

    # f̂ = ℱ[f]
    @view(plan.f[1:n]) .= f
    @view(plan.f[n+1:end]) .= zero(T)
    mul!(plan.f̂, plan.fwd, plan.f)

    # g = ℱ¯¹[f̂ ⊙ K̂]
    plan.f̂ .*= plan.K̂
    mul!(plan.f, plan.rev, plan.f̂)
    return plan.f[conv_range(n, m, shape)]
end

