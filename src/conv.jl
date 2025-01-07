using FFTW: plan_rfft
using LinearAlgebra: mul!

baremodule ConvShape
    import ..Base: @enum
    @enum T begin
        FULL
        SAME
        VALID
    end
    export FULL, SAME, VALID
end

function convaxes(ddims::NTuple{N,<:Integer}, kdims::NTuple{N,<:Integer}, shape::ConvShape.T) where {N}
    return ntuple(Val(N)) do i
        n = ddims[i]
        m = kdims[i]
        L = n + m - 1
        if shape === ConvShape.SAME
            b = cld(m + 1, 2)
            return b:(b + n - 1)
        elseif shape === ConvShape.VALID
            return m:(L - m + 1)
        else
            return 1:L
        end
    end
end

struct ConvPlan{N,R,C,VR<:AbstractArray{R,N},VC<:AbstractArray{C,N},FP,RP}
    f::VR
    f̂::VC
    K̂::VC
    ddims::NTuple{N, Int}
    kdims::NTuple{N, Int}
    fwd::FP
    rev::RP
end

function ConvPlan{R}(ddims::NTuple{N,<:Integer}, kdims::NTuple{N,<:Integer}) where {R<:Real, N}
    cdims = ntuple(Val(N)) do ii
        nextprod((2, 3, 5, 7), Int(ddims[ii]) + Int(kdims[ii]) - 1)
    end
    c1, crest... = cdims

    C = complex(R)
    K̂ = Array{C,N}(undef, c1 ÷ 2 + 1, crest...)
    f̂ = Array{C,N}(undef, c1 ÷ 2 + 1, crest...)
    f = Array{R,N}(undef, c1, crest...)
    fwd = plan_rfft(f)
    rev = inv(fwd)

    VR = typeof(f)
    VC = typeof(f̂)
    FP = typeof(fwd)
    RP = typeof(rev)
    return ConvPlan{N,R,C,VR,VC,FP,RP}(f, f̂, K̂, ddims, kdims, fwd, rev)
end

function convaxes(plan::ConvPlan{N}, shape::ConvShape.T) where {N}
    return convaxes(plan.ddims, plan.kdims, shape)
end

function replan_conv!(plan::ConvPlan{N,T}, K::AbstractArray{S, N}) where {N, T, S}
    @boundscheck begin
        Base.require_one_based_indexing(K)
        size(K) == plan.kdims || throw(DimensionMismatch())
    end
    # borrow plan.f to calculate the FFT of K
    @inbounds for I in CartesianIndices(plan.f)
        inK = all(map(≤, Tuple(I), plan.kdims))
        plan.f[I] = inK ? T(K[I]) : zero(T)
    end
    mul!(plan.K̂, plan.fwd, plan.f)
    return plan
end

function plan_conv(f::AbstractArray{U,N}, K::AbstractArray{V,N}) where {N, U, V}
    T = promote_type(_unitless(U), V)
    plan = ConvPlan{T}(size(f), size(K))
    return replan_conv!(plan, K)
end

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

function conv(f::AbstractArray{S,N}, K::AbstractArray{T,N}, shape::ConvShape.T) where {N, S, T}
    return conv(f, plan_conv(f, K), shape)
end

function conv(f::AbstractArray{S,N}, plan::ConvPlan{N,T}, shape::ConvShape.T) where {N, S, T}
    conv!(plan.f, f, plan)
    I = CartesianIndices(convaxes(plan, shape))
    return _isunitless(S) ? plan.f[I] : (@view(plan.f[I]) .* oneunit(S))
end

function conv!(dest::AbstractArray{T,N}, f::AbstractArray{S,N}, plan::ConvPlan{N,T}) where {N, S, T}
    @boundscheck begin
        Base.require_one_based_indexing(dest, f)
        size(f) == plan.ddims || throw(DimensionMismatch())
        size(dest) == size(plan.f) || throw(DimensionMismatch())
    end

    @inbounds for I in CartesianIndices(plan.f)
        indata = all(map(≤, Tuple(I), plan.ddims))
        plan.f[I] = indata ? T(f[I] * oneunit(_invunit(S))) : zero(T)
    end
    mul!(plan.f̂, plan.fwd, plan.f)

    @inbounds for I in CartesianIndices(plan.f̂)
        plan.f̂[I] *= plan.K̂[I]
    end
    mul!(dest, plan.rev, plan.f̂)
    return dest
end
