module KDEMakieExt

import KernelDensityEstimation: UnivariateKDE
import Makie: convert_arguments, plottype, Lines, Point2, PointBased, Stairs

plottype(::UnivariateKDE) = Lines

function convert_arguments(::PointBased, K::UnivariateKDE{T}) where {T}
    return (Point2{T}.(K.x, K.f),)
end

function convert_arguments(::Type{<:Stairs}, K::UnivariateKDE{T}) where {T}
    Δx = step(K.x) / 2

    p = Vector{Point2{T}}(undef, length(K.x) + 2)
    # add leading and trailing zeros to close the histogram, compatible with :pre ordering
    p[1] = Point2{T}(K.x[1] - Δx, zero(T))
    p[2:end-1] .= Point2{T}.(K.x .+ Δx, K.f)
    p[end] = Point2{T}(K.x[end] + Δx, zero(T))

    return (p,)
end

end
