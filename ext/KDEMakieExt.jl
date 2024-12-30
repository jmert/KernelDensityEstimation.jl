module KDEMakieExt

import KernelDensityEstimation: UnivariateKDE
import Makie: convert_arguments, plottype, Lines, PointBased, Stairs

plottype(::UnivariateKDE) = Lines

function convert_arguments(trait::PointBased, K::UnivariateKDE)
    return convert_arguments(trait, K.x, K.f)
end

function convert_arguments(trait::Type{<:Stairs}, K::UnivariateKDE)
    R = eltype(K.f)
    Δx = step(K.x) / 2
    x = [K.x[1] - Δx; K.x .+ Δx; K.x[end] + Δx]
    f = [zero(R); K.f; zero(R)]
    return convert_arguments(trait, x, f)
end

end
