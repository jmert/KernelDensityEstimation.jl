module KDEMakieExt

import KernelDensityEstimation: UnivariateKDE, BivariateKDE
import Makie: Makie, convert_arguments, plottype

plottype(::UnivariateKDE) = Makie.Lines

# suports conversion for Lines
function convert_arguments(trait::Makie.PointBased, K::UnivariateKDE)
    return convert_arguments(trait, K.x, K.f)
end

# override for Stairs
function convert_arguments(trait::Type{<:Makie.Stairs}, K::UnivariateKDE)
    R = eltype(K.f)
    Δx = step(K.x) / 2
    x = [K.x[1] - Δx; K.x .+ Δx; K.x[end] + Δx]
    f = [zero(R); K.f; zero(R)]
    return convert_arguments(trait, x, f)
end


plottype(::BivariateKDE) = Makie.Contour

# supports conversion for Contour
function convert_arguments(trait::Makie.VertexGrid, K::BivariateKDE)
    return convert_arguments(trait, K.x, K.y, K.f)
end

# supports conversion for Heatmap
function convert_arguments(trait::Makie.CellGrid, K::BivariateKDE)
    return convert_arguments(trait, K.x, K.y, K.f)
end

end
