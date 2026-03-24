module KDEDistributionsExt

import KernelDensityEstimation: bounds
using Distributions: UnivariateDistribution, MultivariateDistribution

# Univariate distribution span and boundary conditions
function bounds(data::Tuple{AbstractVector}, dist::UnivariateDistribution)
    spec = (extrema(dist)..., nothing)
    return bounds(data, spec)
end

# Match 1D special case in base package
function bounds(data::AbstractVector, dist::UnivariateDistribution)
    return bounds((data,), dist)
end

# Bivariate distribution
function bounds(data::Tuple{AbstractVector,AbstractVector},
                dist::MultivariateDistribution)
    length(dist) == length(data) || throw(DimensionMismatch())
    rect = extrema(dist)
    spec = map(lh -> (lh..., nothing), zip(rect...))
    return bounds(data, (spec...,))
end

# Multivariate distributions from product of univariate distributions
function bounds(data::Tuple{Vararg{AbstractVector,N}}, dists::Tuple{Vararg{UnivariateDistribution,N}}) where {N}
    return bounds(data, ntuple(i -> (extrema(dists[i])..., nothing), Val(N)))
end

end
