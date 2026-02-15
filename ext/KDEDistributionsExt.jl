module KDEDistributionsExt

import KernelDensityEstimation: bounds
using Distributions: UnivariateDistribution

# Interface: Convert the support of the distribution to a span and boundary condition
function bounds(data::Tuple{AbstractVector}, dist::UnivariateDistribution)
    return bounds(data, (dist,))
end
function bounds(data::Tuple{Vararg{AbstractVector,N}}, dists::Tuple{Vararg{UnivariateDistribution,N}}) where {N}
    return bounds(data, ntuple(i -> (extrema(dists[i])..., nothing), Val(N)))
end

end
