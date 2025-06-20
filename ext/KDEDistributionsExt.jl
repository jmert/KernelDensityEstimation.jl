module KDEDistributionsExt

import KernelDensityEstimation: bounds
using Distributions: UnivariateDistribution

# Interface: Convert the support of the distribution to a span and boundary condition
bounds(data, dist::UnivariateDistribution) = bounds(data, (extrema(dist)..., nothing))

end
