module KDEDistributionsExt

import KernelDensityEstimation
const KDE = KernelDensityEstimation

using Distributions: UnivariateDistribution

# Interface: Convert the support of a distribution to a specific boundary condition
KDE.boundary(spec::UnivariateDistribution) = KDE.boundary(extrema(spec))

# Interface: Expand the distribution to real bounds and a boundary condition
function KDE.bounds(data::AbstractVector{T}, spec::UnivariateDistribution) where {T}
    lo, hi = extrema(spec)
    spec = (isfinite(lo) ? lo : nothing,
            isfinite(hi) ? hi : nothing,
            KDE.boundary((lo, hi)))
    return KDE.bounds(data, spec)
end

end
