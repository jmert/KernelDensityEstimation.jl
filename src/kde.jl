import Logging: @warn

abstract type AbstractKDE{T} end
abstract type AbstractKDEInfo{T} end
abstract type AbstractKDEMethod end
abstract type BandwidthEstimator end

struct UnivariateKDEInfo{T} <: AbstractKDEInfo{T}
    npoints::Int
    bandwidth::T
end

struct UnivariateKDE{T,R<:AbstractRange{T},V<:AbstractVector{T}} <: AbstractKDE{T}
    x::R
    f::V
end

# define basic iteration syntax to destructure the contents of a UnivariateKDE;
#   note that property destructuring syntax should be preferred, but that is not available
#   until Julia 1.8, so include this for more convenient use in older Julia versions
Base.iterate(estim::UnivariateKDE) = iterate(estim, Val(:x))
Base.iterate(estim::UnivariateKDE, ::Val{:x}) = (estim.x, Val(:f))
Base.iterate(estim::UnivariateKDE, ::Val{:f}) = (estim.f, nothing)
Base.iterate(estim::UnivariateKDE, ::Any) = nothing

@noinline _warn_unused(kwargs) = @warn "Unused keyword argument(s)" kwargs=kwargs
warn_unused(kwargs) = length(kwargs) > 0 ? _warn_unused(kwargs) : nothing

between_closed(lo, hi) = ≥(lo) ∘ ≤(hi)
_filter(data, lo, hi) = Iterators.filter(between_closed(lo, hi), data)

function _extrema(data, lo, hi)
    T = eltype(data)
    !isnothing(lo) && !isnothing(hi) && return (T(lo), T(hi))
    a = b = first(data)
    for x in data
        a = min(a, x)
        b = max(b, x)
    end
    return (isnothing(lo) ? a : T(lo),
            isnothing(hi) ? b : T(hi))
end

function _count_var(data)
    T = eltype(data)
    ν = 0
    μ = μ₋₁ = σ² = zero(T)
    for x in data
        ν += 1
        w = one(T) / ν
        μ₋₁, μ = μ, (one(T) - w) * μ + w * x
        σ² = (one(T) - w) * σ² + w * (x - μ) * (x - μ₋₁)
    end
    return (; count = ν, var = σ²)
end

abstract type AbstractBinningKDE <: AbstractKDEMethod end

"""
    struct HistogramBinning <: AbstractBinningKDE end

Base case which generates a density estimate by simply generating a histogram of the data.
"""
struct HistogramBinning <: AbstractBinningKDE end

#"""
#    struct LinearBinning <: AbstractBinningKDE end
#
#Base case which generates a density estimate by linear binning.
#
#See also [`HistogramBinning`](@ref)
#"""
#struct LinearBinning <: AbstractBinningKDE end

function _kdebin(::HistogramBinning, data, lo, hi, Δx, nbins)
    T = eltype(data)
    ν = 0
    f = zeros(T, nbins)
    for x in data
        # N.B. ii is a 0-index offset
        ii = floor(Int, (x - lo) / Δx)
        if ii == nbins && x == hi
            # top bin is closed rather than half-open
            ii -= 1
        end
        (ii < 0 || ii >= nbins) && continue
        f[ii + 1] += one(T)
        ν += 1
    end
    for ii in eachindex(f)
        f[ii] /= ν
    end
    return ν, f
end

function kde(method::AbstractBinningKDE, data;
        lo = nothing, hi = nothing, nbins = nothing, bwratio = 1, kwargs...)
    warn_unused(kwargs)
    T = float(eltype(data))

    # determine lower and upper limits of the histogram
    lo′, hi′ = _extrema(data, lo, hi)
    # filter the data to be only in-bounds
    v = _filter(data, lo′, hi′)

    bw = bandwidth(SilvermanBandwidth(), v)
    if isnothing(nbins)
        nbins′ = max(1, round(Int, bwratio * (hi′ - lo′) / bw))
    else
        nbins′ = Int(nbins)
        nbins′ > 0 || throw(ArgumentError("nbins must be a positive integer"))
    end

    edges = range(lo′, hi′, length = nbins′ + 1)
    Δx = hi′ > lo′ ? step(edges) : one(lo′)  # 1 bin if histogram has zero width
    centers = edges[2:end] .- Δx / 2

    ν, f = _kdebin(method, v, lo′, hi′, Δx, nbins′)
    estim = UnivariateKDE(centers, f)
    info = UnivariateKDEInfo(ν, bw)
    return estim, info
end


struct SilvermanBandwidth <: BandwidthEstimator end
function bandwidth(::SilvermanBandwidth, v)
    # Estimate a nominal bandwidth using Silverman's Rule.
    #
    # From Hansen (2009) — https://users.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf
    # for a Gaussian kernel:
    # - Table 1:
    #   - R(k) = 1 / 2√π
    #   - κ₂(k) = 1
    # - Section 2.9, letting ν = 2:
    #   - bw = σ̂ n^(-1/5) C₂(k)
    #     C₂(k) = 2 ( 8R(k)√π / 96κ₂² )^(1/5) == (4/3)^(1/5)
    T = float(eltype(v))
    n, σ² = _count_var(v)
    bw = ifelse(iszero(σ²), eps(one(T)), sqrt(σ²) * (T(4 // 3) / n)^(one(T) / 5))
    return bw
end


#struct ISJBandwidth <: BandwidthEstimator end
#function bandwidth(::ISJBandwidth, v)
#  # not yet implemented
#end


"""
    x, f = kde(v; lo = nothing, hi = nothing, nbins = nothing,
               bandwidth = nothing, bwratio = 8,
               opt_normalize = true
              )

Calculate a discrete kernel density estimate (KDE) `f(x)` of the sample distribution of `v`.

The KDE is constructed by first histogramming the input `v` into `nbins` bins with
outermost bin edges spanning `lo` to `hi`, which default to the minimum and maximum of
`v`, respectively, if not provided. The histogram is then convolved with a Gaussian
distribution with standard deviation `bandwidth`. The `bwratio` parameter is used to
calculate `nbins` when it is not given and corresponds to the ratio of the bandwidth to
the width of each histogram bin.

This simple KDE can then be post-processed in a number of ways to improve the returned
density estimate:

- `opt_normalize` — If `true`, the KDE is renormalized to account for the convolutions
  interaction with implicit zeros beyond the bounds of the low and high edges of the
  histogram.

- `opt_linearboundary` — If `true`, corrections are applied to account for non-zero
  slope at the edges of the KDE. Note that when this correction is enabled,
  `opt_normalize` is effectively ignored since overall normalization is an implicit part
  of this correction.

- `opt_multiply` —

# Extended help

- A truncated Gaussian smoothing kernel is assumed. The Gaussian is truncated at ``4σ``.
- The linear boundary correction is explained in Refs. Lewis (2019) and Jones (1996).

## References:

- M. C. Jones and P. J. Foster. “A simple nonnegative boundary correction method for kernel
  density estimation”. In: Statistica Sinica 6 (1996), pp. 1005–1013.

- A. Lewis. "GetDist: a Python package for analysing Monte Carlo samples".
  In: _arXiv e-prints_ (Oct. 2019).
  arXiv: [1910.13970 \\[astro-ph.IM\\]](https://arxiv.org/abs/1910.13970)
"""
function kde(data;
             lo = nothing, hi = nothing, nbins = nothing,
             bandwidth = nothing, bwratio = 8,
             opt_normalize = true, opt_linearboundary = true, opt_multiply = true
            )
    estim, info = kde(HistogramBinning(), data; lo, hi, nbins, bwratio)
    counts = estim.f
    Δx = step(estim.x)
    bw = isnothing(bandwidth) ? info.bandwidth : bandwidth

    # make sure the kernel axis is centered on zero
    nn = ceil(Int, 4bw / Δx)
    xx = range(-nn * Δx, nn * Δx, step = Δx)

    function _kde(counts)
        # construct the convolution kernel
        # N.B. Mathematically normalizing the kernel, such as with
        #        kernel = exp.(-(xx ./ bw) .^ 2 ./ 2) .* (Δx / bw / sqrt(2T(π)))
        #      breaks down when bw << Δx. Instead of trying to work around that, just take the
        #      easy route and just post-normalize a simpler calculation.
        kernel = exp.(-(xx ./ bw) .^ 2 ./ 2)
        kernel ./= sum(kernel)

        # convolve the data with the kernel to construct a density estimate
        K̂ = plan_conv(counts, kernel)
        f₀ = conv(counts, K̂, :same)

        if opt_normalize || opt_linearboundary
            # normalize the estimate, which accounts for implicit zeros outside the histogram
            # bounds
            Θ = fill!(similar(counts), true)
            μ₀ = conv(Θ, K̂, :same)
            f₀ ./= μ₀
        end
        if opt_linearboundary
            # apply a linear boundary correction
            # see Eqn 12 & 16 of Lewis (2019)
            #   N.B. the denominator of A₀ should have [W₂]⁻¹ instead of W₂
            kernel .*= xx
            replan_conv!(K̂, kernel)
            μ₁ = conv(Θ, K̂, :same)
            f′ = conv(counts, K̂, :same)

            kernel .*= xx
            replan_conv!(K̂, kernel)
            μ₂ = conv(Θ, K̂, :same)
            μ₀₂ = (μ₂ .*= μ₀)

            # function to force f̂ to be positive
            # see Eqn. 17 of Lewis (2019)
            pos(f₁, f₂) = iszero(f₁) ? zero(f₁) : f₁ * exp(f₂ / f₁ - one(f₁))
            f̂ = pos.(f₀, (μ₀₂ .* f₀ .- μ₁ .* f′) ./ (μ₀₂ .- μ₁.^2))
        else
            f̂ = f₀
        end
        return f̂
    end

    f̂ = _kde(counts)
    if opt_multiply
        nonzero(x) = iszero(x) ? one(x) : x
        g = nonzero.(f̂)
        f̂ = _kde(counts ./ g)
        f̂ .*= g
    end

    return UnivariateKDE(estim.x, f̂)
end
