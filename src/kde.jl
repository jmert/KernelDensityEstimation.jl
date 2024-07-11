import Logging: @warn

"""
```julia
@enum T Closed Open ClosedLeft ClosedRight
const OpenLeft = ClosedRight
const OpenRight = ClosedLeft
```

Enumeration to describe the desired boundary conditions of the domain of the kernel
density estimate ``K``.
For some given data ``d ∈ [a, b]``, the cover conditions have the following impact:

- `Closed`: The domain ``K ∈ [a, b]`` is used directly as the bounds of the binning.
- `Open`: The desired domain ``K ∈ (-∞, +∞)`` is effectively achieved by widening the
  bounds of the data by the size of the finite convolution kernel.
  Specifically, the binning is defined over the range ``[a - 4σ, b + 4σ]`` where ``σ``
  is the bandwidth of the Gaussian convolution kernel.
- `ClosedLeft`: The left half-closed interval ``K ∈ [a, +∞)`` is used as the bounds for
  binning by adjusting the upper limit to the range ``[a, b + 4σ]``.
  The equivalent alias `OpenRight` may also be used.
- `ClosedRight`: The right half-closed interval ``K ∈ (-∞, b]`` is used as the bounds for
  binning by adjusting the lower limit to the range ``[a - 4σ, b]``.
  The equivalent alias `OpenLeft` may also be used.
"""
baremodule Cover
    import ..Base.@enum

    export Closed, Open, ClosedLeft, ClosedRight, OpenLeft, OpenRight, to_cover

    @enum T Closed Open ClosedLeft ClosedRight
    const OpenLeft = ClosedRight
    const OpenRight = ClosedLeft

    to_cover(x::Cover.T) = x
    to_cover(x::Symbol) = x === :open ? Open :
                          x === :closed ? Closed :
                          x === :closedleft ? ClosedLeft :
                          x === :openright ? ClosedLeft :
                          x === :closedright ? ClosedRight :
                          x === :openleft ? ClosedRight :
                          throw(ArgumentError("Unknown cover option: $x"))
end
using .Cover

abstract type AbstractKDE{T} end
abstract type AbstractKDEInfo{T} end
abstract type AbstractKDEMethod end
abstract type BandwidthEstimator end

struct UnivariateKDE{T,R<:AbstractRange{T},V<:AbstractVector{T}} <: AbstractKDE{T}
    x::R
    f::V
end

struct UnivariateKDEInfo{T,K<:UnivariateKDE{T}} <: AbstractKDEInfo{T}
    npoints::Int
    bandwidth::T
    kernel::K
    cover::Cover.T
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

See also [`LinearBinning`](@ref)
"""
struct HistogramBinning <: AbstractBinningKDE end

"""
    struct LinearBinning <: AbstractBinningKDE end

Base case which generates a density estimate by linear binning.

See also [`HistogramBinning`](@ref)
"""
struct LinearBinning <: AbstractBinningKDE end

function _kdebin(::HistogramBinning, data, lo, hi, Δx, nbins)
    T = eltype(data)
    ν = 0
    f = zeros(T, nbins)
    for x in data
        lo ≤ x ≤ hi || continue  # skip out-of-bounds elements

        # calculate bin index; subtraction of (x == hi) makes the last bin a closed bin
        zz = (x - lo) / Δx
        ii = unsafe_trunc(Int, zz) - (x == hi)
        # N.B. ii is a 0-index offset

        f[ii + 1] += one(T)
        ν += 1
    end
    w = inv(ν * Δx)
    for ii in eachindex(f)
        f[ii] *= w
    end
    return ν, f
end

function _kdebin(::LinearBinning, data, lo, hi, Δx, nbins)
    T = eltype(data)
    ν = 0
    f = zeros(T, nbins)
    for x in data
        lo ≤ x ≤ hi || continue  # skip out-of-bounds elements

        # calculate bin index; subtraction of (x == hi) makes the last bin a closed bin
        zz = (x - lo) / Δx
        ii = unsafe_trunc(Int, zz) - (x == hi)
        # N.B. ii is a 0-index offset

        ww = (zz - ii) - one(T) / 2  # signed distance from the bin center
        off = ifelse(signbit(ww), -1, 1)  # adjascent bin direction
        jj = clamp(ii + off, 0, nbins - 1)  # adj. bin, limited to in-bounds where outer half-bins do not share

        ww = abs(ww)  # weights are positive
        f[ii + 1] += one(T) - ww
        f[jj + 1] += ww
        ν += 1
    end
    w = inv(ν * Δx)
    for ii in eachindex(f)
        f[ii] *= w
    end
    return ν, f
end

function kde(method::AbstractBinningKDE, data;
        lo = nothing, hi = nothing, nbins = nothing, bandwidth = nothing, bwratio = 1,
        cover::Cover.T = Open, kwargs...)
    warn_unused(kwargs)
    T = float(eltype(data))

    # determine lower and upper limits of the histogram
    lo′, hi′ = _extrema(data, lo, hi)
    # filter the data to be only in-bounds
    v = _filter(data, lo′, hi′)

    bw = !isnothing(bandwidth) ? bandwidth : estimate_bandwidth(SilvermanBandwidth(), v)
    if isnothing(nbins)
        nbins′ = max(1, round(Int, bwratio * (hi′ - lo′) / bw))
    else
        nbins′ = Int(nbins)
        nbins′ > 0 || throw(ArgumentError("nbins must be a positive integer"))
    end

    lo′ -= (cover == Closed || cover == ClosedLeft) ? zero(T) : 4bw
    hi′ += (cover == Closed || cover == ClosedRight) ? zero(T) : 4bw

    edges = range(lo′, hi′, length = nbins′ + 1)
    Δx = hi′ > lo′ ? step(edges) : one(lo′)  # 1 bin if histogram has zero width
    centers = edges[2:end] .- Δx / 2

    ν, f = _kdebin(method, v, lo′, hi′, Δx, nbins′)
    estim = UnivariateKDE(centers, f)
    kernel = UnivariateKDE(range(zero(T), zero(T), length = 1), [one(T)])
    info = UnivariateKDEInfo(ν, bw, kernel, cover)
    return estim, info
end

"""
    BasicKDE{M<:AbstractBinningKDE} <: AbstractKDEMethod

**Pipeline:** `BasicKDE` → [`AbstractBinningKDE`](@ref)
"""
struct BasicKDE{M<:AbstractBinningKDE} <: AbstractKDEMethod
    binning::M
end
BasicKDE() = BasicKDE(HistogramBinning())

function kde(method::BasicKDE, data; bwratio = 2, kwargs...)
    binned, info = kde(method.binning, data; bwratio = bwratio, kwargs...)
    return kde(method, binned, info)
end
function kde(method::BasicKDE, binned::UnivariateKDE, info::UnivariateKDEInfo)
    x, f = binned
    bw = info.bandwidth
    Δx = step(x)

    # make sure the kernel axis is centered on zero
    nn = ceil(Int, 4bw / Δx)
    xx = range(-nn * Δx, nn * Δx, step = Δx)

    # construct the convolution kernel
    # N.B. Mathematically normalizing the kernel, such as with
    #        kernel = exp.(-(xx ./ bw) .^ 2 ./ 2) .* (Δx / bw / sqrt(2T(π)))
    #      breaks down when bw << Δx. Instead of trying to work around that, just take the
    #      easy route and just post-normalize a simpler calculation.
    kernel = exp.(-(xx ./ bw) .^ 2 ./ 2)
    kernel ./= sum(kernel)

    # convolve the data with the kernel to construct a density estimate
    f̂ = conv(f, kernel, :same)
    estim = UnivariateKDE(x, f̂)
    info = UnivariateKDEInfo(info.npoints, info.bandwidth,
                             UnivariateKDE(xx, kernel), info.cover)
    return estim, info
end


"""
    LinearBoundaryKDE{M<:AbstractBinningKDE} <: AbstractKDEMethod

**Pipeline:** `LinearBoundaryKDE` → [`BasicKDE`](@ref) → [`AbstractBinningKDE`](@ref)
"""
struct LinearBoundaryKDE{M<:AbstractBinningKDE} <: AbstractKDEMethod
    binning::M
end
LinearBoundaryKDE() = LinearBoundaryKDE(HistogramBinning())

function kde(method::LinearBoundaryKDE, data; bwratio = 8, kwargs...)
    binned, info = kde(method.binning, data; bwratio = bwratio, kwargs...)
    return kde(method, binned, info)
end
function kde(method::LinearBoundaryKDE, binned::UnivariateKDE, info::UnivariateKDEInfo)
    h = copy(binned.f)
    (x, f), info = kde(BasicKDE(method.binning), binned, info)

    # apply a linear boundary correction
    # see Eqn 12 & 16 of Lewis (2019)
    #   N.B. the denominator of A₀ should have [W₂]⁻¹ instead of W₂
    kx, K = info.kernel
    K̂ = plan_conv(f, K)

    Θ = fill!(similar(f), true)
    μ₀ = conv(Θ, K̂, :same)

    K = K .* kx
    replan_conv!(K̂, K)
    μ₁ = conv(Θ, K̂, :same)
    f′ = conv(h, K̂, :same)

    K .*= kx
    replan_conv!(K̂, K)
    μ₂ = conv(Θ, K̂, :same)

    # function to force f̂ to be positive
    # see Eqn. 17 of Lewis (2019)
    pos(f₁, f₂) = iszero(f₁) ? zero(f₁) : f₁ * exp(f₂ / f₁ - one(f₁))
    f .= pos.(f ./ μ₀, (μ₂ .* f .- μ₁ .* f′) ./ (μ₀ .* μ₂ .- μ₁.^2))

    return UnivariateKDE(x, f), info
end


struct MultiplicativeBiasKDE{B<:AbstractBinningKDE,M<:AbstractKDEMethod} <: AbstractKDEMethod
    binning::B
    method::M
end
MultiplicativeBiasKDE() = MultiplicativeBiasKDE(HistogramBinning(), LinearBoundaryKDE())


function kde(method::MultiplicativeBiasKDE, data; bwratio = 8, kwargs...)
    # generate pilot KDE
    base, info = kde(method.binning, data; bwratio = bwratio, kwargs...)
    pilot, info = kde(method.method, base, info)

    # use the pilot KDE to flatten the unsmoothed histogram
    nonzero(x) = iszero(x) ? one(x) : x
    pilot.f .= nonzero.(pilot.f)
    base.f ./= pilot.f

    # then run KDE again on the flattened distribution
    iter, _ = kde(method.method, base, info)

    # unflatten and return
    iter.f .*= pilot.f
    return iter, info
end



struct SilvermanBandwidth <: BandwidthEstimator end
function estimate_bandwidth(::SilvermanBandwidth, v)
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
    estim = kde(v; method = MultiplicativeBiasKDE()
                lo = nothing, hi = nothing, nbins = nothing,
                bandwidth = nothing, bwratio = 8, cover = :open)

Calculate a discrete kernel density estimate (KDE) `f(x)` of the sample distribution of `v`.

The KDE is constructed by first histogramming the input `v` into `nbins` bins with
outermost bin edges spanning `lo` to `hi`, which default to the minimum and maximum of
`v`, respectively, if not provided. The span of the histogram may be expanded outward
based on the value of `cover` (dictating whether the boundaries are open or closed).
The histogram is then convolved with a Gaussian distribution with standard deviation
`bandwidth`. The `bwratio` parameter is used to calculate `nbins` when it is not given and
corresponds to the ratio of the bandwidth to the width of each histogram bin.

Acceptable values of `cover` are:
- `:open` or [`Open`](@ref Cover)
- `:closed` or [`Closed`](@ref Cover)
- `:closedleft`, `:openright`, [`ClosedLeft`](@ref Cover), or [`OpenRight`](@ref Cover)
- `:closedright`, `:openleft`, [`ClosedRight`](@ref Cover), or [`OpenLeft`](@ref Cover)

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
             method::AbstractKDEMethod = MultiplicativeBiasKDE(),
             lo = nothing, hi = nothing, nbins = nothing,
             bandwidth = nothing, bwratio = 8, cover = :open
            )
    cover = to_cover(cover)
    estim, _ = kde(method, data; lo = lo, hi = hi, nbins = nbins,
                   bandwidth = bandwidth, bwratio = bwratio, cover = to_cover(cover))
    estim.f ./= sum(estim.f) * step(estim.x)
    return estim
end
