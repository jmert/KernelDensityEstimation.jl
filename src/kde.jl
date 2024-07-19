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
  Specifically, the binning is defined over the range ``[a - 8σ, b + 8σ]`` where ``σ``
  is the bandwidth of the Gaussian convolution kernel.
- `ClosedLeft`: The left half-closed interval ``K ∈ [a, +∞)`` is used as the bounds for
  binning by adjusting the upper limit to the range ``[a, b + 8σ]``.
  The equivalent alias `OpenRight` may also be used.
- `ClosedRight`: The right half-closed interval ``K ∈ (-∞, b]`` is used as the bounds for
  binning by adjusting the lower limit to the range ``[a - 8σ, b]``.
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

"""
    AbstractKDE{T}

Abstract supertype of kernel density estimates.

See also [`UnivariateKDE`](@ref)
"""
abstract type AbstractKDE{T} end

"""
    AbstractKDEInfo{T}

Abstract supertype of auxiliary information used during kernel density estimation.

See also [`UnivariateKDEInfo`](@ref)
"""
abstract type AbstractKDEInfo{T} end

"""
    AbstractKDEMethod

The abstract supertype of all kernel density estimation methods, including the data
binning process (see [`AbstractBinningKDE`](@ref)) and subsequent density estimation
techniques (such as [`BasicKDE`](@ref)).
"""
abstract type AbstractKDEMethod end

"""
    AbstractBinningKDE <: AbstractKDEMethod

The abstract supertype of data binning methods which are the first step in the density
estimation process.
The two supported binning methods are [`HistogramBinning`](@ref) and
[`LinearBinning`](@ref).
"""
abstract type AbstractBinningKDE <: AbstractKDEMethod end

"""
    AbstractBandwidthEstimator

Abstract supertype of kernel bandwidth estimation techniques.
"""
abstract type AbstractBandwidthEstimator end

"""
    UnivariateKDE{T,R<:AbstractRange{T},V<:AbstractVector{T}} <: AbstractKDE{T}

## Fields

- `x::R`: The locations (bin centers) of the corresponding density estimate values.
- `f::V`: The density estimate values.
"""
struct UnivariateKDE{T,R<:AbstractRange{T},V<:AbstractVector{T}} <: AbstractKDE{T}
    x::R
    f::V
end

"""
    UnivariateKDEInfo{T} <: AbstractKDEInfo{T}

## Fields
- `npoints::Int`: The number of values in the original data vector.
- `cover::`[`Cover.T`](@ref Cover): The boundary condition assumed in the density estimation
  process.
- `bandwidth::T`: The bandwidth of the convolution `kernel`.
- `kernel::UnivariateKDE{T}`: The convolution kernel used to process the density estimate.
"""
Base.@kwdef struct UnivariateKDEInfo{T} <: AbstractKDEInfo{T}
    npoints::Int
    cover::Cover.T
    bandwidth::T
    kernel::UnivariateKDE{T}
end

# define basic iteration syntax to destructure the contents of a UnivariateKDE;
#   note that property destructuring syntax should be preferred, but that is not available
#   until Julia 1.8, so include this for more convenient use in older Julia versions
Base.iterate(estim::UnivariateKDE) = iterate(estim, Val(:x))
Base.iterate(estim::UnivariateKDE, ::Val{:x}) = (estim.x, Val(:f))
Base.iterate(estim::UnivariateKDE, ::Val{:f}) = (estim.f, nothing)
Base.iterate(::UnivariateKDE, ::Any) = nothing

@noinline _warn_unused(kwargs) = @warn "Unused keyword argument(s)" kwargs=kwargs
warn_unused(kwargs) = length(kwargs) > 0 ? _warn_unused(kwargs) : nothing

"""
    estim, info = estimate(method::AbstractKDEMethod, data::AbstractVector; kwargs...)
    estim, info = estimate(method::AbstractKDEMethod, data::AbstractKDE, info::AbstractKDEInfo; kwargs...)

Apply the kernel density estimation algorithm `method` to the given data, either in the
form of a vector of `data` or a prior density estimate and its corresponding pipeline
`info` (to support being part of a processing pipeline).

## Returns

- `estim::`[`AbstractKDE`](@ref): The resultant kernel density estimate.
- `info::`[`AbstractKDEInfo`](@ref): Auxiliary information describing details of the
  density estimation either useful or necessary for constructing a pipeline of processing
  steps.
"""
function estimate end

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

"""
    struct HistogramBinning <: AbstractBinningKDE end

Base case which generates a density estimate by histogramming the data.

See also [`LinearBinning`](@ref)
"""
struct HistogramBinning <: AbstractBinningKDE end

"""
    struct LinearBinning <: AbstractBinningKDE end

Base case which generates a density estimate by linear binning of the data.

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
        ii = unsafe_trunc(Int, zz) - ((hi > lo) & (x == hi))
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
        ii = unsafe_trunc(Int, zz) - ((hi > lo) & (x == hi))
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

function estimate(method::AbstractBinningKDE, data;
                  lo = nothing, hi = nothing, nbins = nothing, cover::Cover.T = Open,
                  bandwidth = nothing, bwratio = 1, kwargs...)
    warn_unused(kwargs)
    T = float(eltype(data))

    # determine lower and upper limits of the histogram
    lo′, hi′ = _extrema(data, lo, hi)
    # filter the data to be only in-bounds
    v = _filter(data, lo′, hi′)

    bw = !isnothing(bandwidth) ? T(bandwidth) : estimate_bandwidth(SilvermanBandwidth(), v)

    lo′ -= (cover == Closed || cover == ClosedLeft) ? zero(T) : 8bw
    hi′ += (cover == Closed || cover == ClosedRight) ? zero(T) : 8bw

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
    kernel = UnivariateKDE(range(zero(T), zero(T), length = 1), [one(T)])
    info = UnivariateKDEInfo(; npoints = ν, cover, bandwidth = bw, kernel)
    return estim, info
end

"""
    BasicKDE{M<:AbstractBinningKDE} <: AbstractKDEMethod

A baseline density estimation technique which convolves a binned dataset with a Gaussian
kernel truncated at its ``±4σ`` bounds.

## Fields and Constructor Keywords

- `binning::`[`AbstractBinningKDE`](@ref): The binning type to apply to a data vector as the
  first step of density estimation.
  Defaults to [`HistogramBinning()`](@ref).
"""
Base.@kwdef struct BasicKDE{M<:AbstractBinningKDE} <: AbstractKDEMethod
    binning::M = HistogramBinning()
end

function estimate(method::BasicKDE, data; bwratio = 2, kwargs...)
    binned, info = estimate(method.binning, data; bwratio, kwargs...)
    return estimate(method, binned, info)
end
function estimate(::BasicKDE, binned::UnivariateKDE, info::UnivariateKDEInfo)
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
    info = UnivariateKDEInfo(; info.npoints, info.bandwidth, info.cover,
                               kernel = UnivariateKDE(xx, kernel))
    return estim, info
end


"""
    LinearBoundaryKDE{M<:AbstractBinningKDE} <: AbstractKDEMethod

A method of KDE which applies the linear boundary correction of [Jones1996](@citet) as
described in [Lewis2019](@citet) after [`BasicKDE`](@ref) density estimation.
This correction primarily impacts the KDE near a closed boundary (see [`Cover`](@ref)) and
has the effect of improving any non-zero gradient at the boundary (when compared to
normalization corrections which tend to leave the boundary too flat).

## Fields and Constructor Keywords

- `binning::`[`AbstractBinningKDE`](@ref): The binning type to apply to a data vector as the
  first step of density estimation.
  Defaults to [`HistogramBinning()`](@ref).
"""
Base.@kwdef struct LinearBoundaryKDE{M<:AbstractBinningKDE} <: AbstractKDEMethod
    binning::M = HistogramBinning()
end

function estimate(method::LinearBoundaryKDE, data; bwratio = 8, kwargs...)
    binned, info = estimate(method.binning, data; bwratio, kwargs...)
    return estimate(method, binned, info)
end
function estimate(method::LinearBoundaryKDE, binned::UnivariateKDE, info::UnivariateKDEInfo)
    h = copy(binned.f)
    (x, f), info = estimate(BasicKDE(method.binning), binned, info)

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


"""
    MulitplicativeBiasKDE{B<:AbstractBinningKDE,M<:AbstractKDEMethod} <: AbstractKDEMethod

A method of KDE which applies the multiplicative bias correction described in
[Lewis2019](@citet).
This correction is designed to reduce the broadening of peaks inherent to kernel
convolution by using a pilot KDE to flatten the distribution and run a second iteration
of density estimation (since a perfectly uniform distribution cannot be broadened further).

## Fields and Constructor Keywords

- `binning::`[`AbstractBinningKDE`](@ref): The binning type to apply to a data vector as the
  first step of density estimation.
  Defaults to [`HistogramBinning()`](@ref).

- `method::`[`AbstractKDEMethod`](@ref): The KDE method to use for the pilot and iterative
  density estimation.
  Defaults to [`LinearBoundaryKDE()`](@ref).

Note that if the given `method` has a configurable binning type, it is ignored in favor
of the explicit `binning` chosen.
"""
Base.@kwdef struct MultiplicativeBiasKDE{B<:AbstractBinningKDE,M<:AbstractKDEMethod} <: AbstractKDEMethod
    binning::B = HistogramBinning()
    method::M = LinearBoundaryKDE()
end

function estimate(method::MultiplicativeBiasKDE, data; bwratio = 8, kwargs...)
    # generate pilot KDE
    base, info = estimate(method.binning, data; bwratio, kwargs...)
    pilot, info = estimate(method.method, base, info)

    # use the pilot KDE to flatten the unsmoothed histogram
    nonzero(x) = iszero(x) ? one(x) : x
    pilot.f .= nonzero.(pilot.f)
    base.f ./= pilot.f

    # then run KDE again on the flattened distribution
    iter, _ = estimate(method.method, base, info)

    # unflatten and return
    iter.f .*= pilot.f
    return iter, info
end


"""
    SilvermanBandwidth <: AbstractBandwidthEstimator

Estimates the necessary bandwidth of a vector of data ``v`` using Silverman's Rule for
a Gaussian smoothing kernel:
```math
    h = \\left(\\frac{4}{3n}\\right)^{1/5} σ̂
```
where ``n`` is the length of ``v`` and ``σ̂`` is its sample variance.

## References
- [Hansen2009](@citet)
"""
struct SilvermanBandwidth <: AbstractBandwidthEstimator end
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


#struct ISJBandwidth <: AbstractBandwidthEstimator end
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

The bandwidth is estimated using Silverman's rule ([`SilvermanBandwidth`](@ref)) if no
desired bandwidth is given.

The default `method` of density estimation uses the [`MultiplicativeBiasKDE`](@ref)
pipeline, which includes corrections for boundary effects and peak broadening which should
be an acceptable default in many cases, but a different [`AbstractKDEMethod`](@ref) can
be chosen if necessary.

Acceptable values of `cover` are:
- `:open` or [`Open`](@ref Cover)
- `:closed` or [`Closed`](@ref Cover)
- `:closedleft`, `:openright`, [`ClosedLeft`](@ref Cover), or [`OpenRight`](@ref Cover)
- `:closedright`, `:openleft`, [`ClosedRight`](@ref Cover), or [`OpenLeft`](@ref Cover)

# Extended help

- A truncated Gaussian smoothing kernel is assumed. The Gaussian is truncated at ``4σ``.
"""
function kde(data;
             method::AbstractKDEMethod = MultiplicativeBiasKDE(),
             lo = nothing, hi = nothing, nbins = nothing, cover = :open,
             bandwidth = nothing, bwratio = 8,
            )
    cover = to_cover(cover)
    estim, _ = estimate(method, data; lo, hi, nbins, cover, bandwidth, bwratio)
    estim.f ./= sum(estim.f) * step(estim.x)
    return estim
end
