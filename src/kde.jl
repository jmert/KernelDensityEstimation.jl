import FFTW
import Logging: @warn

"""
```julia
@enum T Closed Open ClosedLeft ClosedRight
const OpenLeft = ClosedRight
const OpenRight = ClosedLeft
```

Enumeration to describe the desired boundary conditions of the domain of the kernel
density estimate ``K``.
For some given data ``d ∈ [a, b]``, the boundary conditions have the following impact:

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
baremodule Boundary
    import ..Base.@enum

    export Closed, Open, ClosedLeft, ClosedRight, OpenLeft, OpenRight

    @enum T Closed Open ClosedLeft ClosedRight
    const OpenLeft = ClosedRight
    const OpenRight = ClosedLeft
end
using .Boundary

"""
    B = boundary(spec)

Convert the specification `spec` to a boundary style `B`.

Packages may specialize this method on the `spec` argument to modify the behavior of
the boundary inference for new argument types.
"""
function boundary end

boundary(spec::Boundary.T) = spec

"""
    B = boundary(spec::Symbol)

Converts the following symbols to its corresponding boundary style:
- `:open` -> [`Open`](@ref Boundary)
- `:closed` -> [`Closed`](@ref Boundary)
- `:closedleft` and `:openright` -> [`ClosedLeft`](@ref Boundary)
- `:closedright` and `:openleft` -> [`ClosedRight`](@ref Boundary)
"""
boundary(spec::Symbol) = spec === :open ? Open :
                         spec === :closed ? Closed :
                         spec === :closedleft ? ClosedLeft :
                         spec === :openright ? ClosedLeft :
                         spec === :closedright ? ClosedRight :
                         spec === :openleft ? ClosedRight :
                         throw(ArgumentError("Unknown boundary option: $spec"))

"""
    B = boundary((lo, hi)::Tuple{<:Real,<:Real})

Infers the appropriate boundary condition `B` given the lower and upper bounds of the
domain. A finite value corresponds to a closed boundary, whereas an appropriately-signed
infinity implies and open boundary. Having either the wrong sign (such as `hi == -Inf`)
or a NaN value is an error.
"""
function boundary((lo, hi)::Tuple{<:Real,<:Real})
    if isfinite(lo) && isfinite(hi)
        return Closed
    elseif isfinite(lo) && (isinf(hi) && hi > zero(hi))
        return ClosedLeft
    elseif isfinite(hi) && (isinf(lo) && lo < zero(lo))
        return ClosedRight
    elseif (isinf(lo) && lo < zero(lo)) && (isinf(hi) && hi > zero(hi))
        return Open
    else
        throw(ArgumentError("Could not infer boundary for `lo = $lo`, `hi = $hi`"))
    end
end


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

"""
    lo, hi, boundary = bounds(data::AbstractVector{T}, spec) where {T}

Determine the appropriate interval, from `lo` to `hi` with boundary style `boundary`, for
the density estimate, given the data vector `data` and KDE argument `bounds`.

Packages may specialize this method on the `spec` argument to modify the behavior of
the interval and boundary refinement for new argument types.
"""
function bounds end

"""
    h = bandwidth(estimator::AbstractBandwidthEstimator, data::AbstractVector{T},
                  lo::T, hi::T, boundary::Boundary.T) where {T}

Determine the appropriate bandwidth `h` of the data set `data` using chosen `estimator`
algorithm.
The bandwidth is provided the range (`lo` through `hi`) and boundary style (`boundary`) of
the request KDE method for use in filtering and/or correctly interpreting the data, if
necessary.
"""
function bandwidth end

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

function Base.show(io::IO, K::UnivariateKDE{T}) where {T}
    if get(io, :compact, false)::Bool
        print(io, UnivariateKDE, '{', T, "}(…)")
    else
        if get(io, :limit, false)::Bool
            print(io, UnivariateKDE, '{', T, '}', '(')
            io′ = IOContext(io, :compact => true)
            print(io′, first(K.x), ':', step(K.x), ':', last(K.x), ", ")
            show(io′, K.f)
            print(io, ')')
        else
            invoke(show, Tuple{typeof(io), Any}, io, K)
        end
    end
end


"""
    UnivariateKDEInfo{T} <: AbstractKDEInfo{T}

Information about the density estimation process, providing insight into both the
entrypoint parameters and some internal state variables.

# Extended help

## Fields

- `bounds::Any`:
  The bounds specification of the estimate as passed to [`init()`](@ref), prior to making
  it concrete via calling [`bounds()`](@ref).
  Defaults to `nothing`.

- `interval::Tuple{T,T}`:
  The concrete interval of the density estimate after calling [`bounds()`](@ref) with the
  value of the `.bounds` field but before adding requisite padding for open boundary
  conditions.
  Defaults to `(zero(T), zero(T))`.

- `boundary::`[`Boundary.T`](@ref Boundary):
  The concrete boundary condition assumed in the density estimate after calling
  [`boundary()`](@ref) with the value of the `.bounds` field.
  Defaults to [`Open`](@ref Boundary).

- `bandwidth_alg::Union{Nothing,`[`AbstractBandwidthEstimator`](@ref)`}`:
  Algorithm used to estimate an appropriate bandwidth, if a concrete value was not
  provided to the estimator, otherwise `nothing`.
  Defaults to `nothing`.

- `bandwidth::T`:
  The bandwidth of the convolution `kernel`.
  Defaults to `zero(T)`.

- `bwratio::T`:
  The ratio between the bandwidth and the width of a histogram bin, used only when the
  number of bins `.nbins` is not explicitly provided.
  Defaults to `one(T)`.

- `lo::T`:
  The lower edge of the first bin in the density estimate, after possibly adjusting for
  an open boundary condition compared to the `.interval` field.
  Defaults to `zero(T)`.

- `hi::T`:
  The upper edge of the last bin in the density estimate, after possibly adjusting for
  an open boundary condition compared to the `.interval` field.
  Defaults to `zero(T)`.

- `nbins::Int`:
  The number of bins used in the histogram at the beinning of the density estimatation.
  Defaults to `-1`.

- `npoints::Int`:
  The number of values in the original data vector.
  Defaults to `-1`.

- `kernel::Union{Nothing,`[`UnivariateKDE`](@ref)`{T}}`:
  The convolution kernel used to process the density estimate.
  Defaults to `nothing`.
"""
Base.@kwdef mutable struct UnivariateKDEInfo{T} <: AbstractKDEInfo{T}
    bounds::Any = nothing
    interval::Tuple{T,T} = (zero(T), zero(T))
    boundary::Boundary.T = Open
    bandwidth_alg::Union{Nothing,AbstractBandwidthEstimator} = nothing
    bandwidth::T = zero(T)
    bwratio::T = one(T)
    lo::T = zero(T)
    hi::T = zero(T)
    nbins::Int = -1
    npoints::Int = -1
    kernel::Union{Nothing,UnivariateKDE{T}} = nothing
end

function Base.show(io::IO, info::UnivariateKDEInfo)
    show(io, typeof(info))
    if get(io, :compact, false)::Bool
        print(io, "(…)")
    else
        first = true
        print(io, '(')
        for fld in fieldnames(typeof(info))
            !first && print(io, ", ")
            print(io, fld, " = ")
            show(io, getfield(info, fld))
            first = false
        end
        print(io, ')')
    end
end

function Base.show(io::IO, ::MIME"text/plain", info::UnivariateKDEInfo)
    show(io, typeof(info))
    println(io, ':')

    wd = textwidth("bandwidth_alg")
    for fld in fieldnames(typeof(info))
        print(io, lpad(fld, wd + 2), ": ")
        show(io, getfield(info, fld))
        println(io)
    end
end


# define basic iteration syntax to destructure the contents of a UnivariateKDE;
#   note that property destructuring syntax should be preferred, but that is not available
#   until Julia 1.8, so include this for more convenient use in older Julia versions
Base.iterate(estim::UnivariateKDE) = iterate(estim, Val(:x))
Base.iterate(estim::UnivariateKDE, ::Val{:x}) = (estim.x, Val(:f))
Base.iterate(estim::UnivariateKDE, ::Val{:f}) = (estim.f, nothing)
Base.iterate(::UnivariateKDE, ::Any) = nothing

function _extrema(data::AbstractVector{T}, lo, hi) where {T}
    !isnothing(lo) && !isnothing(hi) && return (T(lo), T(hi))::Tuple{T,T}
    a = b = first(data)
    for x in data
        a = min(a, x)
        b = max(b, x)
    end
    return (isnothing(lo) ? a : T(lo),
            isnothing(hi) ? b : T(hi))::Tuple{T,T}
end

"""
   lo, hi, boundary = bounds(data::AbstractVector{T}, (lo, hi, boundary)) where {T}

Determine the appropriate span from `lo` to `hi` for the density estimate, given the data
vector `data` and possible explicit known boundaries as arguments. The extrema are used
as necessary to refine a `nothing` input bound.
"""
function bounds(data::AbstractVector{T},
                spec::Tuple{Union{<:Real,Nothing},
                            Union{<:Real,Nothing},
                            Union{Boundary.T,Symbol}}
               ) where {T}
    lo, hi, B = spec
    return (_extrema(data, lo, hi)..., boundary(B))
end

bounds(data, spec::Tuple{Union{<:Real,Nothing},
                         Union{<:Real,Nothing}}) = bounds(data, (spec..., :open))


@noinline _warn_bounds_override(bounds, lo, hi) =
    @warn "Keyword `bounds` is overriding non-nothing `lo` and/or `hi`" bounds=bounds lo=lo hi=hi

@noinline _warn_unused(kwargs) = @warn "Unused keyword argument(s)" kwargs=kwargs

"""
    data, details = init(data::AbstractVector{T};
                         lo = nothing, hi = nothing, boundary = nothing, bounds = nothing,
                         nbins::Union{Nothing,<:Integer} = nothing,
                         bandwidth::Union{<:Real,<:AbstractBandwidthEstimator} = ISJBandwidth(),
                         bwratio::Real = 1, kwargs...) where {T}
"""
function init(data::AbstractVector{T};
              lo::Union{Nothing,<:Real} = nothing,
              hi::Union{Nothing,<:Real} = nothing,
              boundary::Union{Symbol,Boundary.T} = :open,
              bounds = nothing,
              bandwidth::Union{<:Real,<:AbstractBandwidthEstimator} = ISJBandwidth(),
              bwratio::Real = 1,
              nbins::Union{Nothing,<:Integer} = nothing,
              kwargs...) where {T}

    options = UnivariateKDEInfo{T}()

    # Handle the option to provide either a single bounds argument or the triple of lo, hi,
    # and boundary
    if !isnothing(bounds) && (!isnothing(lo) || !isnothing(hi))
        _warn_bounds_override(bounds, lo, hi)
    end
    options.bounds = bounds′ = isnothing(bounds) ? (lo, hi, boundary) : bounds

    # Convert the input bounds to the required canonical representation
    lo′, hi′, boundary′ = KernelDensityEstimation.bounds(data, bounds′)::Tuple{T,T,Boundary.T}
    options.boundary = boundary′
    options.interval = (lo′, hi′)

    # Estimate bandwidth from data, as necessary
    if bandwidth isa AbstractBandwidthEstimator
        options.bandwidth_alg = bandwidth
        bandwidth′ = KernelDensityEstimation.bandwidth(bandwidth, data, lo′, hi′, boundary′)::T
    else
        bandwidth′ = convert(T, bandwidth)
    end
    options.bandwidth = bandwidth′
    options.bwratio = bwratio′ = convert(T, bwratio)::T

    # Then expand the bounds if the bound(s) are open
    lo′ -= (boundary′ == Closed || boundary′ == ClosedLeft) ? zero(T) : 8bandwidth′
    hi′ += (boundary′ == Closed || boundary′ == ClosedRight) ? zero(T) : 8bandwidth′
    options.lo = lo′
    options.hi = hi′

    # Calculate the number of bins to use in the histogram
    if isnothing(nbins)
        nbins′ = max(1, round(Int, bwratio′ * (hi′ - lo′) / bandwidth′))
    else
        nbins′ = Int(nbins)
        nbins′ > 0 || throw(ArgumentError("nbins must be a positive integer"))
    end
    options.nbins = nbins′

    # Warn if we received any parameters which should have been consumed earlier in
    # the pipeline
    if length(kwargs) > 0
        _warn_unused(kwargs)
    end
    return data, options
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

function estimate(method::AbstractBinningKDE, data; kwargs...)
    data, info = init(data; kwargs...)
    lo, hi, nbins = info.lo, info.hi, info.nbins

    edges = range(lo, hi, length = nbins + 1)
    Δx = hi > lo ? step(edges) : one(lo)  # 1 bin if histogram has zero width
    centers = edges[2:end] .- Δx / 2

    ν, f = _kdebin(method, data, lo, hi, Δx, nbins)
    estim = UnivariateKDE(centers, f)
    info.kernel = UnivariateKDE(range(zero(lo), zero(lo), length = 1), [one(lo)])
    info.npoints = ν
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
    info.kernel = UnivariateKDE(xx, kernel)

    # convolve the data with the kernel to construct a density estimate
    f̂ = conv(f, kernel, :same)
    estim = UnivariateKDE(x, f̂)
    return estim, info
end


"""
    LinearBoundaryKDE{M<:AbstractBinningKDE} <: AbstractKDEMethod

A method of KDE which applies the linear boundary correction of [Jones1996](@citet) as
described in [Lewis2019](@citet) after [`BasicKDE`](@ref) density estimation.
This correction primarily impacts the KDE near a closed boundary (see [`Boundary`](@ref)) and
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

See also [`ISJBandwidth`](@ref)

## References
- [Hansen2009](@citet)
"""
struct SilvermanBandwidth <: AbstractBandwidthEstimator end

function bandwidth(::SilvermanBandwidth, v::AbstractVector{T},
                   lo::T, hi::T, ::Boundary.T) where {T}
    # Get the count and variance simultaneously
    #   Calculate variance via Welford's algorithm
    ν = 0
    μ = μ₋₁ = σ² = zero(T)
    for x in v
        lo ≤ x ≤ hi || continue  # skip out-of-bounds elements
        ν += 1
        w = one(T) / ν
        μ₋₁, μ = μ, (one(T) - w) * μ + w * x
        σ² = (one(T) - w) * σ² + w * (x - μ) * (x - μ₋₁)
    end
    # From Hansen (2009) — https://users.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf
    # for a Gaussian kernel:
    # - Table 1:
    #   - R(k) = 1 / 2√π
    #   - κ₂(k) = 1
    # - Section 2.9, letting ν = 2:
    #   - bw = σ̂ n^(-1/5) C₂(k)
    #     C₂(k) = 2 ( 8R(k)√π / 96κ₂² )^(1/5) == (4/3)^(1/5)
    return iszero(σ²) ? eps(one(T)) :
        sqrt(σ²) * (T(4 // 3) / ν)^(one(T) / 5)
end

module _ISJ
    using Roots: Roots, ZeroProblem, solve

    # Calculates norm of the the j-th derivative of the convolved density function, e.g.
    #
    #   ||∂ʲ/∂xʲ[ f(x) * K_h(x) ]||²
    #
    # but in an efficient way which makes use of knowing K_h(x) is a Gaussian with
    # standard deviation h and using the dicrete cosine transform of the distribution
    # since convolution and derivatives are efficient in Fourier space.
    function ∂ʲ(f̂::Vector{T}, h::T, j::Integer) where T <: Real
        N = length(f̂)
        expfac = -(2T(π) * h)^2

        norm = zero(T)
        for n in 1:(N - 1)
            f̂ₙ² = abs2(f̂[n + 1])
            kₙ = T(n) / 2N
            norm += f̂ₙ² * kₙ^(2j) * exp(expfac * kₙ^2)
        end
        if j == 0
            norm += abs2(f̂[1]) / 2
        end
        norm *= (2T(π))^(2j) / 2N
        return norm
    end

    # Calculates the γ function, defined in Botev et al as the right-hand side of Eqn. 29
    function γ(ν::Int, f̂::Vector{T}, j::Int, h::T) where {T<:Real}
        N² = ∂ʲ(f̂, h, j + 1)
        fac1 = (T(1) + T(2) ^ -T(j + 0.5)) / 3
        fac2 = prod(T(1):2:T(2j-1)) / (sqrt(T(π) / 2) * ν * N²)
        return (fac1 * fac2) ^ (T(1) / (2j + 3))
    end

    # Calculates the iteratively-defined γˡ function, defined in Botev et al, between
    # Eqns. 29 and 30.
    function γˡ(l::Int, ν::Integer, f̂::Vector{T}, h::T) where {T<:Real}
        for j in l:-1:1
            h = γ(ν, f̂, j, h)
        end
        return h
    end

    # Express the fixed-point equation (Botev et al Eqn. 30) as an expression where the
    # root is the desired bandwidth.
    function fixed_point_equation(h::T, (l, ν, f̂)::Tuple{Int, Int, Vector{T}}) where {T<:Real}
        ξ = ((6sqrt(T(2)) - 3) / 7) ^ (one(T) / 5)
        t = h - ξ * γˡ(l, ν, f̂, h)
        return t
    end

    function estimate(l::Int, ν::Int, f̂::Vector{T}, h₀::T) where {T<:Real}
        problem = ZeroProblem(fixed_point_equation, h₀)
        t = solve(problem; p = (l, ν, f̂))
        if isnan(t)
            throw(ErrorException("ISJ estimator failed to converge. More data is needed."))
        end
        return t
    end
end

"""
    ISJBandwidth <: AbstractBandwidthEstimator

Estimates the necessary bandwidth of a vector of data ``v`` using the Improved
Sheather-Jones (ISJ) plug-in estimator of [Botev2010](@citet).

This estimator is more capable of choosing an appropriate bandwidth for bimodal (and other
highly non-Gaussian) distributions, but comes at the expense of greater computation time
and no guarantee that the estimator converges when given very few data points.

See also [`SilvermanBandwidth`](@ref)

## Fields
- `binning::`[`AbstractBinningKDE`](@ref): The binning type to apply to a data vector as the
  first step of bandwidth estimation.
  Defaults to [`HistogramBinning()`](@ref).

- `bwratio::Int`: The relative resolution of the binned data used by the ISJ plug-in
  estimator — there are `bwratio` bins per interval of size ``h₀``, where the intial
  rough initial bandwidth estimate is given by the [`SilvermanBandwidth`](@ref) estimator.
  Defaults to 2.

- `niter::Int`: The number of iterations to perform in the plug-in estimator.
  Defaults to 7, in accordance with Botev et. al. who state that higher orders show little
  benefit.

## References
- [Botev2010](@citet)
"""
Base.@kwdef struct ISJBandwidth{B<:AbstractBinningKDE} <: AbstractBandwidthEstimator
    binning::B = HistogramBinning()
    bwratio::Int = 2
    niter::Int = 7
end

function bandwidth(isj::ISJBandwidth{<:Any}, v::AbstractVector{T},
                   lo::T, hi::T, boundary::Boundary.T) where {T}
    # The Silverman bandwidth estimator should be sufficient to obtain a fine-enough
    # binning that the ISJ algorithm can iterate.
    # We need a histogram, so just reuse the binning base case of the estimator pipeline
    # to provide what we need.
    (x, f), info = estimate(isj.binning, v; lo, hi, boundary, isj.bwratio,
                            bandwidth = SilvermanBandwidth())

    ν = info.npoints
    Δx = step(x)
    # The core of the ISJ algorithm works in a normalized unit system where Δx = 1.
    # Two things of note:
    #
    #   1. We initialize the fixed-point algorithm with the Silverman bandwidth, but
    #      scaled correctly for the change in axis. Then afterwards, the ISJ bandwidth
    #      will need to be scaled back to the original axis, e.g. h → Δx × h
    h₀ = info.bandwidth / Δx
    #   2. Via the Fourier scaling theorem, f(x / Δx) ⇔ Δx × f̂(k), we must scale the DCT
    #      by the grid step size.
    f̂ = FFTW.r2r!(f, FFTW.REDFT10) .* Δx

    # Now we simply solve for the fixed-point solution:
    return Δx * _ISJ.estimate(isj.niter, ν, f̂, h₀)
end


"""
    estim = kde(v;
                method = MultiplicativeBiasKDE(),
                lo = nothing, hi = nothing, boundary = :open, bounds = nothing,
                bandwidth = ISJBandwidth(), bwratio = 8 nbins = nothing)

Calculate a discrete kernel density estimate (KDE) `f(x)` from samples `v`.

The default `method` of density estimation uses the [`MultiplicativeBiasKDE`](@ref)
pipeline, which includes corrections for boundary effects and peak broadening which should
be an acceptable default in many cases, but a different [`AbstractKDEMethod`](@ref) can
be chosen if necessary.

The interval of the density estimate can be controlled by either the set of `lo`, `hi`, and
`boundary` keywords or the `bounds` keyword, where the former are conveniences for setting
`bounds = (lo, hi, boundary)`.
The minimum and maximum of `v` are used if `lo` and/or `hi` are `nothing`, respectively.
(See also [`bounds`](@ref).)

The KDE is constructed by first histogramming the input `v` into `nbins` many bins with
outermost bin edges spanning `lo` to `hi`. The span of the histogram may be expanded outward
based on `boundary` condition, dictating whether the boundaries are open or closed.
The `bwratio` parameter is used to calculate `nbins` when it is not given and corresponds
(approximately) to the ratio of the bandwidth to the width of each histogram bin.

Acceptable values of `boundary` are:
- `:open` or [`Open`](@ref Boundary)
- `:closed` or [`Closed`](@ref Boundary)
- `:closedleft`, `:openright`, [`ClosedLeft`](@ref Boundary), or [`OpenRight`](@ref Boundary)
- `:closedright`, `:openleft`, [`ClosedRight`](@ref Boundary), or [`OpenLeft`](@ref Boundary)

The histogram is then convolved with a Gaussian distribution with standard deviation
`bandwidth`.
The default bandwidth estimator is the Improved Sheather-Jones ([`ISJBandwidth`](@ref)) if
no explicit bandwidth is given.
"""
function kde(data;
             method::AbstractKDEMethod = MultiplicativeBiasKDE(),
             lo = nothing, hi = nothing, boundary = :open, bounds = nothing,
             bandwidth = ISJBandwidth(), bwratio = 8, nbins = nothing,
            )
    estim, _ = estimate(method, data; lo, hi, boundary, bounds, nbins, bandwidth, bwratio)
    # The pipeline is not perfectly norm-preserving, so renormalize before returning to
    # the user.
    estim.f ./= sum(estim.f) * step(estim.x)
    return estim
end
