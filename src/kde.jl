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
function boundary((lo, hi)::Tuple{<:Number,<:Number})
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

Base.eltype(::AbstractKDE{T}) where {T} = T


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
    p = estimator_order(::Type{<:AbstractKDEMethod})

The bias scaning of the density estimator method, where a return value of `p` corresponds to
bandwidth-dependent biases of the order ``\\mathcal{O}(h^{2p})``.
"""
function estimator_order end

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
    UnivariateKDE{T,U,R<:AbstractRange{T},V<:AbstractVector{U}} <: AbstractKDE{T}

## Fields

- `x::R`: The locations (bin centers) of the corresponding density estimate values.
- `f::V`: The density estimate values.
"""
struct UnivariateKDE{T,U,R<:AbstractRange{T},V<:AbstractVector{U}} <: AbstractKDE{T}
    x::R
    f::V
end

function _univariate_type(::Type{T}) where {T}
    U = typeof(inv(oneunit(T)))
    R = typeof(range(zero(T), zero(T), length = 1))
    V = Vector{U}
    return UnivariateKDE{T, U, R, V}
end
UnivariateKDE{T}(x, f) where {T} = _univariate_type(T)(x, f)


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

Base.:(==)(K1::UnivariateKDE, K2::UnivariateKDE) = K1.x == K2.x && K1.f == K2.f
Base.hash(K::UnivariateKDE, h::UInt) = hash(K.f, hash(K.x, hash(:UnivariateKDE, h)))


"""
    UnivariateKDEInfo{T} <: AbstractKDEInfo{T}

Information about the density estimation process, providing insight into both the
entrypoint parameters and some internal state variables.

# Extended help

## Fields

- `method::`[`AbstractKDEMethod`](@ref):
  The estimation method used to generate the KDE.

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

- `npoints::Int`:
  The number of values in the original data vector.
  Defaults to `-1`.

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

- `kernel::Union{Nothing,`[`UnivariateKDE`](@ref)`{T}}`:
  The convolution kernel used to process the density estimate.
  Defaults to `nothing`.
"""
Base.@kwdef mutable struct UnivariateKDEInfo{T,R,K<:UnivariateKDE} <: AbstractKDEInfo{T}
    method::AbstractKDEMethod
    bounds::Any = nothing
    interval::Tuple{T,T} = (zero(T), zero(T))
    boundary::Boundary.T = Open
    npoints::Int = -1
    bandwidth_alg::Union{Nothing,AbstractBandwidthEstimator} = nothing
    bandwidth::T = zero(T)
    bwratio::R = one(R)
    lo::T = zero(T)
    hi::T = zero(T)
    nbins::Int = -1
    kernel::Union{Nothing,K} = nothing
end

function UnivariateKDEInfo{T}(; kwargs...) where {T}
    R = typeof(one(T))
    K = _univariate_type(R)
    UnivariateKDEInfo{T,R,K}(; kwargs...)
end

function Base.show(io::IO, info::UnivariateKDEInfo{T}) where {T}
    print(io, UnivariateKDEInfo, '{', T, '}')
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

let wd = maximum(map(length ∘ string, fieldnames(UnivariateKDEInfo)))
    function Base.show(io::IO, ::MIME"text/plain", info::UnivariateKDEInfo{T}) where {T}
        print(io, UnivariateKDEInfo, '{', T, '}')
        println(io, ':')

        for fld in fieldnames(typeof(info))
            print(io, lpad(fld, wd + 2), ": ")
            show(io, getfield(info, fld))
            println(io)
        end
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
                spec::Tuple{Union{<:Number,Nothing},
                            Union{<:Number,Nothing},
                            Union{Boundary.T,Symbol}}
               ) where {T}
    lo, hi, B = spec
    return (_extrema(data, lo, hi)..., boundary(B))
end

bounds(data, spec::Tuple{Union{<:Number,Nothing},
                         Union{<:Number,Nothing}}) = bounds(data, (spec..., :open))


@noinline _warn_bounds_override(bounds, lo, hi) =
    @warn "Keyword `bounds` is overriding non-nothing `lo` and/or `hi`" bounds=bounds lo=lo hi=hi

@noinline _warn_unused(kwargs) = @warn "Unused keyword argument(s)" kwargs=kwargs

"""
    data, details = init(method::K, data::AbstractVector{T};
                         lo::Union{Nothing,<:Number} = nothing,
                         hi::Union{Nothing,<:Number} = nothing,
                         boundary::Union{Symbol,Boundary.T} = :open,
                         bounds = nothing,
                         bandwidth::Union{<:Number,<:AbstractBandwidthEstimator} = ISJBandwidth(),
                         bwratio::Real = 1,
                         nbins::Union{Nothing,<:Integer} = nothing,
                         kwargs...) where {K<:AbstractKDEMethod, T}
"""
function init(method::K, data::AbstractVector{T};
              lo::Union{Nothing,<:Number} = nothing,
              hi::Union{Nothing,<:Number} = nothing,
              boundary::Union{Symbol,Boundary.T} = :open,
              bounds = nothing,
              bandwidth::Union{<:Number,<:AbstractBandwidthEstimator} = ISJBandwidth(),
              bwratio::Real = 8,
              nbins::Union{Nothing,<:Integer} = nothing,
              kwargs...) where {K<:AbstractKDEMethod, T}
    @nospecialize method
    options = UnivariateKDEInfo{T}(; method)

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

    # Calculate (or count) the number of data points in the interval
    ν = let l = lo′, h = hi′
        count(x -> l ≤ x ≤ h, data)
    end
    options.npoints = ν

    # Estimate bandwidth from data, as necessary
    if bandwidth isa AbstractBandwidthEstimator
        options.bandwidth_alg = bandwidth
        bandwidth′ = KernelDensityEstimation.bandwidth(bandwidth, data, lo′, hi′, boundary′)::T
        m = estimator_order(typeof(method))
        # Use a larger bandwidth for higher-order estimators which achieve lower bias
        # See Lewis (2019) Eqn 35 and Footnote 10.
        if m > 1
            bandwidth′ *= oftype(one(T), ν) ^ (1 // 5 - 1 // (4m + 1))
        end
    else
        bandwidth′ = convert(T, bandwidth)
    end
    options.bandwidth = bandwidth′
    options.bwratio = bwratio′ = oftype(one(T), bwratio)

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


estimate(M::AbstractKDEMethod, v; kwargs...) = estimate(M, init(M, v; kwargs...)...)


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

function _kdebin(::B, data, lo, hi, nbins) where B <: Union{HistogramBinning, LinearBinning}
    T = eltype(data)  # unitful
    R = typeof(inv(oneunit(T)))  # inverse unitful
    U = typeof(one(T))  # unitless
    wd = hi - lo

    # calculate Δx and Δs = 1/Δx, and use twice-precision-like steps to keep track of the
    # residuals which were rounded away
    if lo == hi
        Δx = zero(T)
        δx = zero(T)
        Δs = oneunit(R)
        δs = zero(R)
    else
        Δx = wd / U(nbins)
        δx = fma(-Δx, U(nbins), wd) / U(nbins)
        Δs = U(nbins) / wd
        δs = fma(-Δs, wd, U(nbins)) / wd
    end

    ν = 0
    f = zeros(R, nbins)
    for x in data
        lo ≤ x ≤ hi || continue  # skip out-of-bounds elements
        if x == hi
            # handle the closed-right bound of the last bin specially
            zz = T(nbins)
            ii = nbins
        else
            # calculate fractional (0-indexed) bin position, using semi-extended precision
            zz = Δs * (x - lo) + δs * (x - lo)
            # truncate to integer (1-indexed) bin index
            ii = unsafe_trunc(Int, zz) + 1

            # despite the attempt to extend precision, values exactly equal to the ideal
            # bin edges can still cause problems; compare the value against the
            # next bin's edge value, and if x is not less than the edge, adjust the index
            # up by one more.
            ee = lo + Δx * ii + δx * ii
            ii += x >= ee
        end

        ν += 1
        if B === HistogramBinning
            f[ii] += oneunit(R)
        elseif B === LinearBinning
            # calculate weight as relative distance from containing bin center
            ww = (zz - ii + 1) - one(U) / 2
            off = ifelse(signbit(ww), -1, 1)  # adjascent bin direction
            jj = clamp(ii + off, 1, nbins)  # adj. bin, limited to in-bounds where outer half-bins do not share

            ww = abs(ww)  # weights are positive
            f[ii] += oneunit(R) * (one(U) - ww)
            f[jj] += oneunit(R) * ww
        end
    end
    w = (oneunit(T) * Δs) / ν
    for ii in eachindex(f)
        f[ii] *= w
    end
    return ν, f
end

estimator_order(::Type{<:AbstractBinningKDE}) = 0

function estimate(method::AbstractBinningKDE, data::AbstractVector{T}, info::UnivariateKDEInfo) where {T}
    lo, hi, nbins = info.lo, info.hi, info.nbins
    ν, f = _kdebin(method, data, lo, hi, nbins)

    if lo == hi
        centers = range(lo, hi, length = 1)
    else
        edges = range(lo, hi, length = nbins + 1)
        centers = edges[2:end] .- step(edges) / 2
    end
    estim = UnivariateKDE{T}(centers, f)

    info.kernel = let R = typeof(one(T))
        UnivariateKDE{R}(range(zero(R), zero(R), length = 1), [one(R)])
    end
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

estimator_order(::Type{<:BasicKDE}) = 1

function estimate(method::BasicKDE, data::AbstractVector, info::UnivariateKDEInfo)
    binned, info = estimate(method.binning, data, info)
    return estimate(method, binned, info)
end
function estimate(::BasicKDE, binned::UnivariateKDE{T}, info::UnivariateKDEInfo) where {T}
    x, f = binned
    bw = info.bandwidth / oneunit(T)
    Δx = step(x) / oneunit(T)

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
    info.kernel = UnivariateKDE{eltype(xx)}(xx, kernel)

    # convolve the data with the kernel to construct a density estimate
    f̂ = conv(f, kernel, :same)
    estim = UnivariateKDE{T}(x, f̂)
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

estimator_order(::Type{<:LinearBoundaryKDE}) = 1

function estimate(method::LinearBoundaryKDE, data::AbstractVector, info::UnivariateKDEInfo)
    binned, info = estimate(method.binning, data, info)
    return estimate(method, binned, info)
end
function estimate(method::LinearBoundaryKDE, binned::UnivariateKDE{T}, info::UnivariateKDEInfo) where {T}
    h = copy(binned.f)
    (x, f), info = estimate(BasicKDE(method.binning), binned, info)
    R = typeof(one(T))

    # apply a linear boundary correction
    # see Eqn 12 & 16 of Lewis (2019)
    #   N.B. the denominator of A₀ should have [W₂]⁻¹ instead of W₂
    kx, K = info.kernel
    K̂ = plan_conv(f, K)

    Θ = fill!(similar(f, R), one(R))
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

    return UnivariateKDE{T}(x, f), info
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

estimator_order(::Type{<:MultiplicativeBiasKDE}) = 2

function estimate(method::MultiplicativeBiasKDE, data::AbstractVector, info::UnivariateKDEInfo)
    binned, info = estimate(method.binning, data, info)
    return estimate(method, binned, info)
end
function estimate(method::MultiplicativeBiasKDE, binned::UnivariateKDE{T}, info::UnivariateKDEInfo) where {T}
    # generate pilot KDE
    pilot, info = estimate(method.method, binned, info)

    # use the pilot KDE to flatten the unsmoothed histogram
    nonzero(x) = iszero(x) ? oneunit(x) : x
    pilot.f .= nonzero.(pilot.f)
    binned.f ./= pilot.f .* oneunit(T)

    # then run KDE again on the flattened distribution
    iter, _ = estimate(method.method, binned, info)

    # unflatten and return
    iter.f .*= pilot.f .* oneunit(T)
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
    μ = μ₋₁ = zero(T)
    σ² = zero(T)^2  # unitful numbers require correct squaring
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
        sqrt(σ²) * (oftype(one(T), (4 // 3)) / ν)^(one(T) / 5)
end

module _ISJ
    include("roots.jl")

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
    function estimate(l::Int, ν::Int, f̂::Vector{T}, h₀::T) where {T<:Real}
        ξ = ((6sqrt(T(2)) - 3) / 7) ^ (one(T) / 5)
        return brent(eps(h₀), 8h₀) do h
            return h - ξ * γˡ(l, ν, f̂, h)
        end
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

- `fallback::Bool`: Whether to fallback to the [`SilvermanBandwidth`](@ref) if the ISJ
  estimator fails to converge. If `false`, an exception is thrown instead.

## References
- [Botev2010](@citet)
"""
Base.@kwdef struct ISJBandwidth{B<:AbstractBinningKDE,R<:Real} <: AbstractBandwidthEstimator
    binning::B = HistogramBinning()
    bwratio::R = 2
    niter::Int = 7
    fallback::Bool = true
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
    f̂ = one(T) == oneunit(T) ? f : f .* oneunit(T)
    FFTW.r2r!(f̂, FFTW.REDFT10)
    f̂ .*= (Δx / oneunit(T))

    # Now we simply solve for the fixed-point solution:
    h = Δx * _ISJ.estimate(isj.niter, ν, f̂, h₀)

    # Check that the fixed-point solver converged to a positive value
    if isnan(h) || h < zero(T)
        if isj.fallback
            h = info.bandwidth  # fallback to the Silverman estimate
        else
            throw(ErrorException("ISJ estimator failed to converge. More data is needed."))
        end
    end
    return h
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
    data′, info = init(method, data;
                       lo, hi, boundary, bounds, nbins, bandwidth, bwratio)
    estim, _ = estimate(method, data′, info)
    # The pipeline is not perfectly norm-preserving, so renormalize before returning to
    # the user.
    estim.f ./= sum(estim.f) * step(estim.x)
    return estim
end
