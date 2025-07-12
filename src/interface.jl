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

"""
    AbstractBandwidthEstimator

Abstract supertype of kernel bandwidth estimation techniques.
"""
abstract type AbstractBandwidthEstimator end

"""
    estim, info = estimate(method::AbstractKDEMethod, data::AbstractVector, weights::Union{Nothing, AbstractVector}; kwargs...)
    estim, info = estimate(method::AbstractKDEMethod, data::AbstractKDE, info::AbstractKDEInfo; kwargs...)

Apply the kernel density estimation algorithm `method` to the given data, either in the
form of a vector of `data` (and optionally with corresponding vector of `weights`) or a
prior density estimate and its corresponding pipeline `info` (to support being part of a
processing pipeline).

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
    h = bandwidth(estimator::AbstractBandwidthEstimator, data::AbstractVector{T}
                  lo::T, hi::T, boundary::Boundary.T;
                  weights::Union{Nothing, <:AbstractVector} = nothing
                  ) where {T}

Determine the appropriate bandwidth `h` of the data set `data` (optionally with
corresponding `weights`) using chosen `estimator` algorithm.
The bandwidth is provided the range (`lo` through `hi`) and boundary style (`boundary`) of
the request KDE method for use in filtering and/or correctly interpreting the data, if
necessary.
"""
function bandwidth end

"""
    B = boundary(spec)

Convert the specification `spec` to a boundary style `B`.

Packages may specialize this method on the `spec` argument to modify the behavior of
the boundary inference for new argument types.
"""
function boundary end

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
