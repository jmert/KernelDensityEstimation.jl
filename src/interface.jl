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
