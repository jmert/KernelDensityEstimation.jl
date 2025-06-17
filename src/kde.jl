import FFTW
import LinearAlgebra: rmul!, rdiv!
import Logging: @warn
@static if VERSION < v"1.9"
    import Base: invokelatest
end


function bounds(::Any, spec::Any)
    throw(ArgumentError("Unknown how to interpret bounds `$spec` as interval and boundary condition"))
end

"""
   lo, hi, bc = bounds(x, (lo, hi, bc))

Refine the interval `lo` to `hi` and boundary condition `bc` to replace values of `nothing`
with appropriate values based on the data in vector `x`.

- If either `lo` or `hi` are nothing, the extrema are used to refine to finite bounds.

- If `boundary` is nothing, the values of `lo` and hi` are used to infer the boundary
  conditions — a finite value corresponds to a closed boundary condition, whereas an
  appropriately-signed infinity implies an open boundary.

## Examples
```jldoctest; setup = :(import .KDE: bounds; using .KDE.Boundary)
julia> bounds(-1:0.1:1, (nothing, nothing, :closed))
(-1.0, 1.0, Closed)

julia> bounds(-1:0.1:1, (0, nothing, :closedleft))
(0.0, 1.0, ClosedLeft)

julia> bounds(-1:0.1:1, (-Inf, Inf, nothing))
(-1.0, 1.0, Open)
```
"""
function bounds(x::AbstractVector{T},
                (lo, hi, boundary)::Tuple{
                    Union{<:Number,Nothing},
                    Union{<:Number,Nothing},
                    Union{Boundary.T,Symbol,Nothing}}
               ) where {T}
    isvalid(v, neg) = isnothing(v) || isfinite(v) || (isinf(v) && signbit(v) == neg)
    isvalid(lo, true)  || throw(ArgumentError("Invalid lower bound: `lo = $lo`"))
    isvalid(hi, false) || throw(ArgumentError("Invalid upper bound: `hi = $hi`"))

    # infer the boundary condition if it is not explicitly given
    if isnothing(boundary)
        if isnothing(lo) || isnothing(hi)
            throw(ArgumentError("Cannot infer boundary conditions with unspecified limits `lo = $lo`, `hi = $hi`"))
        end
        bc = (isfinite(lo) && isfinite(hi)) ? Closed :
             (isfinite(lo) && isinf(hi)) ? ClosedLeft :
             (isfinite(hi) && isinf(lo)) ? ClosedRight :
             Open
    else
        bc = convert(Boundary.T, boundary)
    end

    # now normalize ±∞ as `nothing` to infer limits from the data
    lo = !isnothing(lo) && isinf(lo) ? nothing : lo
    hi = !isnothing(hi) && isinf(hi) ? nothing : hi

    # refine missing values of lo, hi where necessary
    if isnothing(lo) || isnothing(hi)
        a = b = first(x)
        for xi in x
            a = min(a, xi)
            b = max(b, xi)
        end
        # refine limits where necessary
        lo′ = isnothing(lo) ? a : T(lo)
        hi′ = isnothing(hi) ? b : T(hi)
    else
        lo′, hi′ = T(lo), T(hi)
    end
    return (lo′, hi′, bc)::Tuple{T,T,Boundary.T}
end

# default boundary condition is open in kde(), so interpret a pair of limits as an open
# boundary condition
bounds(data::AbstractVector, (lo, hi)::NTuple{2, Union{<:Number,Nothing}}) = bounds(data, (lo, hi, Open))


"""
    UnivariateKDE{T,R<:AbstractRange,V<:AbstractVector{T}} <: AbstractKDE{T}

## Fields

- `x::R`: The locations (bin centers) of the corresponding density estimate values.
- `f::V`: The density estimate values.
"""
struct UnivariateKDE{T,R<:AbstractRange,V<:AbstractVector{T}} <: AbstractKDE{T}
    x::R
    f::V
end

function univariate_type_from_axis_eltype(::Type{S}) where {S}
    T = _invunit(S)
    R = typeof(range(zero(S), zero(S), length = 1))
    V = Vector{T}
    return UnivariateKDE{T,R,V}
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
  [`bounds()`](@ref) with the value of the `.bounds` field.
  Defaults to [`Open`](@ref Boundary).

- `neffective::T`:
  Kish's effective sample size of the data, which equals the length of the original data
  vector for uniformly weighted samples.
  Defaults to `NaN`.

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
    neffective::R = R(NaN)
    bandwidth_alg::Union{Nothing,AbstractBandwidthEstimator} = nothing
    bandwidth::T = zero(T)
    bwratio::R = one(R)
    lo::T = zero(T)
    hi::T = zero(T)
    nbins::Int = -1
    kernel::Union{Nothing,K} = nothing
end

function UnivariateKDEInfo{T}(; kwargs...) where {T}
    R = _unitless(T)
    K = univariate_type_from_axis_eltype(R)
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


@noinline function _warn_unused(kwargs)
    @nospecialize
    @warn "Unused keyword argument(s)" kwargs=kwargs
    return nothing
end

"""
    data, weights, details = init(
            method::K, data::AbstractVector{T},
            weights::Union{Nothing,<:AbstractVector} = nothing;
            bounds = (nothing, nothing, Open),
            bandwidth::Union{<:Number,<:AbstractBandwidthEstimator} = ISJBandwidth(),
            bwratio::Real = 1,
            nbins::Union{Nothing,<:Integer} = nothing,
            kwargs...
        ) where {K<:AbstractKDEMethod, T}
"""
function init(method::K,
              data::AbstractVector{T},
              weights::Union{Nothing,<:AbstractVector} = nothing;
              bounds = (nothing, nothing, Open),
              bandwidth::Union{<:Number,<:AbstractBandwidthEstimator} = ISJBandwidth(),
              bwratio::Real = 8,
              nbins::Union{Nothing,<:Integer} = nothing,
              kwargs...) where {K<:AbstractKDEMethod, T}
    @nospecialize method bounds
    options = UnivariateKDEInfo{T}(; method, bounds)

    # Convert the input bounds to the required canonical representation
    lo′, hi′, boundary′ = KernelDensityEstimation.bounds(data, bounds)::Tuple{T,T,Boundary.T}
    options.boundary = boundary′
    options.interval = (lo′, hi′)

    # Calculate the effective sample size, based on weights and the bounds, using the
    # Kish effective sample size definition:
    #
    #   n_eff = sum(weights)^2 / sum(weights .^ 2)
    #
    # https://search.r-project.org/CRAN/refmans/svyweight/html/eff_n.html
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
    wsum = zero(_unitless(T))
    wsqr = zero(wsum)
    I = isnothing(weights) ? eachindex(data) : eachindex(data, weights)
    @simd for ii in I
        x = @inbounds data[ii]
        w = isnothing(weights) ? one(wsum) : oftype(wsum, @inbounds weights[ii])
        keep = lo′ ≤ x ≤ hi′
        wsum += ifelse(keep, w, zero(wsum))
        wsqr += ifelse(keep, w^2, zero(wsqr))
    end
    options.neffective = neff = wsum^2 / wsqr

    # Estimate bandwidth from data, as necessary
    if bandwidth isa AbstractBandwidthEstimator
        options.bandwidth_alg = bandwidth
        bandwidth′ = KernelDensityEstimation.bandwidth(
                bandwidth, data, lo′, hi′, boundary′; weights)::T
        m = estimator_order(typeof(method))
        # Use a larger bandwidth for higher-order estimators which achieve lower bias
        # See Lewis (2019) Eqn 35 and Footnote 10.
        if m > 1
            # p = 1 // 5 - 1 // (4m + 1)
            p = oftype(neff, 4m - 4) / (20m + 5)
            bandwidth′ *= neff ^ p
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
        invokelatest(_warn_unused, kwargs)::Nothing
    end
    return data, weights, options
end


function estimate(M::AbstractKDEMethod, data, weights = nothing; kwargs...)
    return estimate(M, init(M, data, weights; kwargs...)...)
end


estimator_order(::Type{<:AbstractBinningKDE}) = 0

function estimate(method::AbstractBinningKDE,
                  data::AbstractVector{T},
                  weights::Union{Nothing, <:AbstractVector},
                  info::UnivariateKDEInfo) where {T}
    lo, hi, nbins = info.lo, info.hi, info.nbins

    f = Histogramming._histogram(method, (data,),
                                 (Histogramming.HistEdge(lo, hi, nbins),); weights)
    if lo == hi
        centers = range(lo, hi, length = 1)
    else
        edges = range(lo, hi, length = nbins + 1)
        centers = edges[2:end] .- step(edges) / 2
    end
    estim = UnivariateKDE(centers, f)

    info.kernel = let R = _unitless(T)
        UnivariateKDE(range(zero(R), zero(R), length = 1), [one(R)])
    end
    return estim, info
end


function kernel_gaussian(x::StepRangeLen, σ)
    # N.B. Mathematically normalizing the kernel with the expected factor of
    #          Δx / σ / sqrt(2T(π))
    #      breaks down when σ << Δx. Instead of trying to work around that, take the easy
    #      route and just post-normalize an unnormalized calculation.
    τ = inv(sqrt(oftype(σ, 2)) * σ)
    s = zero(σ)

    # N.B. Given x[ii] must be calculated, benchmarking shows it's faster to compute and
    #      store in g and then follow-up with actually calculating the Gaussian, seemingly
    #      because this version vectorizes calculating x[ii] (whereas the next loop's
    #      exp() prevents vectorization).
    g = copyto!(similar(x), x)
    @simd for ii in eachindex(g)
        @inbounds g[ii] = z = exp(-(g[ii] * τ)^2)
        s += z
    end
    rmul!(g, inv(s))
    return g
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

function estimate(method::BasicKDE,
                  data::AbstractVector,
                  weights::Union{Nothing, <:AbstractVector},
                  info::UnivariateKDEInfo)
    binned, info = estimate(method.binning, data, weights, info)
    return estimate(method, binned, info)
end
function estimate(::BasicKDE, binned::UnivariateKDE, info::UnivariateKDEInfo)
    x, f = binned
    T = _invunit(eltype(x))
    bw = info.bandwidth * oneunit(T)
    Δx = step(x) * oneunit(T)

    # make sure the kernel axis is centered on zero
    nn = ceil(Int, 4bw / Δx)
    xx = range(-nn * Δx, nn * Δx, step = Δx)

    kernel = kernel_gaussian(xx, bw)
    info.kernel = UnivariateKDE(xx, kernel)

    # convolve the data with the kernel to construct a density estimate
    f̂ = conv(f, kernel, ConvShape.SAME)
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

estimator_order(::Type{<:LinearBoundaryKDE}) = 1

function estimate(method::LinearBoundaryKDE,
                  data::AbstractVector,
                  weights::Union{Nothing, <:AbstractVector},
                  info::UnivariateKDEInfo)
    binned, info = estimate(method.binning, data, weights, info)
    return estimate(method, binned, info)
end
function estimate(method::LinearBoundaryKDE, binned::UnivariateKDE, info::UnivariateKDEInfo)
    h = binned.f
    (x, f), info = estimate(BasicKDE(method.binning), binned, info)

    # apply a linear boundary correction
    # see Eqn 12 & 16 of Lewis (2019)
    #   N.B. the denominator of A₀ should have [W₂]⁻¹ instead of W₂
    kx, K = info.kernel
    KI = eachindex(K)
    R = eltype(K)
    K̂ = plan_conv(f, K)

    Θ = fill!(similar(f, R), one(R))
    μ₀ = conv(Θ, K̂, ConvShape.SAME)

    @simd for ii in KI
        @inbounds K[ii] *= kx[ii]
    end
    replan_conv!(K̂, K)
    μ₁ = conv(Θ, K̂, ConvShape.SAME)
    f′ = conv(h, K̂, ConvShape.SAME)

    @simd for ii in KI
        @inbounds K[ii] *= kx[ii]
    end
    replan_conv!(K̂, K)
    μ₂ = conv(Θ, K̂, ConvShape.SAME)

    # Function to force f̂ to be positive — see Eqn. 17 of Lewis (2019)
    # N.B. Mathematically f from basic KDE is strictly non-negative, but numerically we
    #      frequently find small negative values. The following only works when it is truly
    #      non-negative.
    @simd for ii in eachindex(f)
        @inbounds begin
            f₀ = max(f[ii], zero(eltype(f)))
            f₁ = f₀ / μ₀[ii]
            f₂ = (μ₂[ii] * f[ii] - μ₁[ii] * f′[ii]) / (μ₀[ii] * μ₂[ii] - μ₁[ii]^2)
            f[ii] = iszero(f₁) ? f₁ : f₁ * exp(f₂ / f₁ - one(f₁))
        end
    end

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

estimator_order(::Type{<:MultiplicativeBiasKDE}) = 2

function estimate(method::MultiplicativeBiasKDE,
                  data::AbstractVector,
                  weights::Union{Nothing, <:AbstractVector},
                  info::UnivariateKDEInfo)
    binned, info = estimate(method.binning, data, weights, info)
    return estimate(method, binned, info)
end
function estimate(method::MultiplicativeBiasKDE, binned::UnivariateKDE, info::UnivariateKDEInfo)
    # generate pilot KDE
    pilot, info = estimate(method.method, binned, info)
    I = eachindex(pilot.f)

    T = _invunit(eltype(binned))
    # use the pilot KDE to flatten the unsmoothed histogram
    @inline nonzero(x) = iszero(x) ? one(x) : x * oneunit(T)
    @simd for ii in I
        @inbounds binned.f[ii] /= nonzero(pilot.f[ii])
    end

    # then run KDE again on the flattened distribution
    iter, _ = estimate(method.method, binned, info)

    # unflatten and return
    @simd for ii in I
        @inbounds iter.f[ii] *= nonzero(pilot.f[ii])
    end
    return iter, info
end


@noinline function _warn_bounds_override(bounds, lo, hi, boundary)
    @nospecialize
    @warn("Keyword `bounds` is overriding non-nothing `lo`, `hi`, and/or `boundary`.",
          bounds = bounds, lo = lo, hi = hi, boundary = boundary)
    return nothing
end

"""
    estim = kde(data;
                weights = nothing, method = MultiplicativeBiasKDE(),
                lo = nothing, hi = nothing, boundary = :open, bounds = nothing,
                bandwidth = ISJBandwidth(), bwratio = 8 nbins = nothing)

Calculate a discrete kernel density estimate (KDE) `estim` from samples `data`, optionally
weighted by a corresponding vector of `weights`.

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
             weights = nothing, method::AbstractKDEMethod = MultiplicativeBiasKDE(),
             lo = nothing, hi = nothing, boundary = nothing, bounds = nothing,
             bandwidth = ISJBandwidth(), bwratio = 8, nbins = nothing,
            )
    # Warn if the bounds argument is going to override any values provided in lo, hi, or
    # boundary
    if !isnothing(bounds) && (!isnothing(lo) || !isnothing(hi) || !isnothing(boundary))
        invokelatest(_warn_bounds_override, bounds, lo, hi, boundary)
        lo = hi = boundary = nothing
    end

    # Consolidate high-level keywords for (; lo, hi, boundary) into the lower-level
    # interface's (; bounds) keyword.
    if isnothing(bounds)
        bc = isnothing(boundary) ? Open : boundary
        bounds = (lo, hi, bc)
    end
    estim, _ = estimate(method, data, weights;
                        bounds, nbins, bandwidth, bwratio)

    # The pipeline is not perfectly norm-preserving, so renormalize before returning to
    # the user.
    rdiv!(estim.f, sum(estim.f) * step(estim.x))
    return estim
end
