import LinearAlgebra: Symmetric, cholesky, cholesky!, isposdef

# convenient wrapper for univariate inputs
function bandwidth(estim::AbstractBandwidthEstimator,
                   data::AbstractVector, bounds::Tuple{Any,Any,Boundary.T},
                   weights::Union{Nothing,<:AbstractVector} = nothing)
    return bandwidth(estim, (data,), (bounds,), weights)
end


# Get the effective sample size and (co)variance simultaneously
#
#   - Calculate variance via Welford's algorithm:
#
#     https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
#
#   - Calculate the effective sample size, based on weights and the bounds, using the
#     Kish effective sample size definition:
#
#         n_eff = sum(weights)^2 / sum(weights .^ 2)
#
#     https://search.r-project.org/CRAN/refmans/svyweight/html/eff_n.html
#     https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
function _neff_covar(coords::Tuple{Vararg{AbstractVector,N}},
                     lo::Tuple{Vararg{Any,N}},
                     hi::Tuple{Vararg{Any,N}},
                     weights::Union{Nothing,<:AbstractVector}) where {N}
    T = promote_type(map(_unitless∘eltype, coords)...)
    wsum = wsqr = isnothing(weights) ? zero(T) : zero(eltype(weights))

    x = zeros(T, N)
    μ = zeros(T, N)
    μ₋₁ = zeros(T, N)
    Σ = zeros(T, N, N)
    I = isnothing(weights) ? eachindex(coords...) : eachindex(weights, coords...)
    for ii in I
        # @. x = coords[ii] * _invunit(typeof(coords[ii]))
        # !all(@. lo ≤ x ≤ hi) && continue
        indomain = true
        for jj in 1:N
            v = coords[jj][ii]
            x[jj] = v * oneunit(_invunit(typeof(v)))
            indomain &= lo[jj] ≤ v ≤ hi[jj]
        end
        !indomain && continue

        w = isnothing(weights) ? one(wsum) : weights[ii]
        wsum += w
        wsqr += w^2
        ω = w / wsum
        ω̄ = one(ω) - ω

        # @. μ₋₁ = μ
        # @. μ = ω̄ * μ₋₁ + ω * x
        for jj in 1:N
            μ₋₁[jj] = μ[jj]
            μ[jj] = ω̄ * μ[jj] + ω * x[jj]
        end
        # @. Σ = ω̄ * Σ + ω * (x - μ) * (x - μ₋₁)'
        for jj in 1:N
            yy = x[jj] - μ₋₁[jj]
            for kk in 1:N
                Σ[kk,jj] = ω̄ * Σ[kk, jj] + ω * (x[kk] - μ[kk]) * yy
            end
        end
    end
    neff = wsum^2 / wsqr
    return neff, Σ
end

# specialize for 1D case where the variance is scalar, so allocating arrays can be avoided
function _neff_covar(coords::Tuple{AbstractVector},
                     lo::Tuple{Any},
                     hi::Tuple{Any},
                     weights::Union{Nothing,<:AbstractVector})
    T = promote_type(map(_unitless∘eltype, coords)...)
    wsum = wsqr = isnothing(weights) ? zero(T) : zero(eltype(weights))

    x = zero(T)
    μ = zero(T)
    μ₋₁ = zero(T)
    Σ = zero(T)
    I = isnothing(weights) ? eachindex(coords...) : eachindex(weights, coords...)
    for ii in I
        v = coords[1][ii]
        lo[1] ≤ v ≤ hi[1] || continue
        x = v * oneunit(_invunit(typeof(v)))

        w = isnothing(weights) ? one(wsum) : weights[ii]
        wsum += w
        wsqr += w^2
        ω = w / wsum
        ω̄ = one(ω) - ω

        μ₋₁ = μ
        μ = ω̄ * μ + ω * x
        Σ = ω̄ * Σ + ω * (x - μ) * (x - μ₋₁)
    end
    neff = wsum^2 / wsqr
    return neff, Σ
end


"""
    SilvermanBandwidth <: AbstractBandwidthEstimator

Estimates the necessary bandwidth of data at ``d``-dimensional coordinates
``(𝒗_1, 𝒗_2, …, 𝒗_d)`` with weights ``𝒘`` using Silverman's Rule for a Gaussian smoothing
kernel.

For the univariate (``d = 1``) case:
```math
    h = \\left(\\frac{4}{3n_\\mathrm{eff}}\\right)^{1/5} σ̂
```
where ``n_\\mathrm{eff}`` is the effective number of degrees of freedom of the data and
``σ̂^2`` is its sample variance.

In the multivariate case (``d ≥ 2``):
```math
    𝒉 = \\left(\\frac{4}{(2 + d)n_\\mathrm{eff}}\\right)^{1/(4 + d)} \\sqrt{𝜮̂}
```
where ``\\sqrt{𝜮̂}`` is a Cholesky decomposition of the weighted sample covariance.

See also [`ISJBandwidth`](@ref)

# Extended help

The sample (co)variance and effective number of degrees of freedom are calculated using
weighted statistics, where the latter is defined to be Kish's effective sample size
``n_\\mathrm{eff} = (\\sum_i 𝒘_i)^2 / \\sum_i 𝒘_i^2``.
For uniform weights, this reduces to the length of the vector(s) ``𝒗_j``.

## References
- [Hansen2009](@citet)
"""
struct SilvermanBandwidth <: AbstractBandwidthEstimator end

function bandwidth(::SilvermanBandwidth,
                   data::Tuple{Vararg{AbstractVector,N}},
                   bounds::Tuple{Vararg{Tuple{Any,Any,Boundary.T},N}},
                   weights::Union{Nothing,<:AbstractVector} = nothing
                   ) where {N}
    T = promote_type(map(_unitless∘eltype, data)...)
    lo = map(b -> b[1], bounds)
    hi = map(b -> b[2], bounds)
    neff, Σ = _neff_covar(data, lo, hi, weights)
    # From Hansen (2009) — https://users.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf
    # for a Gaussian kernel:
    # - Table 1:
    #   - R(k) = 1 / 2√π
    #   - κ₂(k) = 1
    #   - ν = 2
    # - Section 2.11 (for uncorrelated dimensions):
    #   - bw_j = σ̂_j C₂(k,d) n^(-1/(4+d))
    #     C₂(k,d) = (4/(2 + d))^(1/(4+d))
    #   - bw_j = σ̂_j * (4/(2 + d)n)^(1/(4+d))
    bw_scale = (oftype(one(T), (4one(T) / (2 + N))) / neff)^(one(T) / (4 + N))
    # where the bandwidth scaling factor is derived for uncorrelated dimensions.
    # Because a non-diagonal covariance matrix can be diagonalized via eigenvalue
    # decomposition, the scaling factor still applies:
    cov_scale = bw_scale ^ 2
    # Σ *= cov_scale
    for jj in 1:N
        for kk in 1:N
            Σ[kk, jj] *= cov_scale
        end
    end
    C = cholesky(Symmetric(Σ, :L), check = false)
    isposdef(C) && return C

    # If the cholesky factorization failed, its presumably due to a degenerate case like
    # zero variance in one of the directions or perfect correlation between two dimensions.
    # Regularize the problem by adding a small perterbation to the diagonal of the
    # covariance matrix, and then try again.

    # ϵ = eps(maximum(Σ[diagind(Σ)]))
    ϵ = floatmin(T)
    for jj in 1:N
        ϵ = max(ϵ, Σ[jj, jj])
    end
    ϵ = eps(ϵ)

    # Σ .+= ϵ * I
    for jj in 1:N
        Σ[jj, jj] += ϵ
    end
    return cholesky!(Symmetric(Σ, :L))
end

# specialize for 1D case where the variance is scalar, so allocating arrays can be avoided
function bandwidth(::SilvermanBandwidth,
                   data::Tuple{AbstractVector},
                   bounds::Tuple{Tuple{Any,Any,Boundary.T}},
                   weights::Union{Nothing,<:AbstractVector} = nothing
                   )
    T = promote_type(map(_unitless∘eltype, data)...)
    lo = (bounds[1][1],)
    hi = (bounds[1][2],)
    neff, Σ = _neff_covar(data, lo, hi, weights)
    # From Hansen (2009) — https://users.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf
    # for a Gaussian kernel:
    # - Table 1:
    #   - R(k) = 1 / 2√π
    #   - κ₂(k) = 1
    #   - ν = 2
    # - Section 2.11 (for uncorrelated dimensions):
    #   - bw_j = σ̂_j C₂(k,d) n^(-1/(4+d))
    #     C₂(k,d) = (4/(2 + d))^(1/(4+d))
    #   - bw_j = σ̂_j * (4/(2 + d)n)^(1/(4+d))
    bw_scale = (oftype(one(T), (4one(T) / 3)) / neff)^(one(T) / 5)
    return iszero(Σ) ? sqrt(eps(bw_scale^2)) : sqrt(Σ) * bw_scale
end

module ISJ
    # An implementation of Brent's method, translated from the algorithm described in
    #   https://en.wikipedia.org/wiki/Brent%27s_method
    #
    # The reference description provides no guidance on the stopping criteria, so we choose to
    # use both relative and absolute tolerances (similar to `isapprox()`).
    function brent(f, a::T, b::T; abstol = nothing, reltol = nothing) where {T}
        δ = isnothing(abstol) ? eps(abs(b - a)) : T(abstol)
        ε = isnothing(reltol) ? eps(abs(b - a)) : T(reltol)
        fa = f(a)
        fb = f(b)
        if fa * fb ≥ zero(T)
            # not a bracketing interval
            return oftype(a, NaN)
        end
        if abs(fa) < abs(fb)
            b, a = a, b
            fb, fa = fa, fb
        end

        c, fc = a, fa
        d = s = b
        mflag = true

        while true
            Δ = abs(b - a)
            if iszero(fb) || Δ <= δ || Δ <= abs(b) * ε
                # converged
                return b
            end

            if fa != fc && fb != fc
                # inverse quadratic interpolation
                s = ( a * fb * fc / ((fa - fb) * (fa - fc))
                    + b * fa * fc / ((fb - fa) * (fb - fc))
                    + c * fa * fb / ((fc - fa) * (fc - fb)))
            else
                # secant
                s = b - fb * (b - a) / (fb - fa)
            end

            u, v = (3a + b) / 4, b
            if u > v
                u, v = v, u
            end
            tol = max(δ, max(abs(b), abs(c), abs(d)) * ε)

            cond1 = !(u < s < v)
            cond23 = abs(s - b) ≥ abs(mflag ? b - c : c - d) / 2
            cond45 = abs(mflag ? b - c : c - d) < tol
            if cond1 || cond23 || cond45
                # bisection
                s = (a + b) / 2
                mflag = true
            else
                mflag = false
            end
            fs = f(s)
            if iszero(fs)
                return s
            end

            c, fc, d = b, fb, c
            if sign(fa) * sign(fs) < zero(T)
                b, fb = s, fs
            else
                a, fa = s, fs
            end

            if abs(fa) < abs(fb)
                b, a = a, b
                fb, fa = fa, fb
            end
        end
    end

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
    function γ(neff::T, f̂::Vector{T}, j::Int, h::T) where {T<:Real}
        N² = ∂ʲ(f̂, h, j + 1)
        fac1 = (T(1) + T(2) ^ -T(j + 0.5)) / 3
        fac2 = mapreduce(T, *, 1:2:(2j-1)) / (sqrt(T(π) / 2) * neff * N²)
        return (fac1 * fac2) ^ (T(1) / (2j + 3))
    end

    # Calculates the iteratively-defined γˡ function, defined in Botev et al, between
    # Eqns. 29 and 30.
    function γˡ(l::Int, neff::T, f̂::Vector{T}, h::T) where {T<:Real}
        for j in l:-1:1
            h = γ(neff, f̂, j, h)
        end
        return h
    end

    # Express the fixed-point equation (Botev et al Eqn. 30) as an expression where the
    # root is the desired bandwidth.
    function estimate(l::Int, neff::T, f̂::Vector{T}, h₀::T) where {T<:Real}
        ξ = ((6sqrt(T(2)) - 3) / 7) ^ (one(T) / 5)
        return brent(eps(h₀), 8h₀) do h
            return h - ξ * γˡ(l, neff, f̂, h)
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

function bandwidth(isj::ISJBandwidth{<:Any},
                   data::Tuple{AbstractVector{T}},
                   bounds::Tuple{Tuple{T, T, Boundary.T}},
                   weights::Union{Nothing, <:AbstractVector} = nothing
                   ) where {T}
    # The Silverman bandwidth estimator should be sufficient to obtain a fine-enough
    # binning that the ISJ algorithm can iterate.
    # We need a histogram, so just reuse the binning base case of the estimator pipeline
    # to provide what we need.
    (x, f), info = estimate(isj.binning, data[1], weights; bounds, isj.bwratio,
                            bandwidth = SilvermanBandwidth())

    bandwidth = info.bandwidth[1]
    neff = info.neffective
    Δx = (tmp = step(x); tmp / oneunit(tmp))
    # The core of the ISJ algorithm works in a normalized unit system where Δx = 1.
    # Two things of note:
    #
    #   1. We initialize the fixed-point algorithm with the Silverman bandwidth, but
    #      scaled correctly for the change in axis. Then afterwards, the ISJ bandwidth
    #      will need to be scaled back to the original axis, e.g. h → Δx × h
    h₀ = bandwidth / Δx
    #   2. Via the Fourier scaling theorem, f(x / Δx) ⇔ Δx × f̂(k), we must scale the DCT
    #      by the grid step size.
    f̂ = similar(f, _unitless(T))
    @simd for I in eachindex(f)
        @inbounds f̂[I] = f[I] * oneunit(T)
    end
    FFTW.r2r!(f̂, FFTW.REDFT10)
    rmul!(f̂, Δx)

    # Now we simply solve for the fixed-point solution:
    h = Δx * ISJ.estimate(isj.niter, neff, f̂, h₀)

    # Check that the fixed-point solver converged to a positive value
    if isnan(h) || h < zero(h)
        if isj.fallback
            h = bandwidth  # fallback to the Silverman estimate
        else
            throw(ErrorException("ISJ estimator failed to converge. More data is needed."))
        end
    end
    return h
end
