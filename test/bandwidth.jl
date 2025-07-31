using FFTW
using LinearAlgebra: I
using Random: Random, randn
using Statistics: std, var

gaussian(x, σ) = exp.(-x .^ 2 ./ 2σ^2) ./ sqrt(2oftype(σ, π) * σ^2)

# do this only once, and take subsets as needed
rv_norm_σ = 2.1
rv_norm_long = rv_norm_σ .* randn(Random.seed!(Random.default_rng(), 1234), Int(1e8))


# The norm of convolved derivatives can be analytically expressed when a Gaussian is
# convolving a Gaussian.
#
#=

Let ``σ`` be the standard deviation of the target distribution and ``h`` be the
bandwidth of the Gaussian convolution kernel.
The target distribution ``d(x) = e^{-x^2/2σ^2} / \sqrt{2πσ^2}`` is known to have a
Fourier transform which is ``\mathcal{F}[d(x)](k) = e^{-2π^2 σ^2 k^2}``, which we can
directly plug in for the ``\hat{d}(k)`` term:

```math
\begin{align*}
    \left\lVert \tilde{d}_h^{(j)} \right\rVert^2
    &= \int_{-∞}^∞ \left\lVert \hat{d}(k) \right\rVert^2 (2πk)^{2j} e^{-4π^2 h^2 k^2} \,dk
    \\
    &= \int_{-∞}^∞ (2πk)^{2j} e^{-4π^2 (σ^2 + h^2) k^2} \,dk
    \\
    &= \frac{1}{2π} \sqrt{\frac{π}{σ^2 + h^2}}
        \frac{(2j - 1)!!}{2^j \, (σ^2 + h^2)^j}
\end{align*}
```
(trivially for ``j > 0``, and for ``j == 0`` when extended to negative odd integers such
that ``(2j - 1)!! = -1!! = 1``).
=#
function norm_G_conv_djK_dxj(σ, h, j)
    σ, h = promote(σ, h)
    F = reduce(*, 1:2:(2j - 1))
    v2 = σ^2 + h^2
    let π = oftype(σ, π)
        return sqrt(π / v2) * F / (2π * (2v2)^j)
    end
end

# The norm of a Gaussian derivative can be directly calculated by simply setting the
# convolution kernel's width to 0 (equivalent to convolving with a Dirac delta
# function), and therefore we can reuse the previous function to get the norm of
# any arbitrary derivative as well.
norm_djG_dxj(σ, j) = norm_G_conv_djK_dxj(σ, zero(σ), j)

# From Eqn 38 in Appendix A of Botev et al (2010), the Asymptotic Mean Integrated
# Squared Error (AMISE) for a Gaussian kernel estimator has a bandwidth which
# approaches
#
#=
Let ``f`` be a Gaussian of standard deviation ``σ``. Then the AMISE is

\begin{align*}
    h = \left( \frac{1}{2ν √π \lVert f'' \rVert^2} \right)^{1/5}
\end{align*}

where it should be noted that h = √t_* as written in Botev et al.
=#
#
function G_amise(ν, σ)
    return inv(2ν * sqrt(oftype(σ, π)) * norm_djG_dxj(σ, 2)) ^ (1 // 5)
end

@testset "Silverman Bandwidth" begin
    σ = rv_norm_σ

    @testset "Variance Estimation" begin
        # Check the implementation of the standard deviation and effective sample size
        # calculation
        v = view(rv_norm_long, 1:1_000_000)
        v1 = (v,)
        lo = (-10σ,)
        hi = (+10σ,)
        neff1, var1 = @inferred KDE._neff_covar(v1, lo, hi, nothing)
        @test neff1 == length(v)
        @test var1 ≈ σ^2 atol = 16.0 / sqrt(neff1)
        # weights = nothing is same as uniform (all ones)
        neff2, var2 = @inferred KDE._neff_covar(v1, lo, hi, fill(1.0, length(v)))
        @test neff2 == neff1
        @test var2 == var1

        # 1D is special-cased
        #   It should not allocate
        if VERSION >= v"1.12.0-beta3"
            @test (@allocated KDE._neff_covar(v1, lo, hi, nothing)) == 0
        else
            @test_broken (@allocated KDE._neff_covar(v1, lo, hi, nothing)) == 0
        end
        # 1D is special cased — verify that the general case matches
        gensig = Tuple{#=coords=# Tuple{Vararg{AbstractVector,N}},
                       #=lo=# Tuple{Vararg{Any,N}},
                       #=hi=# Tuple{Vararg{Any,N}},
                       #=weights=# Nothing
                       } where N
        neff3, var3 = invoke(KDE._neff_covar, gensig, v1, lo, hi, nothing)
        @test neff3 == neff1
        @test var3 isa AbstractMatrix{Float64}
        @test only(var3) == var1


        # multidimensional covariances

        # using the same data multiple times is perfect correlation, so all entries
        # in the covariance matrix are identical (and equal to the variance of the data)
        #   2x2
        neff, covar = KDE._neff_covar((v, v), -10σ.*(1,1), 10σ.*(1,1), nothing)
        @test neff == length(v)
        @test covar == var1 .* ones(2, 2)
        #   3x3
        neff, covar = KDE._neff_covar((v, v, v), -10σ.*(1,1,1), 10σ.*(1,1,1), nothing)
        @test neff == length(v)
        @test covar == var1 .* ones(3, 3)
        # circularly-shifting one of the inputs decorrelates the inputs, so then the
        # covariance matrix should be approximately diagonal
        w = circshift(v, -1)
        #   2x2
        neff, covar = KDE._neff_covar((v, w), -10σ.*(1,1), 10σ.*(1,1), nothing)
        @test neff == length(v)
        @test covar ≈ var1 * I  atol=16/sqrt(length(v))
        #   3x3 (not diagonal)
        neff, covar = KDE._neff_covar((v, v, w), -10σ.*(1,1,1), 10σ.*(1,1,1), nothing)
        @test neff == length(v)
        Σ = [1 1 0
             1 1 0
             0 0 1] .* var1
        @test covar ≈ Σ  atol=16/sqrt(length(v))
    end

    # Test that the estimator approximately matches the asymptotic behavior for a
    # the known Gaussian distribution behavior.
    #   N.B. use very large numbers to reduce sample variance
    @testset "AMISE with N = $N" for N in Int.((1e6, 1e7, 1e8))
        atol = (sqrt(eps(1.0)) / N) ^ (1 // 5)  # scale error similarly to AMISE
        v = view(rv_norm_long, 1:N)
        t = G_amise(N, σ)
        h = KDE.bandwidth(KDE.SilvermanBandwidth(), v, -6σ, 6σ, KDE.Open)
        @test t ≈ h atol=atol
    end

    # Verify that the bandwidth estimator respects the lo/hi limits and excludes
    # out-of-bounds elements
    v = view(rv_norm_long, 1:256)
    h₀ = KDE.bandwidth(KDE.SilvermanBandwidth(), v, -6σ, 6σ, KDE.Closed)
    h₁ = KDE.bandwidth(KDE.SilvermanBandwidth(), v, 0.0, 6σ, KDE.Closed)
    let z = v, γ = (4 // 3length(z))^(1 // 5)
        @test h₀ ≈ γ * std(z, corrected = false)
    end
    let z = filter(>=(0), v), γ = (4 // 3length(z))^(1 // 5)
        @test h₁ ≈ γ * std(z, corrected = false)
    end

    @testset "Unitful numbers" begin
        σ = Quantity(rv_norm_σ, u"m")
        v = Quantity.(view(rv_norm_long, 1:100), u"m")
        @test (@inferred KDE.bandwidth(KDE.SilvermanBandwidth(), v, -6σ, 6σ, KDE.Open)) isa eltype(v)
    end
end # Silverman Bandwidth

@testset "ISJ Bandwidth" begin
    import .KDE: ISJBandwidth, SilvermanBandwidth

    @testset "Internals" begin
        # Check that the implementation of ISJ.∂ʲ agrees with the analytic answer we
        # can derive for a Gaussian distribution
        h₀, σ₀, npts = 1.0, 2.1, 512  # arbitrary
        x = range(-6σ₀, 6σ₀, length = npts)
        Δx, g = step(x), gaussian(x, σ₀)
        # scale everything to effectively be Δx == 1, to match the internal assumption of the
        # implementation of ∂ʲ
        f̂ = FFTW.r2r(g, FFTW.REDFT10) .* Δx
        h, σ = h₀ / Δx, σ₀ / Δx
        for j in 0:7
            @test KDE.ISJ.∂ʲ(f̂, h, j) ≈ norm_G_conv_djK_dxj(σ, h, j) atol=sqrt(eps(1.0))
        end
    end

    # Test that the estimator approximately matches the asymptotic behavior for a
    # the known Gaussian distribution behavior.
    #   N.B. use very large numbers to reduce sample variance
    @testset "AMISE with N = $N" for N in Int.((1e6, 1e7, 1e8))
        atol = (sqrt(eps(1.0)) / N) ^ (1 // 5)  # scale error similarly to AMISE
        σ, v = rv_norm_σ, view(rv_norm_long, 1:N)
        t = G_amise(N, σ)
        h = KDE.bandwidth(ISJBandwidth(), v, -6σ, 6σ, KDE.Open)
        @test t ≈ h atol=atol
    end

    # Given a bounded (truncated) distribution, the estimator should give a bandwidth
    # smaller if the boundary is incorrectly declared as open versus [half-]closed.
    let σ = rv_norm_σ
        v = filter(>(0), view(rv_norm_long, 1:20_000))
        open_h = KDE.bandwidth(ISJBandwidth(), v, 0.0, 6.0σ, KDE.Open)
        close_h = KDE.bandwidth(ISJBandwidth(), v, 0.0, 6.0σ, KDE.ClosedLeft)
        @test 5open_h < close_h
    end

    # The ISJ estimator fails in some cases. Easy examples are:
    #   1. Very small data sets.
    args = ([1.0, 1.1], 0.0, 2.0, KDE.Open)
    @test KDE.bandwidth(ISJBandwidth(fallback = true), args...) ==
          KDE.bandwidth(SilvermanBandwidth(), args...)
    @test_throws(ErrorException("ISJ estimator failed to converge. More data is needed."),
                 KDE.bandwidth(ISJBandwidth(fallback = false), args...))
    #   2. A uniform distribution on a finite interval.
    args = (collect(0.0:0.1:1.0), 0.0, 1.0, KDE.Closed)
    @test KDE.bandwidth(ISJBandwidth(fallback = true), args...) ==
          KDE.bandwidth(SilvermanBandwidth(), args...)
    @test_throws(ErrorException("ISJ estimator failed to converge. More data is needed."),
                 KDE.bandwidth(ISJBandwidth(fallback = false), args...))

    @testset "Unitful numbers" begin
        σ = Quantity(rv_norm_σ, u"m")
        v = Quantity.(view(rv_norm_long, 1:100), u"m")
        @test (@inferred KDE.bandwidth(KDE.ISJBandwidth(fallback = false), v, -6σ, 6σ, KDE.Open)) isa eltype(v)
    end
end  # ISJ Bandwidth


# "normal" estimators have a bias that scales like O(h^2); the histogramming steps are
# given a sentinel of 0 in order to indicate them as fundamentally different, but the
# difference has no impact on the automatic bandwidth.
@test KDE.estimator_order(KDE.HistogramBinning) == 0
@test KDE.estimator_order(KDE.LinearBinning) == 0
@test KDE.estimator_order(KDE.BasicKDE) == 1
@test KDE.estimator_order(KDE.LinearBoundaryKDE) == 1

# The bandwidth in the initialization step is made wider for the MultiplicativeBiasKDE
# method due to it being a higher-order estimator.
@test KDE.estimator_order(KDE.MultiplicativeBiasKDE) == 2
let N = 1_000, σ = rv_norm_σ, v = view(rv_norm_long, 1:N),
    bandwidth = KDE.SilvermanBandwidth(), bounds = (-6σ, 6σ, KDE.Open)
    h₀ = KDE.bandwidth(bandwidth, v, bounds...)
    h₁ = KDE.init(KDE.MultiplicativeBiasKDE(), v; bandwidth, bounds)[3].bandwidth[1]
    @test h₁ / h₀ ≈ N ^ (1//5 - 1//9)
end
