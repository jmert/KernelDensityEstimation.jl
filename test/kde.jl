using .KDE: estimate

using FFTW
using Statistics: std, var
using Random: Random, randn

edges2centers(r) = r[2:end] .- step(r) / 2
centers2edges(r) = (Δ = step(r) / 2; range(first(r) - Δ, last(r) + Δ, length = length(r) + 1))

gaussian(x, σ) = exp.(-x .^ 2 ./ 2σ^2) ./ sqrt(2oftype(σ, π) * σ^2)

# do this only once, and take subsets as needed
rv_norm_σ = 2.1
rv_norm_long = rv_norm_σ .* randn(Random.seed!(Random.default_rng(), 1234), Int(1e8))


@testset "Options Handling" begin
    import .KDE: init

    # Make sure the option processing is fully inferrable
    WT = @NamedTuple{lo::Any, hi::Any, nbins::Any, boundary::Any, bandwidth::Any, bwratio::Any}
    #  don't cover too many variations, since that just adds to the test time
    variations = [WT((lo, hi, nbins, boundary, bandwidth, bwratio)) for
        lo in (nothing, -1),
        hi in (nothing,),
        nbins in (nothing, 10),
        boundary in (:open, KDE.Closed),
        bandwidth in (KDE.SilvermanBandwidth(), 1.0),
        bwratio in (1,)
    ]
    method = KDE.LinearBinning()

    RT = Tuple{Vector{Float64}, KDE.UnivariateKDEInfo{Float64}}
    @testset "options = $kws" for kws in variations
        @test @inferred(init(method, [1.0]; kws...)) isa RT
    end
    # also test that default values work
    @test @inferred(init(method, [1.0], bandwidth = 1.0)) isa RT

    # Unused options that make it down to the option processing step log a warning message
    @test_logs((:warn, "Unused keyword argument(s)"),
               KDE.init(method, [1.0], bandwidth = 1.0, unusedarg=true))
end

@testset "Bounds Handling" begin
    # direct boundary specifications
    @test KDE.boundary(KDE.Open)        == KDE.boundary(:open)        == KDE.Open
    @test KDE.boundary(KDE.Closed)      == KDE.boundary(:closed)      == KDE.Closed
    @test KDE.boundary(KDE.ClosedLeft)  == KDE.boundary(:closedleft)  == KDE.ClosedLeft
    @test KDE.boundary(KDE.ClosedRight) == KDE.boundary(:closedright) == KDE.ClosedRight
    @test KDE.boundary(KDE.OpenLeft)    == KDE.ClosedRight
    @test KDE.boundary(KDE.OpenRight)   == KDE.ClosedLeft
    @test_throws(ArgumentError(match"Unknown boundary option: .*?"r), KDE.boundary(:something))

    # inferring boundary specifications
    @test KDE.boundary((-Inf, Inf)) == KDE.Open
    @test KDE.boundary((0.0, Inf))  == KDE.ClosedLeft
    @test KDE.boundary((-Inf, 0))   == KDE.ClosedRight
    @test KDE.boundary((0, 1))      == KDE.Closed
    for bnds in [(NaN, 1.0), (Inf, 1.0), (0.0, NaN), (0.0, -Inf)]
        @test_throws(ArgumentError(match"Could not infer boundary for `lo = .*?`, `hi = .*?`"r),
                     KDE.boundary(bnds))
    end


    v = [1.0, 2.0]
    kws = (; lo = 0.0, hi = 5.0, nbins = 5, bandwidth = 1.0, bwratio = 1)

    # closed boundaries -- histogram cells will span exactly lo/hi input
    x = range(kws.lo, kws.hi, length = kws.nbins + 1)
    k, info = estimate(KDE.HistogramBinning(), v; boundary = :closed, kws...)
    @test info.boundary === KDE.Closed
    @test length(k.x) == kws.nbins
    @test step(k.x) == step(x)
    @test centers2edges(k.x) == x

    # open boundaries --- histogram extends to 8x further left/right than lo/hi
    x = range(kws.lo - 8kws.bandwidth, kws.hi + 8kws.bandwidth, length = kws.nbins + 1)
    k, info = estimate(KDE.HistogramBinning(), v; boundary = :open, kws...)
    @test info.boundary === KDE.Open
    @test length(k.x) == kws.nbins
    @test step(k.x) == step(x)
    @test centers2edges(k.x) == x

    # half-open boundaries --- either left or right side is extended, while the other is
    # exact
    x = range(kws.lo, kws.hi + 8kws.bandwidth, length = kws.nbins + 1)
    k, info = estimate(KDE.HistogramBinning(), v; boundary = :closedleft, kws...)
    @test info.boundary === KDE.ClosedLeft
    @test length(k.x) == kws.nbins
    @test step(k.x) == step(x)
    @test centers2edges(k.x) == x

    x = range(kws.lo - 8kws.bandwidth, kws.hi, length = kws.nbins + 1)
    k, info = estimate(KDE.HistogramBinning(), v; boundary = :closedright, kws...)
    @test info.boundary === KDE.ClosedRight
    @test length(k.x) == kws.nbins
    @test step(k.x) == step(x)
    @test centers2edges(k.x) == x

    # test that the half-open aliases work
    k, info = estimate(KDE.HistogramBinning(), v; boundary = :openleft, kws...)
    @test info.boundary === KDE.ClosedRight
    k, info = estimate(KDE.HistogramBinning(), v; boundary = :openright, kws...)
    @test info.boundary === KDE.ClosedLeft

    # test that bounds= works the same as lo=,hi=,boundary=
    kws′ = (; kws.nbins, kws.bandwidth, kws.bwratio)
    k1, info = estimate(KDE.HistogramBinning(), v; boundary = :closed, kws...)
    k2, info = estimate(KDE.HistogramBinning(), v; bounds = (kws.lo, kws.hi, :closed), kws′...)
    @test k1.x == k2.x && k1.f == k2.f
    # bounds=(lo, hi) is interpreted as bounds=(lo, hi, :open)
    k1, info = estimate(KDE.HistogramBinning(), v; boundary = :open, kws...)
    k2, info = estimate(KDE.HistogramBinning(), v; bounds = (kws.lo, kws.hi), kws′...)
    @test k1.x == k2.x && k1.f == k2.f

    # bounds= argument overrides lo=,hi=,boundary= and warns
    M = KDE.LinearBinning()
    @test_logs((:warn, "Keyword `bounds` is overriding non-nothing `lo` and/or `hi`"),
               KDE.init(M, [1.0]; bounds = (0.0, 1.0, :open), lo = -1.0, bandwidth = 1.0))
    @test_logs((:warn, "Keyword `bounds` is overriding non-nothing `lo` and/or `hi`"),
               KDE.init(M, [1.0]; bounds = (0.0, 1.0, :open), hi = 2.0, bandwidth = 1.0))
end

@testset "Simple Binning" begin
    kws = (; bandwidth = KDE.SilvermanBandwidth(), bwratio = 1)
    # raw data
    v₁ = Float64[0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 4.5]
    # non-zero elements of the probability **mass** function
    #   N.B. KDE returns probability **density** function
    h₁ = [1, 2, 3, 2, 1] ./ length(v₁)
    @test sum(h₁) ≈ 1.0
    # bandwidth estimate
    ν, σ̂ = length(v₁), std(v₁, corrected = false)
    b₀ = σ̂ * (4//3 / ν)^(1/5)  # Silverman's rule

    # prepared using known limits and number of bins
    k₁, info = estimate(KDE.HistogramBinning(), v₁; boundary = :closed,
                        lo = 0, hi = 5, nbins = 5, kws...)
    Δx = step(k₁.x)
    @test k₁.x == edges2centers(0.0:1.0:5.0)
    @test isapprox(k₁.f .* Δx, h₁)
    @test sum(k₁.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # prepared with known number of bins, but automatically selected bounds
    k₂, info = estimate(KDE.HistogramBinning(), v₁; boundary = :closed, nbins = 5, kws...)
    Δx = step(k₂.x)
    @test k₂.x == edges2centers(range(0.5, 4.5, length = 5 + 1))
    @test isapprox(k₂.f .* Δx, h₁)
    @test sum(k₂.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # prepared with unknown number of bins, but known bounds
    k₃, info = estimate(KDE.HistogramBinning(), v₁; boundary = :closed, lo = 0, hi = 5,
                        kws...)
    Δx = step(k₃.x)
    @test k₃.x == edges2centers(range(0.0, 5.0, length = round(Int, 5 / b₀) + 1))
    @test isapprox(filter(!iszero, k₃.f) .* Δx, h₁)
    @test sum(k₃.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # prepared with unknown limits and number of bins
    k₄, info = estimate(KDE.HistogramBinning(), v₁; boundary = :closed, kws...)
    Δx = step(k₄.x)
    @test isapprox(filter(!iszero, k₄.f) .* Δx, h₁)
    @test sum(k₄.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # prepared using an alternative sampling density (compared to previous)
    k₅, info = estimate(KDE.HistogramBinning(), v₁; boundary = :closed, kws..., bwratio = 16)
    Δx = step(k₅.x)
    @test step(k₅.x) < step(k₄.x)
    @test isapprox(filter(!iszero, k₅.f) .* Δx, h₁)
    @test sum(k₅.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # make sure errors do not occur when uniform data is provided
    let (k, info) = estimate(KDE.HistogramBinning(), ones(100); boundary = :closed, kws...)
        @test collect(k.x) == [1.0]
        @test isfinite(info.bandwidth) && !iszero(info.bandwidth)
        @test step(k.x) == 0.0  # zero-width bin
        @test sum(k.f) == 1.0  # like a Kronecker delta
    end

    # make sure the bandwidth argument is converted to the appropriate common type
    _, info = estimate(KDE.HistogramBinning(), v₁; kws...)
    @test info.bandwidth isa Float64
    _, info = estimate(KDE.HistogramBinning(), v₁; bandwidth = 1)  # explicit but not same type
    @test info.bandwidth isa Float64
    _, info = estimate(KDE.HistogramBinning(), Float32.(v₁); kws...)
    @test info.bandwidth isa Float32
    _, info = estimate(KDE.HistogramBinning(), Float32.(v₁); bandwidth = 1)
    @test info.bandwidth isa Float32


    expT = Tuple{KDE.UnivariateKDE{Float64,<:AbstractRange{Float64},<:AbstractVector{Float64}},
                 KDE.UnivariateKDEInfo{Float64}}
    @test @inferred(estimate(KDE.HistogramBinning(), [1.0, 2.0]; nbins = 2, kws...)) isa expT
end

@testset "Histogramming Accuracy" begin
    # This range is an example taken from Julia's test suite for its range handling,
    # called out as having challenging values
    r = 1.0:1/49:27.0
    v = Vector(r)

    # For regular histogram binning, using the bin edges as values must result in a uniform
    # distribution except the last bin which is doubled (due to being closed on the right).
    ν, H = KDE._kdebin(KDE.HistogramBinning(), v, first(r), last(r), length(r) - 1)
    @test ν == length(r)
    @test all(@view(H[1:end-1]) .== H[1])
    @test H[end] == 2H[1]

    # For linear binning, the first and last bins differ from the rest, getting all of the
    # weight from the two edges but also gaining half a contribution from their (only)
    # neighbors. (The remaining interior bins give up half of their weight but
    # simultaneously gain from a neighbor, so they are unchanged.)
    ν, H = KDE._kdebin(KDE.LinearBinning(), v, first(r), last(r), length(r) - 1)
    @test ν == length(r)
    @test all(@view(H[2:end-1]) .≈ H[2])
    @test H[end] ≈ H[1]
    @test H[1] ≈ 1.5H[2]

    # A case where naively calculating the cell index and weight factors suffers from the
    # limits of finite floating point calculations, e.g.
    #
    #     zz = (x - lo) / Δx  # fractional index
    #     ii = trunc(Int, zz) - (x == hi)  # bin index, including right-closed last bin
    #     ww = (zz - ii) - 0.5
    #
    # results in ww ≈ 1.5 due to the value of zz not being an integer despite x == hi
    lo = 0.005653766369679568
    hi = x = 0.006728850177869153
    nbins = 122

    _, H = KDE._kdebin(KDE.LinearBinning(), [x], lo, hi, nbins)
    @test all(iszero, @view H[1:end-1])
    @test H[end] > 0.0
end

@testset "Basic Kernel Density Estimate" begin
    nbins = 11  # use odd value to have symmetry
    nsamp = 10nbins
    Δx = 1.0 / nbins

    v_uniform = range(-1, 1, length = nsamp)
    p_uniform = 1.0 / nbins

    # Tune the bandwidth to be much smaller than the bin sizes, which effectively means
    # we just get back the internal histogram (when performing only a basic KDE).
    (x, f), _ = estimate(KDE.BasicKDE(), v_uniform, boundary = :closed, nbins = nbins, bandwidth = 0.1Δx)
    @test all(isapprox.(f .* step(x), p_uniform, atol = 1e-3))
    # Then increase the bandwidth to be the same size of the bins. The outermost bins will
    # impacted by the kernel convolving with implicit zeros beyond the edges of the
    # distribution, so we should expect a significant decrease in the first and last bin.
    (x, f), _ = estimate(KDE.BasicKDE(), v_uniform, boundary = :closed, nbins = nbins, bandwidth = Δx)
    @test all(isapprox.(f[2:end-1] .* step(x), p_uniform, atol = 1e-3))  # approximate p_uniform
    @test all(<(-1e-3), f[[1,end]] .* step(x) .- p_uniform) # systematically less than p_uniform

    # If we put a Kronecker delta in the center of a span with a wide (lo, hi) spread,
    # we should effectively just get back the convolution kernel.
    v_kron = zeros(100)
    bw = 1.0

    (x, f1), _ = estimate(KDE.BasicKDE(), v_kron; boundary = :closed, lo = -6bw, hi = 6bw, nbins = 251, bandwidth = bw)
    (x, f2), _ = estimate(KDE.BasicKDE(), v_kron; boundary = :closed, lo = -6bw, hi = 6bw, nbins = 251, bandwidth = bw)
    g = exp.(.-(x ./ bw) .^2 ./ 2) ./ (bw * sqrt(2π)) .* step(x)
    @test all(isapprox.(f1 .* step(x), g, atol = 1e-5))
    @test all(isapprox.(f2 .* step(x), g, atol = 1e-5))

    kws = (; bandwidth = KDE.SilvermanBandwidth())

    # On an open interval, probability should be conserved.
    k, _ = estimate(KDE.BasicKDE(), v_uniform; boundary = :open, kws...)
    @test sum(k.f) * step(k.x) ≈ 1.0

    # But on (semi-)closed intervals, the norm will drop
    kl, _ = estimate(KDE.BasicKDE(), v_uniform; boundary = :closedleft, kws...)
    nl = sum(kl.f) * step(kl.x)
    @test nl < 0.96

    kr, _ = estimate(KDE.BasicKDE(), v_uniform; boundary = :closedright, kws...)
    nr = sum(kr.f) * step(kr.x)
    @test nr < 0.96
    @test nl ≈ nr  # should be nearly symmetric

    kc, _ = estimate(KDE.BasicKDE(), v_uniform; boundary = :closed, kws...)
    nc = sum(kc.f) * step(kc.x)
    @test nc < 0.91
end

@testset "Linear Boundary Correction" begin
    nbins = 11  # use odd value to have symmetry
    nsamp = 10nbins
    Δx = 1.0 / nbins

    v_uniform = range(-1, 1, length = nsamp)
    p_uniform = 1.0 / nbins

    # Enable the normalization option, which makes a correction for the implicit zeros
    # being included in the convolution
    (x, f), _ = estimate(KDE.LinearBoundaryKDE(), v_uniform, boundary = :closed, nbins = nbins, bandwidth = Δx)
    @test all(isapprox.(f .* step(x), p_uniform, atol = 1e-3))  # approximately p_uniform
    # and that correction keeps working as the bandwidth causes multiple bins to be affected
    (x, f), _ = estimate(KDE.LinearBoundaryKDE(), v_uniform, boundary = :closed, nbins = nbins, bandwidth = 6Δx)
    @test all(isapprox.(f .* step(x), p_uniform, atol = 1e-3))  # approximately p_uniform

    # The boundary correction only applies when a hard boundary is encountered, so the
    # BasicKDE and LinearBoundaryKDE should provide very similar distributions when
    # the boundary is Open.
    (x₁, f₁), _ = estimate(KDE.BasicKDE(), v_uniform, boundary = :open, nbins = nbins, bandwidth = 2Δx)
    (x₂, f₂), _ = estimate(KDE.LinearBoundaryKDE(), v_uniform, boundary = :open, nbins = nbins, bandwidth = 2Δx)
    @test all(isapprox.(f₁, f₂, atol = 1e-3))
end

@testset verbose = true "Bandwidth Estimators" begin
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
        # Test that the estimator approximately matches the asymptotic behavior for a
        # the known Gaussian distribution behavior.
        #   N.B. use very large numbers to reduce sample variance
        @testset "AMISE with N = $N" for N in Int.((1e6, 1e7, 1e8))
            atol = (sqrt(eps(1.0)) / N) ^ (1 // 5)  # scale error similarly to AMISE
            σ, v = rv_norm_σ, view(rv_norm_long, 1:N)
            t = G_amise(N, σ)
            h = KDE.bandwidth(KDE.SilvermanBandwidth(), v, -6σ, 6σ, KDE.Open)
            @test t ≈ h atol=atol
        end
    end # Silverman Bandwidth

    @testset "ISJ Bandwidth" begin
        import .KDE: ISJBandwidth, SilvermanBandwidth

        @testset "Internals" begin
            # Check that the implementation of _ISJ.∂ʲ agrees with the analytic answer we
            # can derive for a Gaussian distribution
            h₀, σ₀, npts = 1.0, 2.1, 512  # arbitrary
            x = range(-6σ₀, 6σ₀, length = npts)
            Δx, g = step(x), gaussian(x, σ₀)
            # scale everything to effectively be Δx == 1, to match the internal assumption of the
            # implementation of ∂ʲ
            f̂ = FFTW.r2r(g, FFTW.REDFT10) .* Δx
            h, σ = h₀ / Δx, σ₀ / Δx
            for j in 0:7
                @test KDE._ISJ.∂ʲ(f̂, h, j) ≈ norm_G_conv_djK_dxj(σ, h, j) atol=sqrt(eps(1.0))
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
        h₁ = KDE.init(KDE.MultiplicativeBiasKDE(), v; bandwidth, bounds)[2].bandwidth
        @test h₁ / h₀ ≈ N ^ (1//5 - 1//9)
    end
end  # Bandwidth Estimators

@testset "Type Handling" begin
    rv = view(rv_norm_long, 1:100)
    K1 = kde(rv)

    # Equality and hashing
    K2 = kde(rv)
    @test K1 !== K2
    @test K1 == K2
    @test hash(K1) == hash(K2)
    K3 = kde(rv[1:50])
    @test K1 !== K3
    @test K1 != K3
    @test hash(K1) != hash(K3)

    K4 = kde(Float32.(rv))
    @test eltype(K1) == eltype(K1.x) == eltype(K1.f) == Float64
    @test eltype(K4) == eltype(K4.x) == eltype(K4.f) == Float32
end

@testset "Show" begin
    K, info = estimate(KDE.BasicKDE(), @view rv_norm_long[1:100])
    buf = IOBuffer()

    @testset "UnivariateKDE" begin
        # Check that compact-mode printing is shorter than full printing
        show(IOContext(buf, :compact => true), K)
        shortmsg = String(take!(buf))
        show(IOContext(buf, :compact => false), K)
        longmsg = String(take!(buf))
        @test length(shortmsg) < length(longmsg)
        @test occursin("…", shortmsg) && !occursin("…", longmsg)

        # For the longer output, there's both a limited and unlimited variation. Again,
        # check that the lengths differ as expected.
        show(IOContext(buf, :compact => false, :limit => true), K)
        limitlongmsg = String(take!(buf))
        @test length(shortmsg) < length(limitlongmsg)
        @test length(limitlongmsg) < length(longmsg)

        # Verify the headers appear as expected
        @test occursin("UnivariateKDE{Float64}", shortmsg)
        @test occursin("UnivariateKDE{Float64,", longmsg)  # full parametric typing
        @test occursin("UnivariateKDE{Float64}", limitlongmsg)
    end

    @testset "UnivariateKDEInfo" begin
        # Check that compact-mode printing is shorter than full printing
        show(IOContext(buf, :compact => true), info)
        shortmsg = String(take!(buf))
        show(IOContext(buf, :compact => false), info)
        longmsg = String(take!(buf))
        @test length(shortmsg) < length(longmsg)
        @test occursin("…", shortmsg) && !occursin("…", longmsg)

        # For the longer output, there's both a limited and unlimited variation. Again,
        # check that the lengths differ as expected.
        show(IOContext(buf, :compact => false, :limit => true), info)
        limitlongmsg = String(take!(buf))
        @test length(shortmsg) < length(limitlongmsg)
        @test length(limitlongmsg) < length(longmsg)

        # Verify the headers appear as expected
        @test occursin("UnivariateKDEInfo{Float64}", shortmsg)
        @test occursin("UnivariateKDEInfo{Float64}", longmsg)  # full parametric typing
        @test occursin("UnivariateKDEInfo{Float64}", limitlongmsg)


        # The three-arg version for text/plain output is more sophisticated
        ioc = IOContext(buf, :displaysize => (50, 120), :limit => true)
        show(ioc, MIME"text/plain"(), info)
        pretty = String(take!(buf))
        plines = split(strip(pretty), '\n')
        header = popfirst!(plines)
        # heading line contains type info
        @test endswith(header, "UnivariateKDEInfo{Float64}:")
        # fields are then properly padded
        alignment = map(l -> findfirst(==(':'), l), plines)
        @test all(==(alignment[1]), alignment)
        # the sentinel width is properly kept up-to-date
        padding = map(l -> findfirst(!=(' '), l) - 1, plines)
        @test minimum(padding) == 2
    end
end
