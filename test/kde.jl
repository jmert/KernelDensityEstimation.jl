using KernelDensityEstimation
const KDE = KernelDensityEstimation

using Statistics: std, var
using Random: rand

edges2centers(r) = r[2:end] .- step(r) / 2
centers2edges(r) = (Δ = step(r) / 2; range(first(r) - Δ, last(r) + Δ, length = length(r) + 1))

@testset "Bounds Handling" begin
    v = [1.0, 2.0]
    kws = (; lo = 0.0, hi = 5.0, nbins = 5, bandwidth = 1.0, bwratio = 1)

    # closed boundaries -- histogram cells will span exactly lo/hi input
    x = range(kws.lo, kws.hi, length = kws.nbins + 1)
    k, info = kde(KDE.HistogramBinning(), v; cover = KDE.Closed, kws...)
    @test info.cover === KDE.Closed
    @test length(k.x) == kws.nbins
    @test step(k.x) == step(x)
    @test centers2edges(k.x) == x

    # open boundaries --- histogram extends to 4x further left/right than lo/hi
    x = range(kws.lo - 4kws.bandwidth, kws.hi + 4kws.bandwidth, length = kws.nbins + 1)
    k, info = kde(KDE.HistogramBinning(), v; cover = KDE.Open, kws...)
    @test info.cover === KDE.Open
    @test length(k.x) == kws.nbins
    @test step(k.x) == step(x)
    @test centers2edges(k.x) == x

    # half-open boundaries --- either left or right side is extended, while the other is
    # exact
    x = range(kws.lo, kws.hi + 4kws.bandwidth, length = kws.nbins + 1)
    k, info = kde(KDE.HistogramBinning(), v; cover = KDE.ClosedLeft, kws...)
    @test info.cover === KDE.ClosedLeft
    @test length(k.x) == kws.nbins
    @test step(k.x) == step(x)
    @test centers2edges(k.x) == x

    x = range(kws.lo - 4kws.bandwidth, kws.hi, length = kws.nbins + 1)
    k, info = kde(KDE.HistogramBinning(), v; cover = KDE.ClosedRight, kws...)
    @test info.cover === KDE.ClosedRight
    @test length(k.x) == kws.nbins
    @test step(k.x) == step(x)
    @test centers2edges(k.x) == x

    # test that the half-open aliases work
    k, info = kde(KDE.HistogramBinning(), v; cover = KDE.OpenLeft, kws...)
    @test info.cover === KDE.ClosedRight
    k, info = kde(KDE.HistogramBinning(), v; cover = KDE.OpenRight, kws...)
    @test info.cover === KDE.ClosedLeft
end

@testset "Simple Binning" begin
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
    k₁, info = kde(KDE.HistogramBinning(), v₁; cover = KDE.Closed, lo = 0, hi = 5, nbins = 5)
    Δx = step(k₁.x)
    @test k₁.x == edges2centers(0.0:1.0:5.0)
    @test isapprox(k₁.f .* Δx, h₁)
    @test sum(k₁.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # prepared with known number of bins, but automatically selected bounds
    k₂, info = kde(KDE.HistogramBinning(), v₁; cover = KDE.Closed, nbins = 5)
    Δx = step(k₂.x)
    @test k₂.x == edges2centers(range(0.5, 4.5, length = 5 + 1))
    @test isapprox(k₂.f .* Δx, h₁)
    @test sum(k₂.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # prepared with unknown number of bins, but known bounds
    k₃, info = kde(KDE.HistogramBinning(), v₁; cover = KDE.Closed, lo = 0, hi = 5)
    Δx = step(k₃.x)
    @test k₃.x == edges2centers(range(0.0, 5.0, length = round(Int, 5 / b₀) + 1))
    @test isapprox(filter(!iszero, k₃.f) .* Δx, h₁)
    @test sum(k₃.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # prepared with unknown limits and number of bins
    k₄, info = kde(KDE.HistogramBinning(), v₁; cover = KDE.Closed)
    Δx = step(k₄.x)
    @test isapprox(filter(!iszero, k₄.f) .* Δx, h₁)
    @test sum(k₄.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # prepared using an alternative sampling density (compared to previous)
    k₅, info = kde(KDE.HistogramBinning(), v₁; cover = KDE.Closed, bwratio = 16)
    Δx = step(k₅.x)
    @test step(k₅.x) < step(k₄.x)
    @test isapprox(filter(!iszero, k₅.f) .* Δx, h₁)
    @test sum(k₅.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # make sure errors do not occur when uniform data is provided
    let (k, info) = kde(KDE.HistogramBinning(), ones(100))
        @test length(k.x) == 1
        @test isfinite(info.bandwidth) && !iszero(info.bandwidth)
        @test sum(k.f) * step(k.x) == 1.0
    end

    @test (@inferred kde(KDE.HistogramBinning(), [1.0, 2.0]; nbins = 2)
           isa Tuple{KDE.UnivariateKDE{Float64,<:AbstractRange{Float64},
                                       <:AbstractVector{Float64}},
                     KDE.UnivariateKDEInfo{Float64}})

end

@testset "Basic Kernel Density Estimate" begin
    nbins = 11  # use odd value to have symmetry
    nsamp = 10nbins
    Δx = 1.0 / nbins

    v_uniform = range(-1, 1, length = nsamp)
    p_uniform = 1.0 / nbins

    # Tune the bandwidth to be much smaller than the bin sizes, which effectively means
    # we just get back the internal histogram (when performing only a basic KDE).
    (x, f), _ = kde(KDE.BasicKDE(), v_uniform, cover = KDE.Closed, nbins = nbins, bandwidth = 0.1Δx)
    @test all(isapprox.(f .* step(x), p_uniform, atol = 1e-3))
    # Then increase the bandwidth to be the same size of the bins. The outermost bins will
    # impacted by the kernel convolving with implicit zeros beyond the edges of the
    # distribution, so we should expect a significant decrease in the first and last bin.
    (x, f), _ = kde(KDE.BasicKDE(), v_uniform, cover = KDE.Closed, nbins = nbins, bandwidth = Δx)
    @test all(isapprox.(f[2:end-1] .* step(x), p_uniform, atol = 1e-3))  # approximate p_uniform
    @test all(<(-1e-3), f[[1,end]] .* step(x) .- p_uniform) # systematically less than p_uniform

    # If we put a Kronecker delta in the center of a span with a wide (lo, hi) spread,
    # we should effectively just get back the convolution kernel.
    v_kron = zeros(100)
    bw = 1.0

    (x, f1), _ = kde(KDE.BasicKDE(), v_kron; cover = KDE.Closed, lo = -6bw, hi = 6bw, nbins = 251, bandwidth = bw)
    (x, f2), _ = kde(KDE.BasicKDE(), v_kron; cover = KDE.Closed, lo = -6bw, hi = 6bw, nbins = 251, bandwidth = bw)
    g = exp.(.-(x ./ bw) .^2 ./ 2) ./ (bw * sqrt(2π)) .* step(x)
    @test all(isapprox.(f1 .* step(x), g, atol = 1e-5))
    @test all(isapprox.(f2 .* step(x), g, atol = 1e-5))
end

@testset "Linear Boundary Correction" begin
    nbins = 11  # use odd value to have symmetry
    nsamp = 10nbins
    Δx = 1.0 / nbins

    v_uniform = range(-1, 1, length = nsamp)
    p_uniform = 1.0 / nbins

    # Enable the normalization option, which makes a correction for the implicit zeros
    # being included in the convolution
    (x, f), _ = kde(KDE.LinearBoundaryKDE(), v_uniform, cover = KDE.Closed, nbins = nbins, bandwidth = Δx)
    @test all(isapprox.(f .* step(x), p_uniform, atol = 1e-3))  # approximately p_uniform
    # and that correction keeps working as the bandwidth causes multiple bins to be affected
    (x, f), _ = kde(KDE.LinearBoundaryKDE(), v_uniform, cover = KDE.Closed, nbins = nbins, bandwidth = 6Δx)
    @test all(isapprox.(f .* step(x), p_uniform, atol = 1e-3))  # approximately p_uniform

    # The boundary correction only applies when a hard boundary is encountered, so the
    # BasicKDE and LinearBoundaryKDE should provide very similar distributions when
    # the cover is Open.
    (x₁, f₁), _ = kde(KDE.BasicKDE(), v_uniform, cover = KDE.Open, nbins = nbins, bandwidth = 2Δx)
    (x₂, f₂), _ = kde(KDE.LinearBoundaryKDE(), v_uniform, cover = KDE.Open, nbins = nbins, bandwidth = 2Δx)
    @test all(isapprox.(f₁, f₂, atol = 1e-3))
end
