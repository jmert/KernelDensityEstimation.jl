using KernelDensityEstimation
const KDE = KernelDensityEstimation

using Statistics: std, var
using Random: rand

edges2centers(r) = r[2:end] .- step(r) / 2

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
    k₁, info = kde(KDE.HistogramBinning(), v₁; lo = 0, hi = 5, nbins = 5)
    Δx = step(k₁.x)
    @test k₁.x == edges2centers(0.0:1.0:5.0)
    @test isapprox(k₁.f .* Δx, h₁)
    @test sum(k₁.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # prepared with known number of bins, but automatically selected bounds
    k₂, info = kde(KDE.HistogramBinning(), v₁; nbins = 5)
    Δx = step(k₂.x)
    @test k₂.x == edges2centers(range(0.5, 4.5, length = 5 + 1))
    @test isapprox(k₂.f .* Δx, h₁)
    @test sum(k₂.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # prepared with unknown number of bins, but known bounds
    k₃, info = kde(KDE.HistogramBinning(), v₁; lo = 0, hi = 5)
    Δx = step(k₃.x)
    @test k₃.x == edges2centers(range(0.0, 5.0, length = round(Int, 5 / b₀) + 1))
    @test isapprox(filter(!iszero, k₃.f) .* Δx, h₁)
    @test sum(k₃.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # prepared with unknown limits and number of bins
    k₄, info = kde(KDE.HistogramBinning(), v₁)
    Δx = step(k₄.x)
    @test isapprox(filter(!iszero, k₄.f) .* Δx, h₁)
    @test sum(k₄.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # prepared using an alternative sampling density (compared to previous)
    k₅, info = kde(KDE.HistogramBinning(), v₁; bwratio = 16)
    Δx = step(k₅.x)
    @test step(k₅.x) < step(k₄.x)
    @test isapprox(filter(!iszero, k₅.f) .* Δx, h₁)
    @test sum(k₅.f) * Δx ≈ 1.0
    @test info.bandwidth ≈ b₀

    # make sure errors do not occur when uniform data is provided
    let (k, info) = kde(KDE.HistogramBinning(), ones(100))
        @test length(k.x) == 1
        @test isfinite(info.bandwidth) && !iszero(info.bandwidth)
        @test sum(k.f) == 1.0
    end

    @test (@inferred kde(KDE.HistogramBinning(), [1.0, 2.0]; nbins = 2)
           isa Tuple{KDE.UnivariateKDE{Float64,<:AbstractRange{Float64},
                                       <:AbstractVector{Float64}},
                     KDE.UnivariateKDEInfo{Float64}})

end

@testset "Basic Estimate" begin
    nbins = 11  # use odd value to have symmetry
    nsamp = 10nbins
    Δx = 1.0 / nbins

    v_uniform = range(-1, 1, length = nsamp)
    p_uniform = 1.0 / nbins
    x, f = kde(v_uniform)

    @test length(x) == length(f)


    # Tune the bandwidth to be much smaller than the bin sizes, which effectively means
    # we just get back the internal histogram (when the normalization option is disabled).
    x, f = kde(v_uniform, nbins = nbins, bandwidth = 0.1Δx,
               opt_normalize = false, opt_linearboundary = false, opt_multiply = false)
    @test all(isapprox.(f .* step(x), p_uniform, atol = 1e-3))
    # Then increase the bandwidth to be the same size of the bins. The outermost bins will
    # impacted by the kernel convolving with implicit zeros beyond the edges of the
    # distribution, so we should expect a significant decrease in the first and last bin.
    x, f = kde(v_uniform, nbins = nbins, bandwidth = Δx,
               opt_normalize = false, opt_linearboundary = false, opt_multiply = false)
    @test all(isapprox.(f[2:end-1] .* step(x), p_uniform, atol = 1e-3))  # approximate p_uniform
    @test all(<(-1e-3), f[[1,end]] .* step(x) .- p_uniform) # systematically less than p_uniform
    # Enable the normalization option, which makes a correction for the implicit zeros
    # being included in the convolution
    x, f = kde(v_uniform, nbins = nbins, bandwidth = Δx,
               opt_normalize = true, opt_linearboundary = false, opt_multiply = false)
    @test all(isapprox.(f .* step(x), p_uniform, atol = 1e-3))  # approximately p_uniform
    # and that correction keeps working as the bandwidth causes multiple bins to be affected
    x, f = kde(v_uniform, nbins = nbins, bandwidth = 6Δx,
               opt_normalize = true, opt_linearboundary = false, opt_multiply = false)
    @test all(isapprox.(f .* step(x), p_uniform, atol = 1e-3))  # approximately p_uniform



    # If we put a Kronecker delta in the center of a span with a wide (lo, hi) spread,
    # we should effectively just get back the convolution kernel.
    v_kron = zeros(100)
    bw = 1.0

    x, f1 = kde(v_kron, lo = -6bw, hi = 6bw, nbins = 251, bandwidth = bw,
                opt_normalize = false, opt_linearboundary = false, opt_multiply = false)
    x, f2 = kde(v_kron, lo = -6bw, hi = 6bw, nbins = 251, bandwidth = bw,
                opt_normalize = true, opt_linearboundary = false, opt_multiply = false)
    g = exp.(.-(x ./ bw) .^2 ./ 2) ./ (bw * sqrt(2π)) .* step(x)
    @test all(isapprox.(f1 .* step(x), g, atol = 1e-5))
    @test all(isapprox.(f2 .* step(x), g, atol = 1e-5))
end
