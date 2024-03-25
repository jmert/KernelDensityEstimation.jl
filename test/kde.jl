using KernelDensityEstimation
using KernelDensityEstimation: _count_var, _kde_prepare

using Statistics: std, var
using Random: rand

@testset "Preparations" begin
    @testset "Helper: filtered count & variance" begin
        x = 10 .* rand(Float64, 500)
        for (l, h) in [(0, 10), (5, 10), (5, 8)]
            pred(z) = l ≤ z ≤ h
            x′ = filter(pred, x)
            ν′, σ′² = length(x′), var(x′, corrected = false)
            ν, σ² = _count_var(pred, x)
            @test ν == ν′
            @test σ² ≈ σ′²
        end
    end

    @testset "Preparatory histogramming" begin
        # raw data
        v₁ = Float64[0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 4.5]
        # expected histogram (excluding zero bins when binned finer)
        h₁ = [1, 2, 3, 2, 1] ./ length(v₁)
        @test sum(h₁) ≈ 1.0
        # bandwidth estimate
        ν, σ̂ = length(v₁), std(v₁, corrected = false)
        b₀ = σ̂ * (4//3 / ν)^(1/5)  # Silverman's rule

        # prepared using known limits and number of bins
        e₁, c₁, b₁ = _kde_prepare(v₁, 0.0, 5.0, 5)
        @test e₁ == 0.0:1.0:5.0
        @test c₁ == h₁
        @test sum(c₁) ≈ 1.0
        @test b₁ ≈ b₀

        # prepared with known number of bins, but automatically selected bounds
        e₂, c₂, b₂ = _kde_prepare(v₁, nothing, nothing, 5)
        @test e₂ == range(0.5, 4.5, length = 5 + 1)
        @test c₂ == h₁
        @test sum(c₁) ≈ 1.0
        @test b₂ ≈ b₀

        # prepared with unknown number of bins, but known bounds
        e₃, c₃, b₃ = _kde_prepare(v₁, 0.0, 5.0, nothing)
        @test e₃ == range(0.0, 5.0, length = round(Int, 8 * 5 / b₀) + 1)
        @test filter(!iszero, c₃) == h₁
        @test sum(c₁) ≈ 1.0
        @test b₃ ≈ b₀

        # prepared with unknown limits and number of bins
        e₄, c₄, b₄ = _kde_prepare(v₁, nothing, nothing, nothing)
        @test filter(!iszero, c₄) == h₁
        @test sum(c₁) ≈ 1.0
        @test b₄ ≈ b₀

        # prepared using an alternative sampling density (compared to previous)
        e₅, c₅, b₅ = _kde_prepare(v₁, nothing, nothing, nothing; bwratio = 16)
        @test step(e₅) < step(e₄)
        @test filter(!iszero, c₅) == h₁
        @test sum(c₅) ≈ 1.0
        @test b₅ ≈ b₀

        # make sure errors do not occur when uniform data is provided
        let (e, c, bw) = _kde_prepare(ones(100))
            @test length(e) == 2
            @test iszero(last(e) - first(e))
            @test isfinite(bw) && !iszero(bw)
            @test sum(c) == 1.0
        end
    end

    @testset "Type stability" begin
        @test (@inferred _kde_prepare([1.0, 2.0], nothing, nothing, 2)
               isa Tuple{typeof(1.0:0.5:2.0), Vector{Float64}, Float64})
    end
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
    @test all(f .≈ p_uniform)
    # Then increase the bandwidth to be the same size of the bins. The outermost bins will
    # impacted by the kernel convolving with implicit zeros beyond the edges of the
    # distribution, so we should expect a significant decrease in the first and last bin.
    x, f = kde(v_uniform, nbins = nbins, bandwidth = Δx,
               opt_normalize = false, opt_linearboundary = false, opt_multiply = false)
    @test all(abs.(f[2:end-1] .- p_uniform) .< 1e-3)  # approximate p_uniform
    @test all(     f[[1,end]] .- p_uniform  .< -1e-3) # systematically less than p_uniform
    # Enable the normalization option, which makes a correction for the implicit zeros
    # being included in the convolution
    x, f = kde(v_uniform, nbins = nbins, bandwidth = Δx,
               opt_normalize = true, opt_linearboundary = false, opt_multiply = false)
    @test all(abs.(f .- p_uniform) .< 1e-3)  # approximately p_uniform
    # and that correction keeps working as the bandwidth causes multiple bins to be affected
    x, f = kde(v_uniform, nbins = nbins, bandwidth = 6Δx,
               opt_normalize = true, opt_linearboundary = false, opt_multiply = false)
    @test all(abs.(f .- p_uniform) .< 1e-3)  # approximately p_uniform



    # If we put a Kronecker delta in the center of a span with a wide (lo, hi) spread,
    # we should effectively just get back the convolution kernel.
    v_kron = zeros(100)
    bw = 1.0

    x, f1 = kde(v_kron, lo = -6bw, hi = 6bw, nbins = 251, bandwidth = bw,
                opt_normalize = false, opt_linearboundary = false, opt_multiply = false)
    x, f2 = kde(v_kron, lo = -6bw, hi = 6bw, nbins = 251, bandwidth = bw,
                opt_normalize = true, opt_linearboundary = false, opt_multiply = false)
    g = exp.(.-(x ./ bw) .^2 ./ 2) ./ (bw * sqrt(2π)) .* step(x)
    @test all(isapprox.(f1, g, atol = 1e-5))
    @test all(isapprox.(f2, g, atol = 1e-5))
end
