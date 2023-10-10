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
        e₁, c₁, b₁ = _kde_prepare(v₁, 0.0, 5.0; npts = 5)
        @test e₁ == 0.0:1.0:5.0
        @test c₁ == h₁
        @test sum(c₁) ≈ 1.0
        @test b₁ ≈ b₀

        # prepared with known number of bins, but automatically selected bounds
        e₂, c₂, b₂ = _kde_prepare(v₁, nothing, nothing; npts = 5)
        @test e₂ == range(0.5, 4.5, length = 5 + 1)
        @test c₂ == h₁
        @test sum(c₁) ≈ 1.0
        @test b₂ ≈ b₀

        # prepared with unknown number of bins, but known bounds
        e₃, c₃, b₃ = _kde_prepare(v₁, 0.0, 5.0; npts = nothing)
        @test e₃ == range(0.0, 5.0, length = round(Int, 8 * 5 / b₀) + 1)
        @test filter(!iszero, c₃) == h₁
        @test sum(c₁) ≈ 1.0
        @test b₃ ≈ b₀

        # prepared with unknown limits and number of bins
        e₄, c₄, b₄ = _kde_prepare(v₁, nothing, nothing; npts = nothing)
        @test filter(!iszero, c₄) == h₁
        @test sum(c₁) ≈ 1.0
        @test b₄ ≈ b₀

        # prepared using an alternative sampling density (compared to previous)
        e₅, c₅, b₅ = _kde_prepare(v₁, nothing, nothing; npts = nothing, bwratio = 16)
        @test step(e₅) < step(e₄)
        @test filter(!iszero, c₅) == h₁
        @test sum(c₅) ≈ 1.0
        @test b₅ ≈ b₀
    end
end
