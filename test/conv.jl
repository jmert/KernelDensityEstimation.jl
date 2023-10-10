using Test

import KernelDensityEstimation: conv

@testset "Literal examples" begin
    let u = Float64[1, 1, 1], v = Float64[1, 1, 0, 0, 0, 1, 1]
        @test conv(u, v, :full) ≈ [1, 2, 2, 1, 0, 1, 2, 2, 1]
        @test conv(u, v, :same) ≈ [1, 0, 1]
        @test conv(u, v, :valid) == Float64[]
    end
    let u = Float64[1, 1, -1, 1], v = Float64[1, 1, 0, 0, 0, 1, 1]
        @test conv(u, v, :full) ≈ [1, 2, 0, 0, 1, 1, 2, 0, 0, 1]
        @test conv(u, v, :same) ≈ [0, 1, 1, 2]
        @test conv(u, v, :valid) == Float64[]
    end
    let u = Float64[1, 1, 0, 0, 0, 1, 1], v = Float64[1, 1, 1]
        @test conv(u, v, :full) ≈ [1, 2, 2, 1, 0, 1, 2, 2, 1]
        @test conv(u, v, :same) ≈ [2, 2, 1, 0, 1, 2, 2]
        @test conv(u, v, :valid) ≈ [2, 1, 0, 1, 2]
    end
    let u = Float64[1, 1, 0, 0, 0, 1, 1], v = Float64[1, 1, 1, 1]
        @test conv(u, v, :full) ≈ [1, 2, 2, 2, 1, 1, 2, 2, 2, 1]
        @test conv(u, v, :same) ≈ [2, 2, 1, 1, 2, 2, 2]
        @test conv(u, v, :valid) ≈ [2, 1, 1, 2]
    end
    let u = Float64[-1, 2, 3, -2, 0, 1, 2], v = Float64[2, 4, -1, 1]
        @test conv(u, v, :full) ≈ [-2, 0, 15, 5, -9, 7, 6, 7, -1, 2]
        @test conv(u, v, :same) ≈ [15, 5, -9, 7, 6, 7, -1]
        @test conv(u, v, :valid) ≈ [5, -9, 7, 6]
    end
end

@testset "Other checks" begin
    let u = Float64[1, 1, 1], v = Float32[1, 1, 0, 0, 0, 1, 1]
        @test @inferred(conv(u, v, :full)) isa Vector{Float64}
    end
    let u = Float32[1, 1, 1], v = Float64[1, 1, 0, 0, 0, 1, 1]
        @test @inferred(conv(u, v, :full)) isa Vector{Float64}
    end
end
