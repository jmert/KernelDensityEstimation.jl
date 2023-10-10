using Test
import KernelDensityEstimation: histogram

@testset "Half-open intervals" begin
    @test histogram([0.0], 1:5) == [0, 0, 0, 0]
    @test histogram([1.0], 1:5) == [1, 0, 0, 0]
    @test histogram([2.0], 1:5) == [0, 1, 0, 0]
    @test histogram([3.0], 1:5) == [0, 0, 1, 0]
    @test histogram([4.0], 1:5) == [0, 0, 0, 1]
    @test histogram([5.0], 1:5) == [0, 0, 0, 1]
    @test histogram([6.0], 1:5) == [0, 0, 0, 0]
end

@testset "Other checks" begin
    @test @inferred(histogram([0.0], 1:2)) isa Vector{Int}
end

