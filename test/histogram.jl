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

@testset "Edge cases" begin
    # zero-width ranges can still count occurrences that are equal
    rng = range(1.0, 1.0, length = 2)
    @test histogram(ones(1), rng) == [1]
    @test histogram(ones(5), rng) == [5]
    # zero-width ranges can still have arbitrary length
    rng = range(1.0, 1.0, length = 5)
    @test collect(rng) == ones(5)
    @test histogram(ones(1), rng) == [1, 0, 0, 0]
    @test histogram(ones(5), rng) == [5, 0, 0, 0]
end
