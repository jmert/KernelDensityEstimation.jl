using Test

@testset verbose=true "KernelDensityEstimation tests" begin
    @testset "Convolutions" include("conv.jl")
    @testset verbose=true "Kernel Density Estimation" include("kde.jl")
end
