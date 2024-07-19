using Test

@testset verbose=true "KernelDensityEstimation tests" begin
    @testset "Convolutions" begin; include("conv.jl"); end
    @testset verbose=true "Kernel Density Estimation" begin; include("kde.jl"); end
end
