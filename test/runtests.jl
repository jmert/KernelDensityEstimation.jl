using Test

@testset "Histograms" include("histogram.jl")
@testset "Convolutions" include("conv.jl")
@testset "Kernel Density Estimation" include("kde.jl")
