using Test

import KernelDensityEstimation as KDE
import .KDE: conv, conv!, convaxes, plan_conv, replan_conv!


function conv_direct(F::Array{T}, K::Array{U}) where {T,U}
    ax = convaxes(size(F), size(K), FULL)
    C = zeros(promote_type(T, U), length.(ax)...)

    for m in CartesianIndices(F)
        for n in CartesianIndices(K)
            C[n + m - oneunit(n)] += F[m] * K[n]
        end
    end
    return C
end


@testset "Convolution Sizing" begin
    import .KDE.ConvShape: FULL, SAME, VALID

    # loop to step through 1, 2, and 3D ranges
    for nd in [(), (5,), (5, 7)]
        dimscases = [
            [(6, nd...), (3, nd...)],  # kernel shorter than data; odd kernel size
            [(6, nd...), (4, nd...)],  # kernel shorter than data; even kernel size
            [(5, nd...), (5, nd...)],  # kernel and data same size, odd
            [(6, nd...), (6, nd...)],  # kernel and data same size, odd
            [(3, nd...), (5, nd...)],  # kernel longer than data; odd kernel size
            [(3, nd...), (6, nd...)],  # kernel longer than data; even kernel size
        ]
        @testset "ddims = $ddims, kdims = $kdims" for (ddims, kdims) in dimscases
            # full dimensions are sum of axes lengths minus 1
            fulldims = map(i -> 1:i, ddims .+ kdims .- 1)
            fullaxes = convaxes(ddims, kdims, FULL)
            @test fullaxes == fulldims

            # same take the central portion which is the same length as ddims
            sameaxes = convaxes(ddims, kdims, SAME)
            @test length.(sameaxes) == ddims
            # if the kernel size is odd, there are an equal number of indices removed from
            # both sides of the full range; for even kernel sizes, the leading edge loses
            # one more
            #    N.B. the choice as to whether the leading or trailing edge loses the one
            #         extra for even kernel sizes is relatively arbitrary; Matlab/octave
            #         and numpy/scipy choose opposite conventions
            @test first.(sameaxes) == cld.(kdims .+ 1, 2)

            # valid retains only the inner region of ddims which can completely encompass
            # the kernel
            validaxes = convaxes(ddims, kdims, VALID)
            @test length.(validaxes) == max.(0, ddims .- kdims .+ 1)
        end
    end
end

@testset "Planning N=$N" for N in 1:3
    ext = (3 for _ in 2:N)
    A = zeros(25, ext...)
    K = ones(Float32, 5, ext...)

    @test (@inferred plan_conv(A, K)) isa KDE.ConvPlan{N, Float64}

    p1 = plan_conv(A, K)
    K2 = 2 .* K
    @test (@allocated replan_conv!(p1, K2)) == 0

    @test all(size(p1.f) .≥ size(A) .+ size(K) .- 1)
    # first dimension 25+5-1 == 29 is padded up to efficient FFT size 5¹×3¹×2¹ = 30
    @test size(p1.f, 1) > size(A, 1) + size(K, 1) - 1
end

@testset "Direct comparison (N=$N)" for N in 1:3
    kdims = rand(5:10, N)
    K = rand(kdims...)

    # ddims > kdims
    ddims = rand(11:20, N)
    L = randn(ddims...)
    @test @inferred(conv(L, K, FULL)) ≈ conv_direct(L, K)

    # ddims == kdims
    ddims = kdims
    M = randn(ddims...)
    @test @inferred(conv(L, K, FULL)) ≈ conv_direct(L, K)

    # ddims < kdims
    ddims = rand(2:4, N)
    S = randn(ddims...)
    @test @inferred(conv(L, K, FULL)) ≈ conv_direct(L, K)
end

@testset "Error Conditions" begin
    n, m = 25, 5
    l = n + m - 1  # FULL size
    L = nextprod((2, 3, 5, 7), l)  # FFT buffer size

    f = zeros(n)
    K = zeros(m)
    g = zeros(L)

    # mismatched sizes compared to plan
    plan = plan_conv(f, K)
    @test_throws DimensionMismatch replan_conv!(plan, zeros(m + 1))
    @test_throws DimensionMismatch conv(zeros(10), plan)
    @test_throws DimensionMismatch conv!(g, zeros(10), plan)
    @test_throws DimensionMismatch conv!(zeros(10), f, plan)

    # invalid shape symbol
    #   disabled because the symbol method is commented out in src/conv.jl
    #@test_throws ArgumentError conv(zeros(5), zeros(3), :invalid)
end
