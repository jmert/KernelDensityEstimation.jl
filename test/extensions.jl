macro moduletestset(name::Union{Symbol,String}, block)
    testmod = Symbol(:Test, name, :Ext)
    ex = quote
        module $testmod
            using KernelDensityEstimation, Test
            const KDE = KernelDensityEstimation
            @testset $name $block
        end
    end
    ex.head = :toplevel
    return esc(ex)
end


@moduletestset "Distributions" begin
    using Distributions

    # some options to speed up the KDE algorithm for details we're not checking
    kws = (; method = KDE.LinearBinning(), bandwidth = KDE.SilvermanBandwidth(), bwratio = 1.0)

    D = Normal(0.0, 1.0)
    r = rand(D, 100)
    lo, hi, bc = KDE.bounds(r, D)
    @test bc == KDE.Open
    @test (lo, hi) == extrema(r)
    K = kde(r; bounds = D, kws...)
    @test iszero(first(K.f)) && iszero(last(K.f))  # open boundaries reach density 0 at edges

    D = Exponential(1.0)
    r = rand(D, 100)
    lo, hi, bc = KDE.bounds(r, D)
    @test bc == KDE.ClosedLeft
    @test (lo, hi) == (minimum(D), maximum(r))
    K = kde(r; bounds = D, kws...)
    @test first(K.x) ≈ minimum(D) + step(K.x) / 2
    @test iszero(last(K.f))  # open boundary reaches density 0 at edge

    D = Uniform(2.0, 10.0)
    r = rand(D, 100)
    lo, hi, bc = KDE.bounds(r, D)
    @test bc == KDE.Closed
    @test (lo, hi) == extrema(D)
    K = kde(r; bounds = D, kws...)
    @test first(K.x) ≈ minimum(D) + step(K.x) / 2
    @test last(K.x) ≈ maximum(D) - step(K.x) / 2
end

@moduletestset "Makie" begin
    using CairoMakie
    using .Makie: plotfunc
    using Unitful

    rv = rand(1000)
    K = kde(rv, lo = 0.0, hi = 1.0, boundary = :closed)

    # simply test that invocation is not an error
    @test plotfunc(plot(K).plot) === lines
    @test plotfunc(lines(K).plot) === lines
    @test plotfunc(scatter(K).plot) === scatter
    @test plotfunc(stairs(K).plot) === stairs

    # stairs has special behavior; check that the converted data correctly closes the
    # histogram
    fig, ax, pl = stairs(K)
    y = pl[1][]  # semi-nonpublic interface...
    @test y[1] ≈ [0.0, 0.0] atol=eps()
    @test y[end] ≈ [1.0, 0.0] atol=eps()
    @test y[end][1] == y[end-1][1]

    # unitful handling was added in Makie v0.21
    if isdefined(Makie, :UnitfulConversion)
        # check that unitful KDEs plot as well
        Ku = kde(rv .* 1Unitful.m, lo = 0.0, hi = 1.0, boundary = :closed)
        @test plot(Ku).plot isa Plot{lines}
        @test stairs(Ku).plot isa Plot{stairs}
    end
end

@moduletestset "UnicodePlots" begin
    import UnicodePlots

    io = IOBuffer()
    ht, wd = 50, 120
    ioc = IOContext(io, :displaysize => (ht, wd))

    K = kde(collect(0:0.01:1), bandwidth = 0.5, boundary = :closed)
    show(ioc, MIME"text/plain"(), K)
    str = String(take!(io))

    let S = split(str, '\n')
        @test all(length(s) <= wd for s in S)
        @test length(S) < ht
    end
    @test any(!isascii, str)
end
