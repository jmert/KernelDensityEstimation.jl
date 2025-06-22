using Random

@testset "Distributions" begin
    using Distributions

    # some options to speed up the KDE algorithm for details we're not checking
    kws = (; method = KDE.LinearBinning(), bandwidth = KDE.SilvermanBandwidth(), bwratio = 1.0)

    D = Normal(0.0, 1.0)
    @test KDE.boundary(D) == KDE.Open
    r = rand(D, 100)
    lo, hi, boundary = KDE.bounds(r, D)
    @test boundary == KDE.Open
    @test (lo, hi) == extrema(r)
    K = kde(r; bounds = D, kws...)
    @test iszero(first(K.f)) && iszero(last(K.f))  # open boundaries reach density 0 at edges

    D = Exponential(1.0)
    @test KDE.boundary(D) == KDE.ClosedLeft
    r = rand(D, 100)
    lo, hi, boundary = KDE.bounds(r, D)
    @test boundary == KDE.ClosedLeft
    @test (lo, hi) == (minimum(D), maximum(r))
    K = kde(r; bounds = D, kws...)
    @test first(K.x) ≈ minimum(D) + step(K.x) / 2
    @test iszero(last(K.f))  # open boundary reaches density 0 at edge

    D = Uniform(2.0, 10.0)
    @test KDE.boundary(D) == KDE.Closed
    r = rand(D, 100)
    lo, hi, boundary = KDE.bounds(r, D)
    @test boundary == KDE.Closed
    @test (lo, hi) == extrema(D)
    K = kde(r; bounds = D, kws...)
    @test first(K.x) ≈ minimum(D) + step(K.x) / 2
    @test last(K.x) ≈ maximum(D) - step(K.x) / 2
end

@testset "Makie" begin
    using CairoMakie

    rv = rand(1000)
    K = kde(rv, lo = 0.0, hi = 1.0, boundary = :closed)

    # simply test that invocation is not an error
    @test plot(K).plot isa Plot{lines}
    @test lines(K).plot isa Plot{lines}
    @test scatter(K).plot isa Plot{scatter}
    @test stairs(K).plot isa Plot{stairs}

    # stairs has special behavior; check that the converted data correctly closes the
    # histogram
    fig, ax, pl = stairs(K)
    y = pl[1][]  # semi-nonpublic interface...
    @test y[1] ≈ [0.0, 0.0] atol=eps()
    @test y[end] ≈ [1.0, 0.0] atol=eps()
    @test y[end][1] == y[end-1][1]
end

@testset "UnicodePlots" begin
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
