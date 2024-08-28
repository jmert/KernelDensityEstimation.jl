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
