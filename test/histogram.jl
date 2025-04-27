using Base.Iterators: product
using Unitful

using .KDE: Histogramming

const HB = Histogramming.HistogramBinning()
const LB = Histogramming.LinearBinning()


@testset "Inner Binning ($(N)D, $style)" for N in 1:3, style in (HB, LB)
    using .Histogramming: HistEdge, _hist_inner!

    edges = ((0.0:0.25:1.0 for _ in 1:N)...,)
    edges′ = HistEdge.(edges)
    centers = ((r[2:end] .- 0.5step(r) for r in edges)...,)
    hist = similar(first(centers), (axes(c, 1) for c in centers)...)

    # binning the values at the bin centers should result in exactly one value in each
    # bin (assuming unity weights)
    fill!(hist, 0.0)
    for x in product(centers...)
        _hist_inner!(style, hist, edges′, x, 1.0)
    end
    @test all(hist .=== 1.0)

    # Histogram the bin edges, as a way to test the handling of cell boundary conditions
    fill!(hist, 0.0)
    for x in product(edges...)
        _hist_inner!(style, hist, edges′, x, 1.0)
    end
    if style === HB
        # For values deviating from the bin centers, the HistogramBinning method adds the
        # weight to the bin which contains the value (on the half-open interval ``[a, b)``
        # except for the last bin which is closed, ``[a, b]``).
        #
        # The "interior" excludes last point/edges/faces
        idx1 = CartesianIndex.(product((a[1:end-1] for a in axes(hist))...))
        @test all(hist[idx1] .== 1.0)
        # The last corner in the (hyper)cube has twice as many contributions per axis
        idxN = CartesianIndex((a[end] for a in axes(hist))...)
        @test hist[idxN] == 2.0^N
        @test count(==(hist[idxN]), hist) == 1

    elseif style === LB
        # For values deviating from the bin centers, the LinearHistogram method splits the
        # weight contribution proportionally among the surrounding bits (on the half-open
        # interval ``[a, b)`` except for the last bin which is closed, ``[a, b]``).
        # Using the bin edges as values, the weight is split to the left and right bins,
        # except for the first and last edge which contributes all of its weight to the
        # first and last bins.

        # The interior are points fully-within the histogram
        idx1 = CartesianIndex.(product((a[2:end-1] for a in axes(hist))...))
        @test all(hist[idx1] .== 1.0)
        # The outermost corners in the (hyper)cube has fractional contributions from every
        # vertex, with the weighting volume decreasing as the number of shared vertices
        # decreases.
        idxN = CartesianIndex.(product((extrema(a) for a in axes(hist))...))
        wght = sum(0.5 ^ count_ones(n) for n in 0:(1 << N) - 1)
        @test all(hist[idxN] .== wght)
        @test count(==(wght), hist) == 2^N
    end
    # Rather than figuring out the remaining indexing to correctly check the outer
    # edges of each plane (and the last face in 3D), assume that getting the previous
    # two checks correct plus to the total count correct means that the other cases
    # are handled correctly, too.
    @test sum(hist) == mapreduce(length, *, edges)

    c₀ = ((first(c) for c in centers)...,)

    @testset "Code Generation" begin
        # check for quality of code generation and (avoiding) allocation
        fill!(hist, 0.0)
        @test (@inferred _hist_inner!(style, hist, edges′, c₀, 1.0)) === nothing
        @test (@allocated _hist_inner!(style, hist, edges′, c₀, 1.0)) == 0
    end

    @testset "Unitful numbers" begin
        # given an n-tuple value with units,
        val = ntuple(i -> Quantity(i/6.0, u"m"), Val(N))
        # the histogram axes have the same units, and the density has inverse units
        uedges = map(c -> c .* u"m", edges)
        uhist = zeros(typeof(1.0u"m^-1"), axes(hist)...)
        # verify that the function accepts unitful quantities
        @test _hist_inner!(style, uhist, HistEdge.(uedges), val, 1.0) === nothing
        @test sum(uhist) ≈ 1.0u"m^-1" rtol=2eps(1.0)
    end
end

@testset "Low Level ($(N)D, $style)" for N in 1:3, style in (HB, LB)
    using .Histogramming: HistEdge, _histogram!

    edges = ((0.0:0.25:1.0 for _ in 1:N)...,)
    edges′ = HistEdge.(edges)
    centers = ((r[2:end] .- 0.5step(r) for r in edges)...,)
    hist = similar(first(centers), (axes(c, 1) for c in centers)...)

    # values for dims 2 and up
    coord_rest = ((0.0 for _ in 2:N)...,)
    index_rest = ((1 for _ in 2:N)...,)

    # nothing weights are interpreted as unity weight
    x1 = [(0.33, coord_rest...)]
    fill!(hist, 0)
    _histogram!(style, hist, edges′, x1, nothing)
    @test sum(hist) * step(edges[1])^N == 1.0

    # out-of-bounds elements are not binned
    x0 = [(-1.0, (0.0 for _ in 1:N-1)...,)]
    fill!(hist, 0)
    _histogram!(style, hist, edges′, x0, nothing)
    @test sum(hist) == 0.0

    @testset "Code Generation" begin
        fill!(hist, 0)
        @test (@inferred _histogram!(style, hist, edges′, x1, nothing)) === 1.0
        @test_broken (@allocated _histogram!(style, hist, edges′, x1, nothing)) == 0
    end

    @testset "Unitful numbers" begin
        # given an n-tuple value with units,
        vals = [x .* u"m" for x in x1]
        # the histogram axes have the same units, and the density has inverse units
        uedges = map(c -> c .* u"m", edges)
        uhist = zeros(typeof(1.0u"m^-1"^N), axes(hist)...)
        # verify that the function accepts unitful quantities
        @test _histogram!(style, uhist, HistEdge.(uedges), vals, nothing) === 1.0
        @test sum(uhist) * step(uedges[1])^N ≈ 1.0 rtol=2eps(1.0)
    end
end

@testset "Weighting" begin
    using .Histogramming: HistEdge, _histogram!

    N = 100
    rv = randn(N)
    data = reinterpret(reshape, Tuple{Float64}, rv)

    Nlen = Float64(N)
    Npos = Float64(count(>=(0), rv))

    weight1 = ones(length(rv))
    weight2 = 2 .* weight1

    Nbin = 24
    edges = (HistEdge(-5.0, 5.0, Nbin),)
    edges_pos = (HistEdge(0.0, 5.0, Nbin ÷ 2),)

    H0 = zeros(Nbin)
    H1 = zeros(Nbin)
    H2 = zeros(Nbin)
    @testset "$style" for style in (HB, LB)
        fill!(H0, 0); fill!(H1, 0); fill!(H2, 0)

        # binning uses the sum of weights (not effective sample size as KDE does)
        wsum0 = _histogram!(style, H0, edges, data, nothing)
        wsum1 = _histogram!(style, H1, edges, data, weight1)
        wsum2 = _histogram!(style, H2, edges, data, weight2)
        @test wsum0 == N
        @test wsum1 == N
        @test wsum2 == 2N

        # because the above are all uniform weights, the histograms themselves should be
        # equal (up to floating point rounding differences)
        @test H0 ≈ H1 atol=eps(1.0)
        @test H0 ≈ H2 atol=eps(1.0)

        # binning weights respect limits and ignore out-of-bounds entries
        fill!(H0, 0); fill!(H1, 0); fill!(H2, 0)

        wsum0 = _histogram!(style, @view(H1[1:end÷2]), edges_pos, data, weight1)
        wsum1 = _histogram!(style, @view(H1[1:end÷2]), edges_pos, data, weight1)
        wsum2 = _histogram!(style, @view(H2[1:end÷2]), edges_pos, data, weight2)
        @test wsum0 == Npos
        @test wsum1 == Npos
        @test wsum2 == 2Npos
    end
end

@testset "Histogramming Accuracy" begin
    using .Histogramming: HistEdge, _histogram!

    r64 = 2e0:1e0/49:28e0
    r32 = 2f0:1f0/49:28f0
    l64 = LinRange(r64[1], r64[end], length(r64))
    l32 = LinRange(r32[1], r32[end], length(r32))

    @testset "$r" for r in (r64, l64, r32, l32)
        v = reinterpret(reshape, Tuple{eltype(r)}, Vector(r))

        edges = (HistEdge(r),)
        # For regular histogram binning, using the bin edges as values must result in a uniform
        # distribution except the last bin which is doubled (due to being closed on the right).
        H = zeros(eltype(r), length(r) - 1)
        ν = _histogram!(HB, H, edges, v, nothing)
        @test ν == length(r)
        @test_broken all(@view(H[1:end-1]) .== H[1])
        @test H[end] == 2H[1]

        # For linear binning, the first and last bins differ from the rest, getting all of the
        # weight from the two edges but also gaining half a contribution from their (only)
        # neighbors. (The remaining interior bins give up half of their weight but
        # simultaneously gain from a neighbor, so they are unchanged.)
        fill!(H, 0.0)
        ν = _histogram!(LB, H, edges, v, nothing)
        @test ν == length(r)
        @test all(@view(H[2:end-1]) .≈ H[2])
        @test H[end] ≈ H[1]
        @test H[1] ≈ 1.5H[2]
    end

    # A case where naively calculating the cell index and weight factors suffers from the
    # limits of finite floating point calculations, e.g.
    #
    #     zz = (x - lo) / Δx  # fractional index
    #     ii = trunc(Int, zz) - (x == hi)  # bin index, including right-closed last bin
    #     ww = (zz - ii) - 0.5
    #
    # results in ww ≈ 1.5 due to the value of zz not being an integer despite x == hi
    lo = 0.005653766369679568
    hi = x = 0.006728850177869153
    nbins = 122

    H = zeros(nbins)
    edges = (HistEdge(lo, hi, nbins),)
    _histogram!(LB, H, edges, [(x,)], nothing)
    @test all(iszero, @view H[1:end-1])
    @test H[end] > 0.0
end
