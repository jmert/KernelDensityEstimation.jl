using Base.Iterators: product
using Unitful

using .KDE: Histogramming

const HB = Histogramming.HistogramBinning()
const LB = Histogramming.LinearBinning()


@testset "Inner Binning ($(N)D, $style)" for N in 1:3, style in (HB, LB)
    using .KDE.Histogramming: HistEdge, _hist_inner!

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
