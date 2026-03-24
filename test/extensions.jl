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

    @testset "Univariate" begin
        # open boundary
        D = Normal(0.0, 1.0)
        r = rand(D, 100)
        lo, hi, bc = KDE.bounds((r,), D)[1]
        @test bc == KDE.Open
        @test (lo, hi) == extrema(r)
        K = kde(r; bounds = D, kws...)
        @test iszero(first(K.f)) && iszero(last(K.f))  # open boundaries reach density 0 at edges

        # half-open boundary
        D = Exponential(1.0)
        r = rand(D, 100)
        lo, hi, bc = KDE.bounds((r,), D)[1]
        @test bc == KDE.ClosedLeft
        @test (lo, hi) == (minimum(D), maximum(r))
        K = kde(r; bounds = D, kws...)
        @test first(K.x) ≈ minimum(D) + step(K.x) / 2
        @test iszero(last(K.f))  # open boundary reaches density 0 at edge

        # closed boundary
        D = Uniform(2.0, 10.0)
        r = rand(D, 100)
        lo, hi, bc = KDE.bounds((r,), D)[1]
        @test bc == KDE.Closed
        @test (lo, hi) == extrema(D)
        K = kde(r; bounds = D, kws...)
        @test first(K.x) ≈ minimum(D) + step(K.x) / 2
        @test last(K.x) ≈ maximum(D) - step(K.x) / 2

        # 1D special case, accepting an unwrapped data vector
        @test KDE.bounds((r,), D) === KDE.bounds(r, D)
    end

    @testset "Bivariate" begin
        # N.B. we don't actually care about the values
        x = randn(100)
        y = randn(100)

        # open boundaries, single distribution
        D = MvNormal([0, 0], [1.0 0.0; 0.0 2.0])
        b1, b2 = KDE.bounds((x, y), D)
        @test (b1[1], b1[2]) == extrema(x)
        @test (b2[1], b2[2]) == extrema(y)
        @test b1[3] == KDE.Open
        @test b2[3] == KDE.Open
        @test kde(x, y; bounds = D) !== nothing  # no error

        # open boundaries, multiple distributions
        D = (Normal(0, 1), Normal(0, sqrt(2)))
        b1, b2 = KDE.bounds((x, y), D)
        @test (b1[1], b1[2]) == extrema(x)
        @test (b2[1], b2[2]) == extrema(y)
        @test b1[3] == KDE.Open
        @test b2[3] == KDE.Open
        @test K = kde(x, y; bounds = D) !== nothing  # no error

        # mixed open / closed boundaries, single distribution
        D = product_distribution([Normal(0, 1), Uniform(-5, 5)])
        b1, b2 = KDE.bounds((x, y), D)
        @test (b1[1], b1[2]) == extrema(x)
        @test (b2[1], b2[2]) == (-5, 5)
        @test b1[3] == KDE.Open
        @test b2[3] == KDE.Closed
        @test K = kde(x, y; bounds = D) !== nothing  # no error
    end

    if isdefined(Test, :detect_closure_boxes)
        ext = Base.get_extension(KDE, :KDEDistributionsExt)
        @test length(Test.detect_closure_boxes(ext)) == 0
    end
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

    # conversions for bivariate densities
    rv2 = 2.0 .+ 0.8 .* randn(length(rv))
    K2 = kde(rv, rv2, bounds = ((0, 1, :closed), (-Inf, Inf, :open)))

    # test that invocations succeed without error
    @test plotfunc(plot(K2).plot) === contour
    @test plotfunc(contour(K2).plot) === contour
    @test plotfunc(contourf(K2).plot) === contourf
    @test plotfunc(heatmap(K2).plot) === heatmap

    if isdefined(Test, :detect_closure_boxes)
        ext = Base.get_extension(KDE, :KDEMakieExt)
        @test length(Test.detect_closure_boxes(ext)) == 0
    end
end


@moduletestset "Plots" begin
    using Plots

    rv = rand(1000)
    K = kde(rv, lo = 0.0, hi = 1.0, boundary = :closed)

    # N.B. the plot must be rendered to trigger errors
    buf = IOBuffer()
    sshow(x) = (show(buf, "text/html", x); String(take!(buf)))

    # simply test that the plot does not error
    @test sshow(plot(K)) isa String
    @test sshow(plot(K, linetype = :steppre)) isa String

    # default labels are set
    p = plot(K)
    @test p[1][:xaxis][:guide] == "value"
    @test p[1][:yaxis][:guide] == "density"

    # conversions for bivariate densities
    rv2 = 2.0 .+ 0.8 .* randn(length(rv))
    K2 = kde(rv, rv2, bounds = ((0, 1, :closed), (-Inf, Inf, :open)))

    # test that invocations succeed without error
    @test sshow(plot(K2)) isa String
    @test sshow(plot(K2, seriestype = :heatmap)) isa String

    if isdefined(Test, :detect_closure_boxes)
        ext = Base.get_extension(KDE, :KDERecipesBaseExt)
        @test length(Test.detect_closure_boxes(ext)) == 0
    end
end


@moduletestset "UnicodePlots" begin
    using UnicodePlots

    K = kde(sqrt.(0:0.01:1), bandwidth = 0.5, boundary = :closed)

    function termprint(x)
        context = (:displaysize => (50, 120), :color => true)
        return sprint(x; context) do io, obj
            show(io, MIME"text/plain"(), obj)
        end
    end

    # generate an empty plot (to check that plotting is different from)
    empty = termprint(Plot(Float64[], Float64[]; xlim = (0, 1), ylim = (0, 3)))

    # plot from scratch
    p1 = lineplot(K, color = :green)
    str1 = termprint(p1)
    @test str1 != empty
    @test any(!isascii, str1)

    # plot into a canvas
    p2 = Plot(Float64[], Float64[]; xlim = (0, 1), ylim = (0, 3))
    lineplot!(p2, K, color = :green)
    str2 = termprint(p2)
    @test str1 == str2
    # change color and overplot, resulting in different (color!) plot
    lineplot!(p2, K, color = :blue)
    str3 = termprint(p2)
    @test str3 != str2
    # but contents are the same if the color information is stripped away
    #  pattern based on https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
    stripcolor(s) = replace(s, r"\e\[(\d+;)*\d*m" => "")
    @test stripcolor(str3) == stripcolor(str2)

    # support for bivariate densities
    rv1 = rand(1000)
    rv2 = 2.0 .+ 0.8 .* randn(length(rv1))
    K2 = kde(rv1, rv2, bounds = ((0, 1, :closed), (-Inf, Inf, :open)))

    # plot from scratch
    pc = contourplot(K2)
    strc = termprint(pc)
    @test strc != empty
    @test any(!isascii, strc)

    # plot into a canvas
    pc2 = Plot(Float64[], Float64[]; xlim = (0, 1), ylim = (-1, 5))
    contourplot!(pc2, K2)
    strc2 = termprint(pc2)
    @test strc2 != empty
    @test any(!isascii, strc2)

    if isdefined(Test, :detect_closure_boxes)
        ext = Base.get_extension(KDE, :KDEUnicodePlotsExt)
        @test length(Test.detect_closure_boxes(ext)) == 0
    end
end
