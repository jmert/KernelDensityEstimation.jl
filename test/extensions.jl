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

    if isdefined(Test, :detect_closure_boxes)
        ext = Base.get_extension(KDE, :KDEMakieExt)
        @test length(Test.detect_closure_boxes(ext)) == 0
    end
end


@moduletestset "Plots" begin
    using Plots

    rv = rand(1000)
    K = kde(rv, lo = 0.0, hi = 1.0, boundary = :closed)

    # simply test that invocation is not an error
    @test plot(K) isa Plots.Plot
    @test plot(K, linetype = :steppre) isa Plots.Plot

    # default labels are set
    p = plot(K)
    @test p[1][:xaxis][:guide] == "value"
    @test p[1][:yaxis][:guide] == "density"

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

    if isdefined(Test, :detect_closure_boxes)
        ext = Base.get_extension(KDE, :KDEUnicodePlotsExt)
        @test length(Test.detect_closure_boxes(ext)) == 0
    end
end
