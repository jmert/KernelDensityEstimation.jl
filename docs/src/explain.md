# Explanation

```@contents
Pages = ["explain.md"]
Depth = 2:2
```

## Estimator Pipeline

### Direct Comparisons

The following figures provide direct comparisons of the four major steps in the estimator pipeline described above
through their visual impact on a few example distributions.

1. _LinearBinning_: The data is histogrammed onto a uniformly spaced grid.

   For visualization purposes, the histograms are plotted with bins with a width equal to the automatically determined
   bandwidth (see [Bandwidth Estimators](#Bandwidth-Estimators) below) for the distribution, whereas the remaining
   panels use 8× as many data points to achieve a smoother curve.

2. _BasicKDE_: This step convolves with the histogram with the Gaussian kernel in order to smooth the data from a
   discontinuous histogram into a smooth curve.

   The basic density estimator is sufficient for an unbounded and smooth distribution like the **Normal** case.
   In the cases of the **HalfNormal** and **Uniform** distributions that have non-zero boundaries, though, the
   distribution is severely underestimated near the boundaries.

3. _LinearBoundaryKDE_: This step corrects for the boundary effects by recovering both the normalization (due to
   convolving with implicit zeros beyond the boundary) and recovers the slope of the distribution near boundaries.

   Compared to the BasicKDE step, the **HalfNormal** and **Uniform** distributions near their boundaries are
   significantly improved.

4. _MultiplicativeBiasKDE_: The final stage permits use of a larger bandwidth to achieve a smoother density estimate
   without sacrificing the sharpness of any curves / peaks.
   The algorithm automatically increases the bandwidth when the multiplicative bias correction is used.

   The visual impact is more subtle than in previous stages, but the smoothness of the **Uniform** and **Chisq3**
   distributions compared to the previous stage are a consequence of the multiplicative bias correction permitting
   a larger kernel bandwidth (without broadening the peak in the **Chisq3** distribution).

#### Univariate Estimates

!!! details "Plotting Code"
    ```@example estimator_comparisons
    using CairoMakie
    using Distributions
    using Random

    import KernelDensityEstimation as KDE

    dists = [
        "Normal" => Normal(0.0, 1.0),
        "Chisq3" => Chisq(3.0),
        "HalfNormal" => truncated(Normal(0.0, 1.0); lower = 0.0),
        "Uniform" => Uniform(0.0, 1.0),
    ]

    estimators = [
        KDE.LinearBinning(),
        KDE.BasicKDE(),
        KDE.LinearBoundaryKDE(),
        KDE.MultiplicativeBiasKDE(),
    ]

    for (name, dist) in dists
        fig = Figure(size = (900, 350))
        axs = Axis[]

        for (ii, method) in enumerate(estimators)
            dohist = method isa KDE.AbstractBinningKDE

            Random.seed!(123)  # hide
            rv = rand(dist, 5_000)
            dens = KDE.kde(rv; method, bounds = dist, bwratio = dohist ? 1 : 8)

            ax = Axis(fig[1, ii]; title = string(nameof(typeof(method))),
                                  xlabel = "value", ylabel = "density")
            lines!(ax, dist, color = (:black, 0.5), linestyle = :dash, label = "true")
            plotter! = method isa KDE.AbstractBinningKDE ? stairs! : lines!
            plotter!(ax, dens; label = "estimate",
                               color = dohist ? :blue3 : :firebrick3)
            scatter!(ax, [0], [0], color = (:black, 0.0))  # transparent dot to stop suppressed y=0 axis
            ii > 1 && hideydecorations!(ax, grid = false, ticks = false)

            push!(axs, ax)
        end
        linkaxes!(axs...)

        Label(fig[0, :], name, font = :bold, fontsize = 20)

        colgap!(fig.layout, Fixed(4))
        rowgap!(fig.layout, Fixed(3))
        rowsize!(fig.layout, 1, Aspect(1, 1.0))
        fig.layout.alignmode = Outside(4)
        resize_to_layout!(fig)

        save("comparison_$name.svg", fig)
    end
    nothing  # hide
    ```

![](comparison_Normal.svg)
![](comparison_Chisq3.svg)
![](comparison_HalfNormal.svg)
![](comparison_Uniform.svg)

#### Bivariate Estimates

The following bivariate density estimates use the same distributions as above in the univariate case for the x-axis,
and a standard normal (zero mean, unit standard deviation) is used for the y-axis.

!!! details "Plotting Code"
    ```@example estimator_comparisons_2d
    using CairoMakie
    using Distributions
    using Random

    import KernelDensityEstimation as KDE

    isdefined(Base.Main, :hpd) || Base.include(Base.Main, joinpath(@__DIR__, "hpd.jl"))
    using Base.Main: hpd

    dists = [
        "Normal" => Normal(0.0, 1.0),
        "Chisq3" => Chisq(3.0),
        "HalfNormal" => truncated(Normal(0.0, 1.0); lower = 0.0),
        "Uniform" => Uniform(0.0, 1.0),
    ]

    disty = Normal(0.0, 1.0)
    σmin = logcdf(disty, -4.0)
    σmax = logcdf(disty, +4.0)
    yr = range(-4.0, 4.0, 251)

    estimators = [
        KDE.LinearBinning(),
        KDE.BasicKDE(),
        KDE.LinearBoundaryKDE(),
        KDE.MultiplicativeBiasKDE(),
    ]

    for (name, dist) in dists
        fig = Figure(size = (900, 350))
        axs = Axis[]

        # theory distribution
        xl = minimum(dist) |> z -> isfinite(z) ? z : invlogcdf(dist, σmin)
        xh = maximum(dist) |> z -> isfinite(z) ? z : invlogcdf(dist, σmax)
        xr = range(xl, xh, 251)
        xy = Iterators.map(collect, Iterators.product(xr, yr))
        distxy = product_distribution(dist, disty)
        theory = pdf.(Ref(product_distribution(dist, disty)), xy)

        levels = hpd(theory, 2 .* cdf.(Ref(disty), (1:3)) .- 1)

        Random.seed!(1)  # hide
        rvy = rand(disty, 5_000)

        for (ii, method) in enumerate(estimators)
            dohist = method isa KDE.AbstractBinningKDE

            Random.seed!(123)  # hide
            # empirical distribution
            rvx = rand(dist, 5_000)
            dens = KDE.kde(rvx, rvy; method, bounds = (dist, disty), bwratio = dohist ? (1, 1) : (8, 8))

            ax = Axis(fig[1, ii]; title = string(nameof(typeof(method))),
                                  xlabel = "x-value", ylabel = "y-value")
            contour!(ax, xr, yr, theory; label = "true", levels = levels,
                     color = (:black, 0.5), linestyle = :dash)
            contour!(ax, dens; label = "estimate", levels = levels,
                     color = dohist ? :blue3 : :firebrick3)
            ii > 1 && hideydecorations!(ax, grid = false, ticks = false)

            push!(axs, ax)
        end
        linkaxes!(axs...)
        xlims!(axs[1], extrema(xr) .+ (-0.05, 0.05) .* (xr[end] - xr[1]))
        ylims!(axs[1], extrema(yr) .+ (-0.05, 0.05) .* (yr[end] - yr[1]))

        Label(fig[0, :], name, font = :bold, fontsize = 20)

        colgap!(fig.layout, Fixed(4))
        rowgap!(fig.layout, Fixed(3))
        rowsize!(fig.layout, 1, Aspect(1, 1.0))
        fig.layout.alignmode = Outside(4)
        resize_to_layout!(fig)

        save("comparison_Normal_$name.svg", fig)
    end
    nothing  # hide
    ```

![](comparison_Normal_Normal.svg)
![](comparison_Normal_Chisq3.svg)
![](comparison_Normal_HalfNormal.svg)
![](comparison_Normal_Uniform.svg)

## Bandwidth Estimators
