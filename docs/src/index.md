# Kernel Density Estimation

```@eval
import Markdown
readmetxt = read(joinpath(dirname(@__FILE__), "..", "..", "README.md"), String)
readme = Markdown.parse(readmetxt)

# Keep the contents between the title heading and the first horizontal rule (exclusive)
ii = findfirst(x -> x isa Markdown.Header{1}, readme.content)
jj = findfirst(x -> x isa Markdown.HorizontalRule, readme.content)
readme.content = readme.content[ii+1:jj-1]

readme
```

## Why another kernel density estimation package?

As of Mar 2026, much of the Julia ecosystem uses the
[`KernelDensity.jl`](https://juliahub.com/ui/Packages/General/KernelDensity)
package (possibly implicitly, such as through density plots in Makie.jl, StatsPlots.jl, etc).

Consider the following (toy) examples: one case where we have samples drawn from a Gaussian
distribution, and a second where we restrict to positive values.
We'll also consider their joint distribution in 2D.

```@example example_truncate
using Random
using Distributions
Random.seed!(1234)  # hide

# A Gaussian distribution and sample of random deviates
x, dist = -5.0:0.01:5.0, Normal(0.0, 1.0)
exp_gauss = pdf.(dist, x)
rv_gauss = rand(dist, 500)

# Then the truncated distribution and more samples
tdist = truncated(dist, lower = 0.0)
exp_trunc = pdf.(tdist, x)
rv_trunc = rand(tdist, 500)

# the joint distribution
jdist = product_distribution(dist, tdist)
exp_joint = map(c -> pdf(jdist, [c...]), Iterators.product(x, x))
nothing  # hide
```

!!! details "Plotting setup and shared code"

    ```@example example_truncate
    using CairoMakie

    isdefined(Base.Main, :hpd) || Base.include(Base.Main, joinpath(@__DIR__, "hpd.jl"))
    using Base.Main: hpd

    function draw_densities(density_gauss, density_trunc, density_joint)
        fig = Figure(size = (800, 500))
        kws_theory = (; linestyle = :dash, color = :firebrick)
        kws_density = (; linewidth = 2, color = Makie.wong_colors()[1])

        # Gaussian distribution & KDE
        ax1 = Axis(fig[1, 1])
        lines!(ax1, x, exp_gauss; kws_theory...)
        lines!(ax1, density_gauss.x, density_gauss.density; kws_density...)

        # Truncation Gaussian distribution & KDE
        ax2 = Axis(fig[2, 2])
        lines!(ax2, exp_trunc, x; kws_theory...)
        lines!(ax2, density_trunc.density, density_trunc.x; kws_density...)

        # Joint distribution over both
        ax12 = Axis(fig[2, 1], xlabel = "Gaussian", ylabel = "Truncated Gaussian")
        hm = heatmap!(ax12, density_joint.x, density_joint.y, density_joint.density,
                      colormap = Reverse(:grays), rasterize = true)
        hlines!(0.0, linestyle = :dot, color = :black)
        tt = contour!(ax12, x, x, exp_joint,
                      levels = hpd(exp_joint); kws_theory...)
        cc = contour!(ax12, density_joint.x, density_joint.y, density_joint.density,
                      levels = hpd(density_joint.density); kws_density...)

        linkxaxes!(ax12, ax1)
        linkyaxes!(ax12, ax2)
        hidexdecorations!(ax1, ticks = false, grid = false)
        hideydecorations!(ax2, ticks = false, grid = false)
        xlims!(ax12, -4, 4)
        ylims!(ax12, -1, 4)
        colsize!(fig.layout, 1, Aspect(2, 8/5))
        colsize!(fig.layout, 2, Aspect(1, 1.0))

        return fig
    end
    nothing  # hide
    ```

If we then plot the outputs of running the `KernelDensity.kde` method on the individual and joint
data sets:

```@example example_truncate
import KernelDensity as KD

kd_gauss = KD.kde(rv_gauss)
kd_trunc = KD.kde(rv_trunc)
kd_joint = KD.kde((rv_gauss, rv_trunc))
nothing  # hide
```

```@figure; htmllabel = "Figure 1"
![](example_kerneldensity.svg)

Kernel density estimates (solid blue) for the full (top left), truncated (lower right), and joint (lower left) Gaussian
samples as produced using the default settings from `KernelDensity.jl`.
Each can be compared to the equivalent theory distribution (dashed red), where the 68% and 95% contour levels are
indicated in the 2D plane.
```

!!! details "Plotting Code"

    ```@example example_truncate
    fig = draw_densities(kd_gauss, kd_trunc, kd_joint)
    Label(fig[0, :], "KernelDensity.jl", font = :bold, fontsize = 20)
    resize_to_layout!(fig)
    save("example_kerneldensity.svg", fig)  # hide
    nothing  # hide
    ```

For the Gaussian distribution (top left) where there are no edges, the density estimate appears to be a reasonable
approximation of the known Gaussian distribution.
In comparison, though, the truncated Gaussian distribution (lower right) fails to represent the hard cut-off at
``y = 0``, instead "leaking" below zero with non-zero density despite the known closed boundary.
The joint distribution (lower left) likewise has non-zero density in the excluded region.

Closed boundaries are common among many probability distributions,[^bounded] and therefore the need to estimate a
density corresponding to a (semi-)bounded distribution arises often.
This package provides a density estimator that uses any provided boundary conditions to account for edge boundary
effects, reproducing a more faithful representation of the underlying distribution.

[^bounded]: For example, see the list of distributions with
    [bounded](https://en.wikipedia.org/wiki/List_of_probability_distributions#Supported_on_a_bounded_interval)
    and
    [semi-infinite](https://en.wikipedia.org/wiki/List_of_probability_distributions#Supported_on_semi-infinite_intervals,_usually_[0,%E2%88%9E%29)
    support on Wikipedia.

Repeating the density estimation on the Gaussian and truncated Gaussian distributions shown above instead with this
package's [`kde`](@ref) method:

```@example example_truncate
import KernelDensityEstimation as KDE

kde_gauss = KDE.kde(rv_gauss)
kde_trunc = KDE.kde(rv_trunc, lo = 0.0, boundary = :closedleft)
kde_joint = KDE.kde(rv_gauss, rv_trunc, bounds = ((-Inf, Inf), (0.0, Inf, :closedleft)))
nothing  # hide
```

```@figure; htmllabel = "Figure 2"
![](example_kerneldensityestimation.svg)

Kernel density estimates using the same data as Figure 1 but now processed with this package, including additional
information about the boundary conditions of the distribution.
```

!!! details "Plotting Code"

    ```@example example_truncate
    fig = draw_densities(kde_gauss, kde_trunc, kde_joint)
    Label(fig[0, :], "KernelDensityEstimation.jl", font = :bold, fontsize = 20)
    resize_to_layout!(fig)
    save("example_kerneldensityestimation.svg", fig)  # hide
    nothing  # hide
    ```

Most obviously, the truncated distribution retains its closed boundary condition at ``y = 0`` and does not suffer
from the leakage and suppression of the peak that occurs with the `KernelDensity` estimator.
Furthermore, all density curves are smoother due to use of higher-order estimators which simultaneously
permit using [relatively] wider bandwidth kernels while retaining the shapes of peaks (and non-flat slopes at
closed boundaries).
