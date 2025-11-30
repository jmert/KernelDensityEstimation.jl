```@meta
CurrentModule = KernelDensityEstimation
```
```@setup
using CairoMakie
update_theme!(size = (400, 300))
```

# Package Extensions

```@contents
Pages = ["extensions.md"]
Depth = 2:2
```

!!! important

    This section describes features that are only available when using Julia v1.9 or newer.

## [Distributions.jl](@id ext-distributions)

A univariate distribution from
[`Distributions.jl`](https://juliahub.com/ui/Packages/General/Distributions)
can be used as a value for the `bounds` argument of [`kde`](@ref), wherein the boundary conditions of the distribution
will be used to automatically set appropriate values of `lo`, `hi`, and `boundary`.

For example, generating a density estimate for a non-negative parameter in a Markov chain Monte Carlo (MCMC) chain
is often paired with a similarly non-negative prior.
Instead of needing to explicitly determine and pass through the correct combination of lower and upper bounds and
their boundary conditions, the prior distribution can be used instead.

```@example ext_distributions
using KernelDensityEstimation
using Distributions
using Random  # hide
Random.seed!(1234)  # hide

# a non-negative constraint on a prior
prior = truncated(Normal(0.0, 1.0), lower = 0.0)
# proxy for an MCMC chain
chain = rand(prior, 200)

# prior-based boundary information on left is same as explicit options on right
kde(chain; bounds = prior) == kde(chain; lo = 0.0, boundary = :closedleft)
```


## [Makie.jl](@id ext-makie)

Plotting the [`UnivariateKDE`](@ref) object is natively supported within the
[`Makie.jl`](https://juliahub.com/ui/Packages/General/Makie)
system of packages.
The density estimate is converted via the
[`PointsBased`](https://docs.makie.org/stable/explanations/recipes#Multiple-Argument-Conversion-with-convert_arguments)
trait and defaults to a line plot.

Plotting via `stairs` is a special case, which correctly offsets the bin centers to the trailing bin edge (compatible
with the default `step = :pre` behavior) and adds points to close the histogram with the x-axis.

```@example ext_makie
using KernelDensityEstimation: kde, LinearBinning
using Random  # hide
Random.seed!(100)  # hide

# 500 samples from a Chisq(ν=4) distribution
rv = dropdims(sum(abs2, randn(4, 500), dims=1), dims=1)
nothing  # hide
```

```@example ext_makie
using CairoMakie

K = kde(rv; lo = 0.0, boundary = :closedleft)
H = kde(rv; lo = 0.0, boundary = :closedleft,
            bwratio = 1.0, method = LinearBinning())

fig = Figure(size=(800, 300))
ax1 = Axis(fig[1, 1], title="stairs", ylabel = "density", xlabel = "value")
ax2 = Axis(fig[1, 2], title="lines")
linkaxes!(ax1, ax2)
hideydecorations!(ax2, grid = false, ticks = false)

stairs!(ax1, H)
lines!(ax2, K)

save("ext_makie.svg", current_figure())  # hide
nothing  # hide
```
```@figure
![](ext_makie.svg)
```


## [Plots.jl](@id ext-plots)

Plotting the [`UnivariateKDE`](@ref) object is natively supported within the
[`Plots.jl`](https://juliahub.com/ui/Packages/General/Plots)
ecosystem of backends by defining a plot recipe for `RecipesBase.jl`.

The density estimate is interpreted by default as a `:line` series type with
xlabel `"value"` and ylabel `"density"`.

```@example ext_plots
using KernelDensityEstimation: kde, LinearBinning
using Random  # hide
Random.seed!(100)  # hide

# 500 samples from a Chisq(ν=4) distribution
rv = dropdims(sum(abs2, randn(4, 500), dims=1), dims=1)
nothing  # hide
```

```@example ext_plots
using Plots

K = kde(rv; lo = 0.0, boundary = :closedleft)
H = kde(rv; lo = 0.0, boundary = :closedleft,
            bwratio = 1.0, method = LinearBinning())

plot(
    plot(H, title = "stairs", seriestype = :stepmid),
    plot(K, title = "lines", ylabel = nothing),
    layout = (1, 2), size = (800, 300),
    leftmargin = (2.5, :mm), bottommargin = (3.0, :mm),
    link = :all, legend = false
)

withenv(() -> savefig("ext_plots.svg"), "GKSwstype" => "nul");  # hide
closeall();  # hide
nothing  # hide
```
```@figure
![](ext_plots.svg)
```


## [UnicodePlots.jl](@id ext-unicodeplots)

For quick, approximate visualization of a density within the terminal, an extension is provided for the
[`UnicodePlots.jl`](https://juliahub.com/ui/Packages/General/UnicodePlots)
package and extends the `lineplot` (and `lineplot!`) methods.

```@example ext_unicodeplots
using KernelDensityEstimation
using Random  # hide
Random.seed!(100)  # hide

# 500 samples from a Chisq(ν=4) distribution
rv = dropdims(sum(abs2, randn(4, 500), dims=1), dims=1)
nothing  # hide
```

```@example ext_unicodeplots
using UnicodePlots

K = kde(rv; lo = 0.0, boundary = :closedleft)
lineplot(K)
```
