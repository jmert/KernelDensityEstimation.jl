```@setup
using CairoMakie
update_theme!(size = (400, 300))
```

# How-to Guides

```@contents
Pages = ["howto.md"]
Depth = 3
```

## Extensions

!!! important

    This section describes features that are only available when using Julia v1.9 or newer.

### Distributions.jl

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


### Makie.jl

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

![](ext_makie.svg)

### UnicodePlots.jl

For quick, approximate visualization of a density within the terminal, an extension is provided for the
[`UnicodePlots.jl`](https://juliahub.com/ui/Packages/General/UnicodePlots)
package.
The three-argument `Base.show` method is defined to show the Unicode plot by default, so the distribution will be
previewed at the REPL automatically.

```@example ext_unicodeplots
using KernelDensityEstimation
using UnicodePlots
using Random  # hide
Random.seed!(100)  # hide

# 500 samples from a Chisq(ν=4) distribution
rv = dropdims(sum(abs2, randn(4, 500), dims=1), dims=1)
K = kde(rv; lo = 0.0, boundary = :closedleft)
```

