```@meta
CurrentModule = KernelDensityEstimation
```

# Getting Started

```@contents
Pages = ["index.md"]
Depth = 2:3
```

To install KernelDensityEstimation.jl, it is recommended that you use the
[jmert/Registry.jl](https://github.com/jmert/Registry.jl) package registry, which will let you install (and depend on)
the package similarly to any other Julia package in the default General registry.

```julia-repl
pkg> registry add https://github.com/jmert/Registry.jl

pkg> add KernelDensityEstimation
```

## Univariate Densities

### Simple kernel density estimate

```@setup get_started
using Random
using CairoMakie
CairoMakie.activate!(type = "svg")

Random.seed!(101)
```

For the following example, we'll use a small sample of Gaussian deviates:

```@example get_started
using Distributions

x = rand(Normal(3, 0.1), 250)
nothing  # hide
```

The key interface of this package is the [`kde`](@ref) function.
In its simplest incantation, you provide a vector of data and it returns a kernel density object (in the form of a
[`UnivariateKDE`](@ref) structure).

```@example get_started
using KernelDensityEstimation

K = kde(x)
nothing  # hide
```

The density estimate ``f(x)`` is given at locations `K.x` (as a [`StepRangeLen`](@extref Base.StepRangeLen)) with
density values `K.f`.
For instance, the mean and variance of the distribution are:

```@example get_started
μ1 = step(K.x) * sum(@. K.f * K.x)
μ2 = step(K.x) * sum(@. K.f * K.x^2)

(; mean = μ1, std = sqrt(μ2 - μ1^2))
```

which agree well with the known underlying parameters ``(\mu = 3, \sigma = 0.1)``.

Visualizing the density estimate (see [Extensions — Makie.jl](@ref ext-makie)), we see a fair level of consistency
between the density estimate and the known underlying model.

```@setup get_started
fig = Figure(size = (400, 400))
ax = Axis(fig[1, 1])

lines!(K.x, pdf.(Normal(3, 0.1), K.x), color = :blue3, linestyle = :dash, label = "model")
lines!(K, color = :firebrick3, label = "density estimate")
Legend(fig[2, 1], ax, orientation = :horizontal)

save("getting_started_1.svg", fig)
```
```@figure
![](getting_started_1.svg)
```

### Densities of weighted samples

In some cases, the data to be analyzed is a _weighted_ vector of data (represented as a vector of data and a
corresponding vector of weight factors).
For instance, [importance sampling](https://en.wikipedia.org/wiki/Importance_sampling) of an
[MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) chain results in non-uniform weights that then must
be considered when deriving a density estimate.

Take the following toy example where we have a target parameter ``v`` and nuisance parameter ``p`` that are correlated,
where a uniform prior was assumed for ``p``:

```@example get_started
Random.seed!(200)  # hide
# correlation coefficient and nuisance parameter
ρ, p = 0.85, randn(500)
# correlated target parameter
v = ρ .* p .+ sqrt(1 - ρ^2) .* randn.()
nothing  # hide
```

Now suppose that you have reason to update your prior on ``p``, believing now that positive values are twice as likely
as negative ones.
If the method of generating ``v`` is expensive, and because the change in prior is not extreme, it may be efficient
and acceptable to instead importance sample the existing values by reweighting the samples by the ratio of the
priors:

```math
\begin{align*}
P_1(p) &\propto 1
&
P_2(p) &\propto \begin{cases}
    1 & p < 0 \\
    2 & p \ge 0 \\
    \end{cases}
\end{align*}
```

```@example get_started
P1(z) = 1.0
P2(z) = z ≥ 0 ? 2.0 : 1.0
weights = P2.(p) ./ P1.(p)
nothing  # hide
```

We then simply provide these weights as a keyword argument in the call to `kde`:
```@example get_started
K1 = kde(v)
K2 = kde(v; weights)
nothing  # hide
```

```@setup get_started
fig = Figure(size = (400, 400))
ax = Axis(fig[1, 1])

lines!(ax, K1, color = :blue3, label = "uniform prior")
lines!(ax, K2, color = :firebrick3, label = "reweight (positive more likely)")
Legend(fig[2, 1], ax, orientation = :horizontal)

save("getting_started_weighting.svg", fig)
```
```@figure
![](getting_started_weighting.svg)
```

As expected, this shifts the resultant density estimate to the right, toward more positive values.

!!! note

    The effective sample size ([`UnivariateKDEInfo.neffective`](@ref UnivariateKDEInfo)) is calculated from the weights
    using [Kish's definition](https://search.r-project.org/CRAN/refmans/svyweight/html/eff_n.html).
    Both of the bandwidth estimators ([`SilvermanBandwidth`](@ref) and [`ISJBandwidth`](@ref)) use this definition
    in scaling the bandwidth with the (effective) sample size.

## Bivariate Densities

When two or more parameters are of interest, the bivariate density is commonly used to visualize correlations among
the parameters.

As an example of such correlations, take the following simulation which generates a noisy dataset around a known
line, and we use ordinary least-squares regression of the data to obtain the slope and intercept of the best-fit line.
We visualize the accumulated distribution of the recovered fit parameters.

```@example get_started
Nsamp = 1_000
Npts = 10

# simulation
x = range(5, 15, length = Npts)
D = [x.^0 x.^1]  # design matrix
Random.seed!(101)  # hide
yobs = (0.2 .* x .+ 3.0) .+ rand(Normal(0, 2.2), Npts, Nsamp)

# fitting
params = (D'D) \ (D'yobs)
offsets = @view params[1, :]
slopes = @view params[2, :]
nothing  # hide
```

Each of the vectors `offsets` and `slopes` contain instances of the parameter, correlated over simulation realizations.
We can produce the 1D density estimates as above, and we can create the 2D density estimate by simply providing both
vectors (with the order of arguments corresponding to each increasing dimension in the output density).

```@example get_started
K_slope = kde(slopes)
K_offset = kde(offsets)
K_both = kde(slopes, offsets)
nothing  # hide
```

```@setup get_started
isdefined(Base.Main, :hpd) || Base.include(Base.Main, joinpath(@__DIR__, "hpd.jl"))
using Base.Main: hpd

fig = Figure(size = (400, 400))

ax11 = Axis(fig[1, 1])
plot!(ax11, K_slope)
ax22 = Axis(fig[2, 2])
plot!(ax22, K_offset)
ax21 = Axis(fig[2, 1], ylabel = "offset", xlabel = "slope")
plot!(ax21, K_both, levels = hpd(K_both.f),
      linewidth = 1.5, color = Makie.wong_colors()[1])

linkxaxes!(ax11, ax21)
hidexdecorations!(ax11, ticks = false, grid = false)
hideydecorations!(ax11, ticks = false, grid = false)
hideydecorations!(ax22, ticks = false, grid = false)
rowgap!(fig.layout, Fixed(4))
colgap!(fig.layout, Fixed(4))

xlims!(ax11, hpd(K_slope.x, K_slope.f, 1 - 1e-4)...)
let lim = hpd(K_offset.x, K_offset.f, 1 - 1e-4)
    xlims!(ax22, lim...)
    ylims!(ax21, lim...)
end

save("getting_started_2dsim.svg", fig)
```
```@figure
![](getting_started_2dsim.svg)
```

From the 2D density, we clearly see (expected) correlation between the slope and offset parameters — a relatively low
slope results in a higher offset and vice versa.

!!! important
    Higher-order density estimates are also supported by just adding additional vector arguments to the `kde()` call,
    but the support and utility is limited.
    As they are also harder to usefully visualize, they will not be discussed further here.
