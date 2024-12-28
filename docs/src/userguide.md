```@meta
CurrentModule = KernelDensityEstimation
```

# User Guide

```@contents
Pages = ["userguide.md"]
Depth = 2:2
```

## Getting Started

To install KernelDensityEstimation.jl, it is recommended that you use the
[jmert/Registry.jl](https://github.com/jmert/Registry.jl) package registry, which will let you install (and depend on)
the package similarly to any other Julia package in the default General registry.

```julia-repl
pkg> registry add https://github.com/jmert/Registry.jl

pkg> add KernelDensityEstimation
```

## Simple kernel density estimate

```@setup get_started
using Distributions
using Markdown
using Random
using CairoMakie
CairoMakie.activate!(type = "svg")

Random.seed!(101)
```

For the following example, we'll use a small sample of Gaussian deviates:

```@example get_started
using KernelDensityEstimation
x = 3 .+ 0.1 .* randn(250) # x ~ Normal(3, 0.1)
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
![](getting_started_1.svg)


## Densities with boundaries

The previous example arises often and is handled well by most kernel density estimation solutions.
Being a Gaussian distribution makes it particularly well behaved, but in general distributions which are unbounded
and gently fade away to zero towards ``\pm\infty`` are relatively easy to deal with.
Despite how often the Gaussian distribution is an appropriate [approximation of the] distribution, there are still
many cases where various bounded distributions are expected, and ignoring the boundary conditions can lead to a very
poor density estimate.

Take the simple case of the uniform distribution on the interval ``[0, 1]``.

```@example get_started
Random.seed!(101)  # hide
x = rand(5_000)
nothing  # hide
```

By default, `kde` assumes the distribution is unbounded, and this leads to "smearing" the density across the known
boundaries to the regions ``x < 0`` and ``x > 1``:

```@example get_started
K0 = kde(x)
nothing  # hide
```

```@setup get_started
fig = Figure(size = (400, 400))
ax = Axis(fig[1, 1])

lines!(ax, [0, 0, 1, 1], [0, 1, 1, 0], color = :blue3, linestyle = :dash, label = "uniform")
lines!(ax, K0, color = :firebrick3, label = "density estimate")
Legend(fig[2, 1], ax, orientation = :horizontal)

save("getting_started_unbound_unif.svg", fig)
```
![](getting_started_unbound_unif.svg)

We can inform the estimator that we expect a bounded distribution, and it will use that information to generate a
more appropriate estimate.
To do so, we make use of three keyword arguments in combination:

1. `lo` to dictate the lower bound of the data.
2. `hi` to dictate the upper bound of the data.
3. `boundary` to specify the boundary condition, such as `:open` (unbounded), `:closed` (finite), and half-open
   intervals `:closedleft`/`:openright` and `:closedright`/`:openright`.

In this example, we know our data is bounded on the closed interval ``[0, 1]``, so we can improve the density
estimate by providing that information

```@example get_started
K1 = kde(x, lo = 0, hi = 1, boundary = :closed)
nothing  # hide
```

```@setup get_started
fig = Figure(size = (400, 400))
ax = Axis(fig[1, 1])

lines!(ax, K0, color = :grey75)
lines!(ax, [0, 0, 1, 1], [0, 1, 1, 0], color = :blue3, linestyle = :dash, label = "uniform")
lines!(ax, K1, color = :firebrick3, label = "density estimate")
Legend(fig[2, 1], ax, orientation = :horizontal)

save("getting_started_limit_unif.svg", fig)
```
![](getting_started_limit_unif.svg)

Note that in addition to preventing the smearing of the density beyond the bounds of the known distribution, the
density estimate with correct boundaries is also smoother than the unbounded estimate.
This is because the sharp drops at ``x = \{0, 1\}`` no longer need to be represented, so the algorithm is no longer
compromising on smoothing the interior of the distribution with retaining the cut-offs.

See the docstring for [`kde`](@ref) (and references therein) for more information on the behavior of the `lo`, `hi`,
and `boundary` keyword arguments.
