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

!!! hint
    In addition to the aforementioned triple of `lo`, `hi`, and `boundary` keywords, there is a single `bounds`
    keyword which can replace all three.
    The built-in mechanism only accepts a tuple where `bounds = (lo, hi, boundary)`, but the additional keyword
    makes it possible to customize behavior for new types of arguments.
    For example, there is a [package extension for `Distributions.jl`](@ref ext-distributions) which allows using
    the support of a distribution to automatically infer appropriate boundary conditions and limits.

    See the docstring for [`kde`](@ref) (and references therein) for more information.


## Densities of weighted samples

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
![](getting_started_weighting.svg)

As expected, this shifts the resultant density estimate to the right, toward more positive values.

!!! note

    The effective sample size ([`UnivariateKDEInfo.neffective`](@ref UnivariateKDEInfo)) is calculated from the weights
    using [Kish's definition](https://search.r-project.org/CRAN/refmans/svyweight/html/eff_n.html).
    Both of the bandwidth estimators ([`SilvermanBandwidth`](@ref) and [`ISJBandwidth`](@ref)) use this definition
    in scaling the bandwidth with the (effective) sample size.
