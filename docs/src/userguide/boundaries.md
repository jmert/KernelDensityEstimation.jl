```@meta
CurrentModule = KernelDensityEstimation
```

# Working with domain boundaries

```@contents
Pages = ["boundaries.md"]
Depth = 2:2
```

```@setup boundary1d
using Distributions
using KernelDensityEstimation
using Random
using CairoMakie
CairoMakie.activate!(type = "svg")

Random.seed!(101)
```

The previous example ([Getting Started - Simple kernel density estimate](index.md#Simple-kernel-density-estimate))
arises often and is handled well by most kernel density estimation solutions.
Being a Gaussian distribution makes it particularly well behaved, but in general distributions which are unbounded
and gently fade away to zero towards ``\pm\infty`` are relatively easy to deal with.
Despite how often the Gaussian distribution is an appropriate [approximation of the] distribution, there are still
many cases where various bounded distributions are expected, and ignoring the boundary conditions can lead to a very
poor density estimate.


!!! hint
    This page describes the built-in boundary condition interface.
    See the [package extension for `Distributions.jl`](@ref ext-distributions) for an alternate specification which
    uses distributions to set appropriate boundary conditions and limits.

## Univariate example

Take the simple case of the uniform distribution on the interval ``[0, 1]``.

```@example boundary1d
Random.seed!(101)  # hide
x = rand(5_000)
nothing  # hide
```

By default, `kde` assumes the distribution is unbounded, and this leads to "smearing" the density across the known
boundaries to the regions ``x < 0`` and ``x > 1``:

```@example boundary1d
K0 = kde(x)
nothing  # hide
```

```@setup boundary1d
fig = Figure(size = (400, 400))
ax = Axis(fig[1, 1])

lines!(ax, [0, 0, 1, 1], [0, 1, 1, 0], color = :blue3, linestyle = :dash, label = "uniform")
lines!(ax, K0, color = :firebrick3, label = "density estimate")
Legend(fig[2, 1], ax, orientation = :horizontal)

save("getting_started_unbound_unif.svg", fig)
```
```@figure
![](getting_started_unbound_unif.svg)
```

We can inform the estimator that we expect a bounded distribution, and it will use that information to generate a
more appropriate estimate.
To do so, we may make use of three keyword arguments in combination:

1. `lo` to dictate the lower bound of the data.
2. `hi` to dictate the upper bound of the data.
3. `boundary` to specify the boundary condition, such as `:open` (unbounded), `:closed` (finite), and half-open
   intervals `:closedleft`/`:openright` and `:closedright`/`:openright`.

!!! important

    The `lo`, `hi`, and `boundary` keywords are conveniences provided only by the univariate KDE method.
    The bounds and boundary conditions may equivalently (and more fundamentally) be set via a tuple to the
    `bounds = (lo, hi, boundary)` keyword argument which is used for the bivariate and multivariate interfaces as well.

In this example, we know our data is bounded on the closed interval ``[0, 1]``, so we can improve the density
estimate by providing that information

```@example boundary1d
K1 = kde(x, lo = 0, hi = 1, boundary = :closed)  # or bounds = (0, 1, :closed)
nothing  # hide
```

```@setup boundary1d
fig = Figure(size = (400, 400))
ax = Axis(fig[1, 1])

lines!(ax, K0, color = :grey75)
lines!(ax, [0, 0, 1, 1], [0, 1, 1, 0], color = :blue3, linestyle = :dash, label = "uniform")
lines!(ax, K1, color = :firebrick3, label = "density estimate")
Legend(fig[2, 1], ax, orientation = :horizontal)

save("getting_started_limit_unif.svg", fig)
```
```@figure
![](getting_started_limit_unif.svg)
```

Note that in addition to preventing the smearing of the density beyond the bounds of the known distribution, the
density estimate with correct boundaries is also smoother than the unbounded estimate.
This is because the sharp drops at ``x = \{0, 1\}`` no longer need to be represented, so the algorithm is no longer
compromising on smoothing the interior of the distribution with retaining the cut-offs.

## Bivariate example


## Boundary interface

...

The boundary condition for each dimension is one of the enum values in the [`Boundary`](@ref) module or an
eponymously-named lowercase symbol:

- `:open ≡ Open`
- `:closed ≡ Closed`
- `:closedleft ≡ ClosedLeft` and also aliased as `:openright ≡ OpenRight`
- `:closedright ≡ ClosedRight` and also aliased as `:openleft ≡ OpenLeft`

A boundary specification is (at its core) is one of the following forms:

1. [`BoundsSpec`](@ref) — A "complete" (3-tuple) specification comprised of a lower/left limit, upper/right limit, and a
   boundary condition.  A subset of the tuple elements may be `nothing` or `-Inf`/`+Inf`, in which case the "missing"
   elements are inferred from the data or remainder of the specification.

   See [`bounds`](@ref bounds(::Tuple{Vararg{AbstractVector,N}}, ::Tuple{Vararg{BoundsSpec,N}}) where {N}) for a
   complete description of how missing elements may be inferred.

2. [`BoundsLims`](@ref) — An "incomplete" (2-tuple) specification comprised only of the lower/left and upper/right
   limits.
   The boundary condition is assumed to be [`Open`](@ref Boundary)

3. [`BoundsArgs`](@ref) — Either of the previous two forms or the "missing" specification `nothing` which is
   interpreted as the inferred open interval `(nothing, nothing, Open)`.

The built-in base case is the
[`bounds`](@ref bounds(::Tuple{Vararg{AbstractVector,N}}, ::Tuple{Vararg{BoundsSpec,N}}) where {N})
method that accepts a tuple of data vectors and a tuple of [`BoundsSpec`](@ref) arguments:

...

As special cases when working with 1D data, either the wrapping tuple on the bounds specification may be omitted
or both wrapping tuples may be omitted.

```@repl univariate_bounds
using KernelDensityEstimation: bounds  # hide
bounds(([1.0, 2.0],), (0.0, nothing))
bounds([1.0, 2.0], (0.0, nothing))
```

...
