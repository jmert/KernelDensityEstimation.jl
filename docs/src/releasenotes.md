```@meta
CurrentModule = KernelDensityEstimation
```

# Release Notes

```@contents
Pages = ["releasenotes.md"]
Depth = 2:2
```

---

## v0.8.0 — Unreleased

- Various changes to the interfaces and type definitions have been made to allow for future support of bivariate (and
  possibly multivariate) density estimates.

    - The `boundary` function has been removed, with all of its functionality subsumed by the
      [`bounds`](@ref) function, which has also impacted the built-in definitions and behaviors of the `bounds`
      methods.

    - The type parameterizations of [`AbstractKDE`](@ref) and [`UnivariateKDE`](@ref) have changed in a
      backwards-incompatible way.

      Previously, the univariate structure had four type parameters and subtype relationship,
      ```julia
      UnivariateKDE{T_input, T_density,
                    R<:AbstractRange{T_input},
                    V<:AbstractVector{T_density}
                   } <: AbstractKDE{T_input}
      ```
      where the first two parameters are the element types of the input data and output density estimate, respectively.
      The trailing two then specialize the container types of the binning and density vectors, but critically they use
      separate type variables in order to support unitful quantities where the units are inverses of one another.
      Notably, the type relationship to the abstract supertype was declared to use the **input** element type.

      The new type definition is instead an alias of the more generic [`MultivariateKDE`](@ref) struct and takes the
      form
      ```julia
      UnivariateKDE{T_density,
                    R<:AbstractRange,
                    V<:AbstractVector{T_density}
                   } <: AbstractKDE{T_density}
      ```
      The trailing two type parameters are the same, but now the **output** element type of the density array is used in
      the supertype relationship.
      The input element type also no longer appear as an explicit type parameter.

      These changes are required to align with a future definition of multivariate density estimates.
      The explicit input element type is dropped since each axis may have different element type (i.e. units) and are
      implicitly available via the axis range(s).
      In contrast, the element type of the density array is unique, so this is the natural choice to use in the
      supertype relationship.

     - An **experimental** [`MultivariateKDE`](@ref) type has been added to support higher dimensional density
       estimation.
       [`UnivariateKDE`](@ref) and [`BivariateKDE`](@ref) are type aliases for the 1- and 2-dimensional cases.

---

## v0.7.0 — 2025 Jun 08

- A new `weights` keyword argument has been added to the [`kde`](@ref) function to support weighted data sets.
  As a consequence, the API of multiple interfaces have been changed:
  - [`bandwidth`](@ref) has gained a `weights` keyword.
  - The [`init`](@ref) method returns three values (`data`, `weights`, and `info`) instead of just two.
  - The [`estimate`](@ref) function takes a mandatory `weights` positional argument.
- Revert the "accurate histogram" binning calculation made in the previous release.
  Further testing has shown that much of the extra work being done was ineffective, so no promise is currently
  made about being able to precisely bin `range(lo, hi, nbins)` (nor `LinRange(lo, hi, len)`).

---

## v0.6.0 — 2024 Dec 31

- Public functions have been declared using Julia v1.11+'s `public` keyword.
- The value of the bin-center for the zero-width singleton histogram has been fixed.
- Implement more accurate histogram (and linear) binning calculations.
  For the [`HistogramBinning`](@ref) case, it is now possible to precisely bin `range(lo, hi, nbins)` values into their
  corresponding bins (whereas previously values may be counted incorrectly one bin too low due to rounding in the
  floating point calculations). **(Reverted in v0.7.0)**
- The implementation has been modified to support unitful quantities (without adding a new package dependency) via
  careful consideration and application of appropriate factors of `one` and/or `oneunit` (and relaxing type constraints
  or adding new type parameters to structs, where necessary).
  Given a vector with units `u`, the density object's fields `(K.x, K.f)` have units `(u, u^-1)`, respectively, in
  correspondence with interpreting any integrated range to be a unitless probability
  (i.e. `probability = sum(K.f[a .< K.x .< b]) * step(K.x)`).
- The package no longer depends on `Roots.jl` (used by the ISJ bandwidth estimator); instead, an implementation of
  [Brent's Method](https://en.wikipedia.org/wiki/Brent%27s_method) is now included here directly. This not only
  decreases the dependence on external packages but also reduces both package load time and the precompiled package
  image size.
- The documentation has generally been improved:
  - A new ["Showcase" section](showcase.md) has been added to showcase examples of density estimation with this package.
  - A [User Guide](userguide.md) has been started to give a brief introduction to installing and using the package.
  - The documentation now includes release notes.

---

## v0.5.0 — 2024 Nov 21

- Have the ISJ bandwidth estimator fallback to Silverman rule automatically when it fails to converge.
  This is expected to happen for very flat (closed) distributions, so it's not as rare of an occurrence as originally
  understood.
- Add [package extension](extensions.md#ext-makie) to aid in plotting with
  [`Makie.jl`](https://juliahub.com/ui/Packages/General/Makie.jl).
- Begin adding more extensive documentation to the package:
  - Add a README to give brief justification for the package.
  - Describe the [package extensions](extensions.md) available and what features they provide.
  - Demonstrate the impact of the different stages of the [estimator pipeline](explain.md#estimator-pipeline) by
    comparing each stage for several example distributions.

---

## v0.4.0 — 2024 Sep 09

- Add an interface method `boundary` which can be overloaded to implement mechanisms for automatically
  determining appropriate boundary conditions.
  - The two built-in methods are to convert from symbols to enum (e.g. `:open` to [`Open`](@ref Boundary)) and to infer
    the boundary conditions from a 2-tuple of finite/infinite real values.
- Add an interface method [`bounds`](@ref) (and eponymous keyword argument to `kde`) which can be overloaded to
  implement mechanisms for automatically both the boundary condition (keyword `boundary`) and limits (keywords `lo`
  and `hi`) from an arbitrary value.
- Store more information within the [`UnivariateKDEInfo`](@ref) structure. The [`init`](@ref) uses the new fields to
  pass relevant parameters to later stages of the estimator pipeline.
- On Julia v1.9+, new [extension packages](extensions.md) have been added to integrate with external packages:
  - The aforementioned `boundary` and `bounds` methods have been specialized for univariate distributions from
    [`Distributions.jl`](https://juliahub.com/ui/Packages/General/Distributions)
  - The [`UnicodePlots.jl`](https://juliahub.com/ui/Packages/General/UnicodePlots) package, if loaded, is used to
    visualize the kernel density estimate at the terminal.
- Increase the automatic bandwidth determined by the chosen bandwidth estimator for higher-order estimators (such as
  [`MultiplicativeBiasKDE`](@ref)) which have a lower level of bias.
  (See [Lewis2019; §E, Eqn 35 and Footnote 10](@citet) for further details.)
- Rename the boundary condition enum from `Cover` to [`Boundary`](@ref).
- Fix syntax or usage errors that broke compatibility with Julia v1.6.

> A. Lewis. _GetDist: a Python package for analysing Monte Carlo samples_ (2019),
> [arXiv:1910.13970](https://arxiv.org/abs/1910.13970).

---

## v0.3.0 — 2024 Jul 21

This release adds the ISJ bandwidth estimator described in [Botev2010](@citet) and uses it by default, as it is more
capable of dealing both with non-Gaussian distributions and respects the complexity/nature of bounded domains.

- Add an implementation of the [Improved Sheather-Jones](@ref ISJBandwidth) bandwidth estimator.
- Rename the interface method for bandwidth estimators to [`bandwidth`](@ref).
- Consolidate data and option pre-processing into the [`init`](@ref) method, which is the first step in the density
  estimation pipeline.

> Z. Botev, J. Grotowski and D. Kroese. _Kernel density estimation via diffusion._
>  [The Annals of Statistics 38](https://doi.org/10.1214/10-aos799) (2010),
>  [arXiv:1011.2602](https://arxiv.org/abs/1011.2602).

---

## v0.2.0 — 2024 Jul 19

- Require Julia v1.6+.
- Improved docstrings throughout.
- Added framework for building documentation with Documenter.jl.
- Migrate specific density estimation implementations to be methods of the (new) [`estimate`](@ref) interface function
  (rather than overloading `kde`).
- Fix handing of edge-case where a constant vector is given. For closed boundary conditions, the result is a
  zero-width singleton bin.
- Fix error in widening of the KDE range based on the kernel bandwidth.

---

## v0.1.0 — 2024 Jul 13

Initial release supports:

- Univariate kernel density estimation.
- 2 binning methods ([histogramming](@ref HistogramBinning) and [linear binning](@ref LinearBinning)) and
  3 density estimation techniques ([basic](@ref BasicKDE), with [linear boundary correction](@ref LinearBoundaryKDE),
  and/or with [multiplicative bias correction](@ref MultiplicativeBiasKDE)
- Support for distributions with (half-)closed [boundary conditions](@ref Boundary).
- Automatic bandwidth selection using [Silverman's rule](@ref SilvermanBandwidth).
