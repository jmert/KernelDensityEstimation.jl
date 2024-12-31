```@meta
CurrentModule = KernelDensityEstimation
```

# Release Notes

```@contents
Pages = ["releasenotes.md"]
Depth = 2:2
```

---

## v0.6.0 — 2024 Dec 31

- Public functions have been declared using Julia v1.11+'s `public` keyword.
- The value of the bin-center for the zero-width singleton histogram has been fixed.
- Implement more accurate histogram (and linear) binning calculations.
  For the [`HistogramBinning`](@ref) case, it is now possible to precisely bin `range(lo, hi, nbins)` values into their
  corresponding bins (whereas previously values may be counted incorrectly one bin too low due to rounding in the
  floating point calculations).
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

- Add an interface method [`boundary`](@ref) which can be overloaded to implement mechanisms for automatically
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
