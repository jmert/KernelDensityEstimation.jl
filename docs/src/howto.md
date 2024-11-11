# How-to Guides

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

