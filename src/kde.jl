raw"""
    ν, σ² = _count_var(pred, v)

Return the count `ν` and variance `σ²` of a filtered vector `v` — filtered according to the
predicate function `pred` — simultaneously and without allocating a filtered array.
"""
function _count_var(pred, v)
    T = eltype(v)
    n = 0
    μ = σ² = zero(T)
    for x in v
        !pred(x) && continue
        n += 1
        f = one(T) / n
        # update online mean
        μ₋₁, μ = μ, (one(T) - f) * μ + f * x
        # update online variance, via Welford's algorithm
        σ² = (one(T) - f) * σ² + f * (x - μ) * (x - μ₋₁)
    end
    return n, σ²
end

"""
    edges, counts, bw = _kde_prepare(v, lo = nothing, hi = nothing;
                                     nbins = nothing, bwratio = 8)

Pre-process the input vector `v` into a histogram of `nbins` bins between `lo` and `hi`
(inclusive), returning a `StepRangeLen` of bin edges (`edges`), `Vector{Int}` of bin counts
(`counts)`, and the scalar bandwidth estimate (`bw`) estimated by Silverman's Rule.

**Optional Parameters**

- `lo`: The lowest edge of the histrogram range. Defaults to `minimum(v)`.
- `hi`: The highest edge of the histogram range.Defaults to `maximum(v)`.
- `bwratio`: The ratio of bins per bandwidth to oversample the histogram by when `nbins`
  is determined automatically. Defaults to `8`.
- `nbins`: The number of bins to use in the histogram. Defaults to
  `round(Int, bwratio * (hi - lo) / bw)`.
"""
function _kde_prepare(v, lo = nothing, hi = nothing, nbins = nothing;
                      bwratio = 8)
    T = float(eltype(v))

    # determine lower and upper limits of the histogram
    lo′ = isnothing(lo) ? minimum(v) : lo
    hi′ = isnothing(hi) ? maximum(v) : hi

    # Estimate a nominal bandwidth using Silverman's Rule.
    # Used for:
    # - Automatically choosing a value of nbins, if not provided
    # - Initializing the more accurate bandwidth optimization
    #
    # From Hansen (2009) — https://users.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf
    # for a Gaussian kernel:
    # - Table 1:
    #   - R(k) = 1 / 2√π
    #   - κ₂(k) = 1
    # - Section 2.9, letting ν = 2:
    #   - bw = σ̂ n^(-1/5) C₂(k)
    #     C₂(k) = 2 ( 8R(k)√π / 96κ₂² )^(1/5) == (4/3)^(1/5)
    n, σ² = let pred = ≥(lo′) ∘ ≤(hi′)
        _count_var(pred, v)
    end
    bw = ifelse(iszero(σ²), eps(lo′), sqrt(σ²) * (T(4 // 3) / n)^(one(T) / 5))

    if isnothing(nbins)
        nbins′ = max(1, round(Int, bwratio * (hi′ - lo′) / bw))
    else
        nbins > 0 || throw(ArgumentError("nbins must be a positive integer"))
        nbins′ = Int(nbins)
    end

    # histogram the raw data points into bins which are a factor ≈ N narrower
    # than the bandwidth of the kernel
    # N.B. use          range(..., length = round(Int, N * (hi′ - lo′) / bw))
    #      instead of   range(..., step = bw / N)
    #      in order to guarantee the hi′ endpoint is the last endpoint
    edges = range(lo′, hi′, length = nbins′ + 1)
    # normalize the histogram such that it is a probability mass function
    counts = histogram(v, edges) ./ n

    return edges, counts, bw
end

"""
    x, f = kde(v; lo = nothing, hi = nothing, nbins = nothing,
               bandwidth = nothing, bwratio = 8,
               opt_normalize = true
              )

Calculate a discrete kernel density estimate (KDE) `f(x)` of the sample distribution of `v`.

The KDE is constructed by first histogramming the input `v` into `nbins` bins with
outermost bin edges spanning `lo` to `hi`, which default to the minimum and maximum of
`v`, respectively, if not provided. The histogram is then convolved with a Gaussian
distribution with standard deviation `bandwidth`. The `bwratio` parameter is used to
calculate `nbins` when it is not given and corresponds to the ratio of the bandwidth to
the width of each histogram bin.

This simple KDE can then be post-processed in a number of ways to improve the returned
density estimate:

- `opt_normalize` — If `true`, the KDE is renormalized to account for the convolutions
  interaction with implicit zeros beyond the bounds of the low and high edges of the
  histogram.

- `opt_linearboundary` — If `true`, corrections are applied to account for non-zero
  slope at the edges of the KDE. Note that when this correction is enabled,
  `opt_normalize` is effectively ignored since overall normalization is an implicit part
  of this correction.

# Extended help

- A truncated Gaussian smoothing kernel is assumed. The Gaussian is truncated at ``4σ``.
- The linear boundary correction is explained in Refs. Lewis (2019) and Jones (1996).

## References:

- M. C. Jones and P. J. Foster. “A simple nonnegative boundary correction method for kernel
  density estimation”. In: Statistica Sinica 6 (1996), pp. 1005–1013.

- A. Lewis. "GetDist: a Python package for analysing Monte Carlo samples".
  In: _arXiv e-prints_ (Oct. 2019).
  arXiv: [1910.13970 \\[astro-ph.IM\\]](https://arxiv.org/abs/1910.13970)
"""
function kde(v;
             lo = nothing, hi = nothing, nbins = nothing,
             bandwidth = nothing, bwratio = 8,
             opt_normalize = true, opt_linearboundary = true
            )
    T = eltype(v)
    edges, counts, bw_s = _kde_prepare(v, lo, hi, nbins; bwratio = bwratio)

    bw = isnothing(bandwidth) ? bw_s : bandwidth

    # bin centers instead of the bin edges
    Δx = let Δx = step(edges)
        ifelse(iszero(Δx), bw, Δx)
    end
    centers = edges[2:end] .- Δx / 2
    # make sure the kernel axis is centered on zero
    nn = ceil(Int, 4bw / Δx)
    xx = range(-nn * Δx, nn * Δx, step = Δx)
    # construct the convolution kernel
    # N.B. Mathematically normalizing the kernel, such as with
    #        kernel = exp.(-(xx ./ bw) .^ 2 ./ 2) .* (Δx / bw / sqrt(2T(π)))
    #      breaks down when bw << Δx. Instead of trying to work around that, just take the
    #      easy route and just post-normalize a simpler calculation.
    kernel = exp.(-(xx ./ bw) .^ 2 ./ 2)
    kernel ./= sum(kernel)

    # convolve the data with the kernel to construct a density estimate
    K̂ = plan_conv(counts, kernel)
    f₀ = conv(counts, K̂, :same)

    if opt_normalize || opt_linearboundary
        # normalize the estimate, which accounts for implicit zeros outside the histogram
        # bounds
        Θ = fill!(similar(counts), one(T))
        μ₀ = conv(Θ, K̂, :same)
        f₀ ./= μ₀
    end
    if opt_linearboundary
        # apply a linear boundary correction
        # see Eqn 12 & 16 of Lewis (2019)
        #   N.B. the denominator of A₀ should have [W₂]⁻¹ instead of W₂
        kernel .*= xx
        replan_conv!(K̂, kernel)
        μ₁ = conv(Θ, K̂, :same)
        f′ = conv(counts, K̂, :same)

        kernel .*= xx
        replan_conv!(K̂, kernel)
        μ₂ = conv(Θ, K̂, :same)
        μ₀₂ = (μ₂ .*= μ₀)

        # function to force f̂ to be positive
        # see Eqn. 17 of Lewis (2019)
        pos(f₁, f₂) = iszero(f₁) ? zero(f₁) : f₁ * exp(f₂ / f₁ - one(f₁))
        f̂ = pos.(f₀, (μ₀₂ .* f₀ .- μ₁ .* f′) ./ (μ₀₂ .- μ₁.^2))
    else
        f̂ = f₀
    end

    return centers, f̂
end
