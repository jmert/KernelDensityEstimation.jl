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
                                     npts = nothing, bwratio = 8)

Pre-process the input vector `v` into a histogram of `npts` bins between `lo` and `hi`
(inclusive), returning a `StepRangeLen` of bin edges (`edges`), `Vector{Int}` of bin counts
(`counts)`, and the scalar bandwidth estimate (`bw`) estimated by Silverman's Rule.

**Optional Parameters**

- `lo`: The lowest edge of the histrogram range. Defaults to `minimum(v)`.
- `hi`: The highest edge of the histogram range.Defaults to `maximum(v)`.
- `bwratio`: The ratio of bins per bandwidth to oversample the histogram by when `npts`
  is determined automatically. Defaults to `8`.
- `npts`: The number of bins to use in the histogram. Defaults to
  `round(Int, bwratio * (hi - lo) / bw)`.
"""
function _kde_prepare(v, lo = nothing, hi = nothing;
                      npts = nothing, bwratio = 8)
    T = float(eltype(v))

    # determine lower and upper limits of the histogram
    lo′ = isnothing(lo) ? minimum(v) : lo
    hi′ = isnothing(hi) ? maximum(v) : hi

    # Estimate a nominal bandwidth using Silverman's Rule.
    # Used for:
    # - Automatically choosing a value of npts, if not provided
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

    if isnothing(npts)
        npts′ = max(1, round(Int, bwratio * (hi′ - lo′) / bw))
    else
        npts > 0 || throw(ArgumentError("npts must be a positive integer"))
        npts′ = Int(npts)
    end

    # histogram the raw data points into bins which are a factor ≈ N narrower
    # than the bandwidth of the kernel
    # N.B. use          range(..., length = round(Int, N * (hi′ - lo′) / bw))
    #      instead of   range(..., step = bw / N)
    #      in order to guarantee the hi′ endpoint is the last endpoint
    edges = range(lo′, hi′, length = npts′ + 1)
    # normalize the histogram such that it is a probability mass function
    counts = histogram(v, edges) ./ n

    return edges, counts, bw
end
