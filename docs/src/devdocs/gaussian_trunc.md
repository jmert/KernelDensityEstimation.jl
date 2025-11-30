```@meta
CurrentModule = KernelDensityEstimation
```

# Gaussian Truncation

```@contents
Pages = ["gaussian_trunc.md"]
Depth = 2:2
```

!!! details "Setup code"
    ```@example gaussian_trunc
    using CairoMakie

    using Distributions
    using LinearAlgebra

    function gauss1(x; σ)
        𝒟 = Normal(zero(σ), σ)
        return pdf.(𝒟, x)
    end

    function gauss2(x, y; Σ)
        𝒟 = MvNormal(fill!(similar(Σ, 2), 0), Σ)
        xy = Iterators.product(x, y)
        return map(Iterators.product(x, y)) do xy
            pdf(𝒟, [xy...])
        end
    end
    nothing  # hide
    ```

## Univariate (1D)

### Definition

Given the variance ``σ^2``, the 1D Gaussian distribution is defined for ``x ∈ ℝ`` to be
```math
\begin{align}
    G(x; σ) &≡ \frac{1}{\sqrt{2πσ^2}} e^{-x^2 / 2σ^2}
    \label{eqn:gaussian1}
\end{align}
```
where we assume the distribution has mean equal to zero (``μ = 0``).

Since the Gaussian function is defined over the entire real line, we must truncate it to a finite size before
convolution.
We choose to truncate at the ``±4σ`` bounds, which simply means that the domain is restricted to
``x ∈ [-4σ, +4σ]``.

We choose ``σ = h`` (the bandwidth) and guarantee an odd, whole number of points in the discrete kernel by letting
```math
\begin{align}
    \{x_i\} &= -n_h \Delta x + i \Delta x  &\text{for } i &∈ \{0, 1, …, 2n_h\}
\end{align}
```
where
```math
    n_h = \left\lceil \frac{4h}{\Delta x} \right\rceil
```
and ``\Delta x`` is the bin size of the histogrammed density.


## Multivariate (2D & ND)

### Definition

Given a covariance matrix ``𝚺``, the multivariate Gaussian distribution is defined for ``𝒙 ∈ ℝ^n`` to be
```math
\begin{align}
    G(𝒙; 𝚺) &≡ \frac{1}{\sqrt{(2π)^n |𝚺|}} e^{-𝒙^⊤ 𝚺^{-1} 𝒙 / 2}
    \label{eqn:gaussian}
\end{align}
```
where we assume the distribution has mean equal to zero (``𝝁 = 𝟎``) and ``|𝚺| = \det 𝚺``.
Like the univariate case, the multivariate Gaussian is unbounded in all directions and must be truncated to a
finite range to be of practical use.
We extend the univariate truncation definition that ``x ∈ [-4σ, +4σ]`` in the following way.

We start with the definition from 1D and note that as long as ``𝚺`` is diagonal, the marginalized density over
dimensions is equivalent (up to normalization) to the corresponding 1D Gaussian, as shown in Figure 1.

```@example gaussian_trunc
# Define (co)variances in x, y dimensions
σ_x, σ_y = 2.0, 1.0
Σ = Diagonal([σ_x^2, σ_y^2])

# Construct the 2D Gaussian distribution
xx = range(-10, 10, step = 0.1)
yy = range(-6, 6, step = 0.1)
kern = gauss2(xx, yy; Σ)
# and the corresponding 1D marginal distributions
kern_x = gauss1(xx; σ = σ_x)
kern_y = gauss1(yy; σ = σ_y)
nothing  # hide
```

Given this equivalence, we choose to *define* the ``4σ`` boundary in the multidimensional sense to be the contour level
which results in the same marginalized truncation for an uncorrelated covariance matrix.

```@example gaussian_trunc
# transform a 4σ circle into ellipse using the Cholesky decomposition
ρ = 4.0  # 4σ level
L = cholesky(Σ).L
contour_4σ = mapreduce(hcat, range(0, 2π, 250)) do θ
    s, c = ρ .* sincos(θ)
    return L * [c,s]
end
nothing  # hide
```

```@figure; htmllabel = "Figure 1", latexlabel = "fig:gauss_trunc:2dmarginal"
![](gaussian_trunc_2dmarginal.svg)

A 2D Gaussian with diagonal covariance matrix (bottom-left) and the two marginal densities (solid blue) that result from
summing [and renormalizing] along each axis (top and right), along with the corresponding 1D Gaussian (dashed yellow)
to highlight the equivalence.
The 1D ``4σ`` points are indicated (dashed red) in the marginal densities and extended across the 2D density where
they contain the 2D ellipse (solid red) that we use to define the ``4σ`` contour in higher dimensions.
```

!!! details "Plotting Code"
    ```@example gaussian_trunc
    # 1D Gaussians from marginalizing 2D Gaussian
    m2_x = sum(kern, dims = 2)[:] .* step(yy)
    m2_y = sum(kern, dims = 1)[:] .* step(xx)

    ## Plotting
    fig = Figure(size = (600, 600))

    # 2D density
    ax = Axis(fig[2, 1])
    heatmap!(ax, xx, yy, kern, colormap = Reverse(:grays), rasterize = true)

    # 1D marginalized density along x-axis
    axx = Axis(fig[1, 1], xautolimitmargin = (0.0, 0.0), yautolimitmargin = (0.05, 0.05))
    lines!(axx, xx, m2_x, color = Cycled(1), label = "2D Marginal", linewidth = 2)
    lines!(axx, xx, kern_x, color = Cycled(2), label = "1D", linestyle = :dash)
    hidexdecorations!(axx, grid = false, ticks = false)

    # 1D marginalized density along y-axis, rotated to project through 2D density
    axy = Axis(fig[2, 2], xautolimitmargin = (0.05, 0.05), yautolimitmargin = (0.0, 0.0))
    lines!(axy, m2_y, yy, color = Cycled(1), linewidth = 2)
    lines!(axy, kern_y, yy, color = Cycled(2), linestyle = :dash)
    hideydecorations!(axy, grid = false, ticks = false)

    # add 4σ contour and corresponding bounding box to the 2D density plot
    lines!(ax, contour_4σ, color = (:firebrick3, 0.5), label = "4σ contour")
    kws = (; color = (:firebrick3, 0.5), linestyle = :dash, depth_shift = -1)
    vlines!(ax,  4σ_x .* [-1, 1]; kws...)
    vlines!(axx, 4σ_x .* [-1, 1]; kws...)
    hlines!(ax,  4σ_y .* [-1, 1]; kws...)
    hlines!(axy, 4σ_y .* [-1, 1]; kws...)

    # construct a shared legend across all three axes
    l1 = Makie.get_labeled_plots(axx, merge = true, unique = true)
    l2 = Makie.get_labeled_plots(ax,  merge = true, unique = true)
    lplt = vcat(l1[1], l2[1])
    llbl = vcat(l1[2], l2[2])
    leg = Legend(fig[1, 2], lplt, llbl; tellwidth = false, framevisible = false)

    # link axes
    linkxaxes!(ax, axx)
    linkyaxes!(ax, axy)
    map!(identity, axx.xticks, ax.xticks)  # force equivalence of ticks across linked axes
    map!(identity, axy.xticks, ax.yticks)
    ax.xticks = -8:4:8
    # adjust gaps to make more compact
    colgap!(fig.layout, Fixed(4))
    rowgap!(fig.layout, Fixed(4))
    colsize!(fig.layout, 1, Relative(0.8))  # 80% of space to 2D plot
    rowsize!(fig.layout, 2, Aspect(1, length(yy) / length(xx)))  # square data units
    rowsize!(fig.layout, 1, Aspect(2, 1))  # marginal "heights" are equal
    resize_to_layout!(fig)

    save("gaussian_trunc_2dmarginal.svg", fig, backend = CairoMakie)  # hide
    nothing  # hide
    ```

Therefore, our goal is to obtain the [hyper]rectangle which contains the ``4σ`` contour, as shown in Figure 2, for
any arbitrary covariance.

A non-diagonal covariance may be [diagonalized](https://en.wikipedia.org/wiki/Diagonalizable_matrix#Diagonalization)
via the eigenvalue decomposition, so we can do the reverse to construct a non-diagonal covaraince matrix where we know
the ``4σ`` ellipse _a priori_.

Let ``𝑹`` be a rotation matrix, and let ``𝚺'`` be the non-diagonal covariance constructed by applying the rotation to
our diagonal covariance.
```math
    𝚺' = 𝑹 𝚺 𝑹^⊤
```

```@example gaussian_trunc
θ = 22.5  # arbitrary angle, for demonstration
R = [
    cosd(θ) -sind(θ);
    sind(θ)  cosd(θ)
]
Σ′ = R * Σ * R'
L′ = cholesky(Σ′).L
kern′ = gauss2(xx, yy; Σ = Σ′)
contour_4σ′ = R * contour_4σ
nothing  # hide
```

Our problem is now:

> **Given the arbitrary covariance matrix ``𝚺'``, what is the bounding box ``𝖡 ⊂ ℝ^n = \bigotimes_i [-v_i, +v_i]`` which
> contains the ``4σ`` contour?**

One might be tempted to expect that the maximum coordinate value over all rotated, scaled versions of the Cartesian
unit vectors ``\{𝐞_j\}`` — i.e. the principle axes of the ellipse — would be the answer.
```math
    𝒃_i ≡ \max_j \left( ρσ_j \left| 𝑹  𝐞_j \right| \right)_i
```
Figure 2 visually demonstrates a counterexample where that this is not the case, though.

```@figure; htmllabel = "Figure 2", latexlabel = "fig:gauss_trunc:eigenvec"
![](gaussian_trunc_eigenvec.svg)

Gaussian distribution with non-diagonal covariance along with a direct rotation of the ``4σ`` contour (solid red).
Reducing over the principle axes of the ellipse (solid green and cyan) defines a bounding box (dashed blue) which is
smaller than the box that contains the contour (dashed red).
```

!!! details "Plotting Code"
    ```@example gaussian_trunc
    fig = Figure(size = (500, 500))

    # 2D density and directly-rotated 4σ contour
    ax = Axis(fig[1, 1])
    heatmap!(ax, xx, yy, kern′, colormap = Reverse(:grays), rasterize = true)
    lines!(ax, contour_4σ′, color = :firebrick3, label = "4σ contour")

    # directly-rotated 4σ contour and corresponding empirical bounding box
    Bx = [extrema(contour_4σ′[1, :])...]
    By = [extrema(contour_4σ′[2, :])...]
    vlines!(ax, Bx, color = (:firebrick3, 0.8), linestyle = :dash)
    hlines!(ax, By, color = (:firebrick3, 0.8), linestyle = :dash)

    # principle axes of the ellipse...
    zz = [0.0, 0.0]
    v1 = ρ * σ_x .* R * [1, 0]
    v2 = ρ * σ_y .* R * [0, 1]
    arrows2d!(ax, Point2(zz), Point2(v1), argmode = :endpoint, color = :green3, label = "𝐞₁")
    arrows2d!(ax, Point2(zz), Point2(v2), argmode = :endpoint, color = :cyan3, label = "𝐞₂")
    # and corresponding bounding box
    vb = maximum([v1 v2], dims = 2)[:]
    poly!(ax, Rect(-abs.(vb), 2 .* vb), color = :transparent, strokecolor = :blue3,
          strokewidth = 1.5, linestyle = :dash)

    rowsize!(fig.layout, 1, Aspect(1, length(yy) / length(xx)))
    resize_to_layout!(fig)

    save("gaussian_trunc_eigenvec.svg", fig, backend = CairoMakie)  # hide
    nothing  # hide
    ```

The correct procedure is to directly solve the desired maximization problem.
The condition for finding the maximum value over the contour line in dimension ``i`` can be expressed as maximizing
the inner product
```math
\begin{align} \max_{𝒗} 𝒗^\top 𝐞_i \end{align}
```
subject to the constraint
```math
\begin{align} 𝒗^\top 𝚺^{-1} 𝒗 = ρ^2 \end{align}
```
where ``ρ`` is the ``σ``-level contour to be found (i.e. ``ρ = 4``), ``𝒗`` is an arbitrary vector, and ``𝐞_i`` is the
Cartesian basis vector pointing along the ``i``-th dimension's axis.
Solving with the method of Lagrange multipliers:
```math
\begin{align}
    0 &= \frac{d}{d𝒗} \left[ 𝒗^\top 𝐞_i - λ \left( 𝒗^\top 𝚺^{-1} 𝒗 - ρ^2 \right) \right]
    & {}&{}
        \nonumber
    \\
    0 &= 𝐞_i - 2λ 𝚺^{-1} 𝒗
    & {}&{}
        \nonumber
    \\
    𝒗 &= \frac{1}{2λ} 𝚺 𝐞_i
    &
    ρ^2 &= \left( \frac{1}{2λ} 𝚺 𝐞_i \right)^⊤ 𝚺^{-1} \left( \frac{1}{2λ} 𝚺 𝐞_i \right)
        \nonumber
    \\
    {}&{}
    &
    ρ^2 &= \frac{1}{4λ^2} 𝐞_i^⊤ 𝚺 𝐞_i
        \nonumber
    \\
    𝐞_i^⊤ 𝒗 &= \frac{1}{2λ} 𝐞_i^⊤ 𝚺 𝐞_i
    &
    \frac{1}{2λ} &= \frac{ρ}{\sqrt{𝚺_{ii}}}
        \nonumber
    \\
    𝐞_i^⊤ 𝒗 &= ρ \sqrt{𝚺_{ii}}
        \label{eqn:mvgauss_sigmas_cov}
\end{align}
```
Therefore, the bounding box that contains an arbitrary multivariate Gaussian at the ``ρ``-sigma level is simply a box
whose edges lie at ``±ρ\sqrt{𝚺_{ii}}`` in the ``i``-th dimension.

Because the Gaussian is defined in terms of its inverse covariance matrix, the implementation does not use the
covariance matrix ``𝚺`` directly, instead making use of its Cholesky decomposition:
```math
\begin{align} 𝑳𝑳^⊤ ≡ 𝚺 \end{align}
```
where ``𝑳`` is a lower-triangular matrix.
(See [`bandwidth`](@ref).)
To avoid reconstructing the covariance matrix from its Cholesky factors, we return to
Eqn. ``\ref{eqn:mvgauss_sigmas_cov}`` and replace the covariance with its decomposition and simplify:
```math
\begin{align}
    𝐞_i^⊤ 𝒗 &= ρ \sqrt{𝚺_{ii}} \nonumber \\
    𝐞_i^⊤ 𝒗 &= ρ \sqrt{𝐞_i^⊤ \left(𝑳𝑳^⊤\right) 𝐞_i} \nonumber \\
    𝐞_i^⊤ 𝒗 &= ρ \sqrt{(𝑳^⊤ 𝐞_i)^⊤(𝑳^⊤ 𝐞_i)} \nonumber \\
    𝐞_i^⊤ 𝒗 &= ρ \sqrt{(𝒍_i)^⊤ 𝒍_i}
        \qquad\text{where }\quad (𝒍_i)_n = 𝑳_{ni} \text{ (the $i$-th row of $𝑳$)} \nonumber \\
    𝐞_i^⊤ 𝒗 &= ρ \|𝒍_i\|
\end{align}
```
We find that the square root of the ``i``-th diagonal of the covariance can be equivalently obtained from the norm
of the ``i``-th row of its Cholesky decomposition.

```@example gaussian_trunc
bbox_4σ = dropdims(mapslices(l -> ρ * sqrt(l'l), L′, dims = 2), dims = 2)
nothing  # hide
```

```@figure; htmllabel = "Figure 3", latexlabel = "fig:gauss_trunc:cholnorm"
![](gaussian_trunc_cholnorm.svg)

The same Gaussian distribution and ``4σ`` contour (solid red) as shown in Figure 2 and the target bounding box that
contains it (dashed red).
The directly calculated bounding box (solid blue) derived from the Cholesky decomposition of the covariance matrix
matches the empirical bounding box.
```

!!! details "Plotting Code"
    ```@example gaussian_trunc
    fig = Figure(size = (500, 500))

    # 2D density and directly-rotated 4σ contour
    ax = Axis(fig[1, 1])
    heatmap!(ax, xx, yy, kern′, colormap = Reverse(:grays), rasterize = true)
    lines!(ax, contour_4σ′, color = :firebrick3, label = "4σ contour")

    # directly-rotated 4σ contour and corresponding empirical bounding box
    Bx = [extrema(contour_4σ′[1, :])...]
    By = [extrema(contour_4σ′[2, :])...]
    vlines!(ax, Bx, color = (:firebrick3, 0.8), linestyle = :dash)
    hlines!(ax, By, color = (:firebrick3, 0.8), linestyle = :dash)

    # bounding box computed from the [Cholesky decomposition of the] covariance matrix
    poly!(ax, Rect(-abs.(bbox_4σ), 2 .* bbox_4σ), color = :transparent, strokecolor = :blue3,
          strokewidth = 1.5)

    rowsize!(fig.layout, 1, Aspect(1, length(yy) / length(xx)))
    resize_to_layout!(fig)

    save("gaussian_trunc_cholnorm.svg", fig, backend = CairoMakie)  # hide
    nothing  # hide
    ```
