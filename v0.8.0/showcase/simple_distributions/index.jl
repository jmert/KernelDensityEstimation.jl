using CairoMakie
if Base.isinteractive()
    using GLMakie
end
import Makie: LaTeXString

import KernelDensityEstimation as KDE
using .KDE
using Distributions
using Random

# Distributions to show
dists = [
    Normal(12.0, 2.0)   Exponential(0.5);
    Beta(0.5, 0.5)      MixtureModel([Normal(0, 1), Normal(25, 4)], [0.2, 1-0.2]);
]
# Corresponding number of samples to draw for each distribution.
nsamp = [
      500    500;
    5_000  5_000
]
# Labels to use for each distribution
lbls = let Normal = "\\mathrm{N}",
           Beta = "\\mathrm{Beta}",
           Exp = "\\mathrm{Exp}"
    [
     "$Normal(12,\\, 2)"   "$Exp(0.5)";
     "$Beta(0.5,\\, 0.5)"  "0.2\\;$Normal(0,\\, 1) + 0.8\\;$Normal(25,\\, 4)"
    ]
end

fig = Figure(size = (800, 800))

for II in CartesianIndices(dists)
    dist = dists[II]
    lbl = lbls[II]
    nn = nsamp[II]

    # Draw samples from the distribution, and then generate the KDE with all default
    # options except to correctly set the boundary conditions from the distribution
    rng = Random.Xoshiro(100)
    x = rand(rng, dist, nn)
    K = kde(x, bounds = dist)

    jj, ii = Tuple(II)
    # Use a panel to describe each distribution
    txt = Label(fig[2jj - 1, ii], LaTeXString("\$x_{1:$nn} \\sim $lbl\$"),
                tellwidth = false, fontsize = 16)

    ax = Axis(fig[2jj + 0, ii])
    # Then plot the bounds of the distribution...
    let (lo, hi) = extrema(dist), kws = (; color = (:black, 0.5), linestyle = :dot)
        hlines!(ax, 0.0; kws...)
        isfinite(lo) && vlines!(ax, lo; kws...)
        isfinite(hi) && vlines!(ax, hi; kws...)
    end
    # ... the source distribution ...
    lines!(ax, K.x, pdf.(dist, K.x), color = :blue3, linestyle = :dash,
           label = "distribution")
    # ... and the density estimate
    lines!(ax, K, color = :firebrick3, label = "density estimate")
end
# Adjust spacing between text and plot panel pairs
rowgap!.(Ref(fig.layout), 1:2:2size(dists, 1), Ref(Fixed(8)))

Legend(fig[end+1, :], content(fig[2, 1]), tellheight = true, orientation = :horizontal)

save("index.svg", fig, backend = CairoMakie)
