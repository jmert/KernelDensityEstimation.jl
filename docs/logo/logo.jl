using Distributions
using ForwardDiff
using KernelDensityEstimation: KernelDensityEstimation as KDE, kde
using LinearAlgebra: diagind
using GLMakie
using Random

import CairoMakie
import .Makie.Colors: JULIA_LOGO_COLORS as JLC, Oklab, weighted_color_mean

function loss(x, y; α = 1.0, β = 1.0)
    N = length(x)
    # pull each point towards zero to resist divergence
    attract = α * mean(y.^2)

    # but also push points away from each other
    rr = min.(1e3, inv.(hypot.(x .- x', y .- y')))
    rr[diagind(rr)] .= 0  # no self-interaction
    repel = β * sum(abs2, rr) / max(1, (N^2 - N))

    return attract + repel
end

# run gradient descent to optimize the y positions of each point to avoid clashing with
# neighbors (as much)
function optimize(x; N = 100, α = 1.0, β = 1.0, η = 0.01)
    y = 1e-3 .* (-1) .^ (1:length(x))
    f(a) = loss(x, a; α, β)
    for ii in 1:N
        ∇f = ForwardDiff.gradient(f, y)
        @. y -= η * ∇f
    end
    return y
end

function plot_data_kde!(ax, D, x, K; rng = Random.GLOBAL_RNG, color = :black, scale = 1.0)
    yl = 1.25maximum(K.f)
    el, eh = extrema(D)
    lcolor = weighted_color_mean(0.7, Oklab(colorant"white"), Oklab(color))

    closed_kws = (; linewidth = max(1.5, 4scale), linestyle = (:dash, :dense), color)

    fill_between!(ax, K.x, zeros(size(K.f)), K.f; color = lcolor)
    isfinite(el) && lines!(ax, [el, el], [0.0, K.f[begin]]; closed_kws...)
    isfinite(eh) && lines!(ax, [eh, eh], [0.0, K.f[end]];   closed_kws...)
    lines!(ax, K; linewidth = max(1.5, 6scale), color)

    y = yl .+ 0.05 .* optimize(x; N = 100, η = 0.02, α = 8.0, β = 10.0)
    scatter!(ax, x, y;
             marker = :circle, markersize = max(4, 12scale), color = color)
    return nothing
end

function simplify(K; ϵ = pdf(Normal(0,1), 3.5))
    F = K.f ./ maximum(K.f)
    s = findfirst(>(ϵ), F)
    e = findlast(>(ϵ), F)
    return typeof(K)(K.x[s:e], K.f[s:e])
end

function logo(; theme = :light, scale = 1.0)
    #rng = Xoshiro(1028)

    # chi-squared component
    D_chi2 = Chisq(2)
    #x_chi2 = rand(rng, D_chi2, 10)
    #@show x_chi2
    x_chi2 = [3.21453403980826, 0.2827316556727179, 2.1716645776188503, 6.296474899950287,
              0.2232590492604249, 2.4908091118464486, 1.2068926488526808,
              0.8262280742143103, 0.3485733216618406, 3.885214912162651]
    K_chi2 = kde(x_chi2; bwratio = 33, bounds = D_chi2)
    K_chi2 = simplify(K_chi2)

    # to avoid having dots hidden by the y axis, nudge them away if they're too close
    @. x_chi2 += 0.3 * exp(-(2x_chi2)^2)

    # gaussian component
    D_norm = Normal(10, 1.15)
    #x_norm = rand(rng, D_norm, 10)
    #x_norm .-= mean(x_norm) - 10
    #@show x_norm
    x_norm = [8.485481193139636, 9.871463551884933, 12.087307016715695, 8.620990727348747,
              10.223445663964675, 8.833700600822306, 9.225084835589621, 11.277423606842332,
              9.950064964220722, 11.425037839471324]
    K_norm = kde(x_norm, bwratio = 33, bounds = D_norm)
    K_norm = simplify(K_norm)

    # beta distribution
    D_beta = 10.0 .* Beta(5.0, 1.60) .+ 10.0
    #x_beta = rand(rng, D_beta, 10)
    #@show x_beta
    x_beta = [16.96895857723336, 17.32592478264389, 18.54890458479186, 17.96058921812689,
              16.043701840627964, 16.64261538485738, 18.603421757354912, 14.484678550565858,
              14.324338853118187, 14.948463461819706]
    K_beta = kde(x_beta, bwratio = 33, bounds = D_beta)
    K_beta = simplify(K_beta)

    # adjust the heights of the distributions to have the scatter points above evoke
    # the three Julia dots
    K_chi2.f .*= 0.7
    K_norm.f .*= 1.75
    K_beta.f .*= 1.3

    # baseline size is 256×256, with a scale factor to change behavior
    base = 256
    figsize = (base, base) .* scale

    # adjust features based on scale size
    if scale >= 128 / base
        figsize = figsize .+ (0, 25scale)  # more room for package name text
    elseif scale < 64 / base
        # simplify the dots
        if scale <= 16 / base
            # when too small, don't show the dots at all
            x_chi2 = x_norm = x_beta = Float64[]
        elseif scale <= 32 / base
            # for small sizes, show just a single dot
            x_chi2 = [sum(K_chi2.x .* K_chi2.f) / sum(K_chi2.f)]
            x_norm = [sum(K_norm.x .* K_norm.f) / sum(K_norm.f)]
            x_beta = [sum(K_beta.x .* K_beta.f) / sum(K_beta.f)]
        end
    end

    fig = Figure(size = figsize,
                 figure_padding = max(1, 2scale),
                 backgroundcolor = :transparent,
                )
    ax = Axis(fig[1, 1],
              aspect = 1,
              backgroundcolor = :transparent,
             )
    hidespines!(ax)
    hidedecorations!(ax)

    pl_chi2 = plot_data_kde!(ax, D_chi2, x_chi2, K_chi2; scale, color = JLC.red)
    pl_beta = plot_data_kde!(ax, D_beta, x_beta, K_beta; scale, color = JLC.purple)
    pl_norm = plot_data_kde!(ax, D_norm, x_norm, K_norm; scale, color = JLC.green)

    autolimits!(ax)
    xl, yl = 0.9 .* ax.finallimits[].widths

    cc = theme == :dark ? :white : :black
    lines!([0, 0, xl], [yl, 0, 0], linewidth = max(1.5, 6scale), color = cc)
    scatter!(xl, 0, markersize = max(6, 27scale), marker = :rtriangle, color = cc)
    scatter!(0, yl, markersize = max(6, 27scale), marker = :utriangle, color = cc)

    if scale >= 128 / base
        ax2 = Label(fig[2, 1],
                    rich("KernelDensityEstimation", rich(".jl", color = JLC.blue), color = cc),
                    fontsize = 19.5scale, font = :bold, halign = :center,
                    tellwidth = false)

        rowgap!(fig.layout, 1, 0)
    end
    return fig
end

@static if isdefined(Base, Symbol("@main"))
    import Base: @main
else
    macro main()
        return esc(:main)
    end
end

function (@main)()
    CairoMakie.activate!(type = "svg")
    save("logo32-light.svg",  logo(theme = :light, scale = 32 / 256), backend = CairoMakie)
    save("logo256-light.svg", logo(theme = :light, scale = 1.0), backend = CairoMakie)
    save("logo32-dark.svg",  logo(theme = :dark, scale = 32 / 256), backend = CairoMakie)
    save("logo256-dark.svg", logo(theme = :dark, scale = 1.0), backend = CairoMakie)
end
