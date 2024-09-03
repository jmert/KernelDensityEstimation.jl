module KDEUnicodePlotsExt

import KernelDensityEstimation
const KDE = KernelDensityEstimation

using UnicodePlots: lineplot

function Base.show(io::IO, mime::MIME"text/plain", K::KDE.UnivariateKDE{T}) where {T}
    print(io, KDE.UnivariateKDE, '{', T, '}', '(')
    io′ = IOContext(io, :compact => true)
    println(io′, first(K.x), ':', step(K.x), ':', last(K.x), ", [ … ])")

    sigdigits = 4
    ylim = (0.0, round(maximum(K.f); sigdigits))
    ytickwd = textwidth(string(ylim[2]))

    ht, wd = displaysize(io)
    ht -= 3#=borders=# + 1#=tick label=# + 1#=first line print=# + 2#=repl context=#
    wd -= 4#=borders=# + ytickwd

    pl = lineplot(K...; ylim, margin = 0, height = ht, width = wd)
    show(io, mime, pl)
    return nothing
end

end
