module KDEUnicodePlotsExt

import KernelDensityEstimation: BivariateKDE, UnivariateKDE
import UnicodePlots: Plot, contourplot, contourplot!, heatmap, lineplot, lineplot!

function lineplot(K::UnivariateKDE; kws...)
    lineplot(K.x, K.f; kws...)
end

function lineplot!(plot::Plot, K::UnivariateKDE; kws...)
    lineplot!(plot, K.x, K.f; kws...)
end

function contourplot(K::BivariateKDE; kws...)
    contourplot(K.x, K.y, K.f'; kws...)
end

function contourplot!(plot::Plot, K::BivariateKDE; kws...)
    contourplot!(plot, K.x, K.y, K.f'; kws...)
end

end
