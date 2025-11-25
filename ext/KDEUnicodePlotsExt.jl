module KDEUnicodePlotsExt

import KernelDensityEstimation: UnivariateKDE
import UnicodePlots: Plot, lineplot, lineplot!

function lineplot(K::UnivariateKDE; kws...)
    lineplot(K.x, K.f; kws...)
end

function lineplot!(plot::Plot, K::UnivariateKDE; kws...)
    lineplot!(plot, K.x, K.f; kws...)
end

end
