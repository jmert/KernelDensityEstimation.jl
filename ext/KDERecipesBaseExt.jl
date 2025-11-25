module KDERecipesBaseExt

import KernelDensityEstimation: AbstractBinningKDE, UnivariateKDE
import RecipesBase: @recipe

@recipe function f(K::UnivariateKDE)
    seriestype --> :line
    xlabel --> "value"
    ylabel --> "density"
    return K.x, K.f
end

end
