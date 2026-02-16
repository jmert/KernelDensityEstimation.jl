module KDERecipesBaseExt

import KernelDensityEstimation: UnivariateKDE, BivariateKDE
import RecipesBase: @recipe

@recipe function f(K::UnivariateKDE)
    seriestype --> :line
    xlabel --> "value"
    ylabel --> "density"
    return K.x, K.f
end

@recipe function f(K::BivariateKDE)
    seriestype --> :contour
    return K.x, K.y, K.f
end

end
