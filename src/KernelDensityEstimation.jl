module KernelDensityEstimation

export kde

@static if VERSION >= v"1.11.0-DEV.469"
    @eval $(Expr(:public,
                 # KDE objects
                 :AbstractKDE, :UnivariateKDE, :AbstractKDEInfo, :UnivariateKDEInfo,
                 # KDE pipeline
                 :AbstractKDEMethod, :AbstractBinningKDE, :HistogramBinning, :LinearBinning,
                 :BasicKDE, :LinearBoundaryKDE, :MultiplicativeBiasKDE,
                 # Bandwidth Estimators
                 :ISJBandwidth, :SilvermanBandwidth,
                 # Interfaces
                 :Open, :Closed, :OpenLeft, :OpenRight, :ClosedLeft, :ClosedRight,
                 :boundary, :bounds, :estimate, :estimator_order, :init
                ))
end

_isunitless(::Type{T}) where {T} = one(T) == oneunit(T)
_unitless(::Type{T}) where {T} = typeof(one(T))
_invunit(::Type{T}) where {T} = typeof(inv(oneunit(T)))

include("conv.jl")
include("interface.jl")
include("histogram.jl")
include("bandwidth.jl")
include("kde.jl")


using PrecompileTools: @setup_workload, @compile_workload
@setup_workload let
    v32 = collect(exp2.(range(-10f0, 0f0, length = 16)))
    v64 = collect(exp2.(range(-10e0, 0e0, length = 16)))
    @compile_workload begin
        kde(v32)
        kde(v64)
    end
end

end
