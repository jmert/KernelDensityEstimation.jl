# API

```@meta
CurrentModule = KernelDensityEstimation
```

```@contents
Pages = ["api.md"]
Depth = 2:2
```

## User Interface

```@docs
kde
MultivariateKDE
UnivariateKDE
BivariateKDE
Boundary
```

## Advanced User Interface

```@docs
BoundsSpec
BoundsLims
BoundsArgs
bounds(::Tuple{Vararg{AbstractVector,N}}, ::Tuple{Vararg{BoundsSpec,N}}) where {N}
bounds(::Tuple{AbstractVector}, ::Union{BoundsSpec,BoundsLims})
bounds(::AbstractVector, ::Union{BoundsSpec,BoundsLims})
init
```

### Binning Methods
```@docs
AbstractBinningKDE
HistogramBinning
LinearBinning
```

### Density Estimation Methods
```@docs
BasicKDE
LinearBoundaryKDE
MultiplicativeBiasKDE
```

### Bandwidth Estimators
```@docs
AbstractBandwidthEstimator
SilvermanBandwidth
ISJBandwidth
bandwidth
```

---

## Interfaces

### Density Estimation Methods
```@docs
AbstractKDE
AbstractKDEInfo
MultivariateKDEInfo
UnivariateKDEInfo
BivariateKDEInfo
AbstractKDEMethod
bounds
estimate
estimator_order
```
