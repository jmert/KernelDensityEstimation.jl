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
UnivariateKDE
Boundary
```

## Advanced User Interface

```@docs
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
UnivariateKDEInfo
AbstractKDEMethod
boundary
bounds
estimate
estimator_order
```
