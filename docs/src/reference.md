# Reference

```@meta
CurrentModule = KernelDensityEstimation
```

## User Interface

```@docs
kde
UnivariateKDE
Boundary
```

## Advanced User Interface

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
estimate
```
