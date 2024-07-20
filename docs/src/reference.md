# Reference

```@meta
CurrentModule = KernelDensityEstimation
```

## User Interface

```@docs
kde
UnivariateKDE
Cover
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
estimate
```
