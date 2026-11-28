# particula.gpu.properties.kappa_kohler

*module*

Float64 device evaluation of the global equilibrium κ-Köhler barrier.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/kappa_kohler.py#L1-L1)

## particula.gpu.properties.kappa_kohler._ONE

*attribute* · *module attribute*

```python
_ONE = wp.constant(wp.float64(1.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/kappa_kohler.py#L7-L7)

## particula.gpu.properties.kappa_kohler._THREE

*attribute* · *module attribute*

```python
_THREE = wp.constant(wp.float64(3.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/kappa_kohler.py#L8-L8)

## particula.gpu.properties.kappa_kohler._ZERO

*attribute* · *module attribute*

```python
_ZERO = wp.constant(wp.float64(0.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/kappa_kohler.py#L6-L6)

## particula.gpu.properties.kappa_kohler._log_wet

*function*

```python
def _log_wet(t: wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/kappa_kohler.py#L11-L18)

## particula.gpu.properties.kappa_kohler._peak

*function*

```python
def _peak(left: wp.float64, right: wp.float64, log_k: wp.float64, log_s: wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/kappa_kohler.py#L28-L44)

## particula.gpu.properties.kappa_kohler._softplus

*function*

```python
def _softplus(x: wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/kappa_kohler.py#L21-L25)

## particula.gpu.properties.kappa_kohler.kappa_kohler_critical_dry_diameter

*function*

```python
def kappa_kohler_critical_dry_diameter(kappa: wp.float64, saturation_ratio: wp.float64, kelvin_diameter: wp.float64)
```

Return metres, or a negative sentinel when the result is undefined.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/kappa_kohler.py#L47-L86)
