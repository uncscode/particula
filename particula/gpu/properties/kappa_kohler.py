# mypy: disable-error-code="valid-type,index,operator"
"""Float64 device evaluation of the global equilibrium κ-Köhler barrier."""

import warp as wp

_ZERO = wp.constant(wp.float64(0.0))
_ONE = wp.constant(wp.float64(1.0))
_THREE = wp.constant(wp.float64(3.0))


@wp.func
def _log_wet(t: wp.float64):
    # Warp lacks expm1. This equivalent product retains small t exactly
    # to first order and never evaluates sinh of a large argument.
    if t < _ONE:
        half = wp.float64(0.5) * t
        return wp.log(wp.float64(2.0) * wp.sinh(half) * wp.exp(-half))
    return wp.log(_ONE - wp.exp(-t))


@wp.func
def _softplus(x: wp.float64):
    if x > wp.float64(40.0):
        return x
    return wp.log(_ONE + wp.exp(x))


@wp.func
def _peak(
    left: wp.float64, right: wp.float64, log_k: wp.float64, log_s: wp.float64
):
    lo = left
    hi = right
    for _i in range(60):
        middle = (lo + hi) * wp.float64(0.5)
        t = wp.exp(middle)
        wet = _log_wet(t)
        slope = wp.log(_THREE) + wet + _softplus(t + wet - log_k)
        if slope < wp.log(t + log_s):
            lo = middle
        else:
            hi = middle
    t = wp.exp((lo + hi) * wp.float64(0.5))
    return wp.exp(-_softplus(log_k - (t + _log_wet(t))) / _THREE) / (log_s + t)


@wp.func
def kappa_kohler_critical_dry_diameter(
    kappa: wp.float64, saturation_ratio: wp.float64, kelvin_diameter: wp.float64
):
    """Return metres, or a negative sentinel when the result is undefined."""
    if (
        not wp.isfinite(kappa)
        or not wp.isfinite(saturation_ratio)
        or not wp.isfinite(kelvin_diameter)
        or kappa < _ZERO
        or saturation_ratio <= _ONE
        or kelvin_diameter <= _ZERO
    ):
        return wp.float64(-1.0)
    log_s = wp.log(saturation_ratio)
    scale = _ONE / log_s
    if kappa > _ZERO:
        low = wp.float64(-420.0)
        high = wp.log(wp.float64(1024.0))
        log_k = wp.log(kappa)
        if kappa > wp.float64(34.97056274847714):
            disc = wp.sqrt(
                wp.max(
                    _ZERO,
                    _ONE - wp.float64(36.0) / kappa * (_ONE - _ONE / kappa),
                )
            )
            right = wp.log(
                wp.log(wp.float64(6.0) * (_ONE - _ONE / kappa) / (_ONE + disc))
            )
            left = wp.log(log_k + wp.log((_ONE + disc) / wp.float64(6.0)))
            scale = wp.max(
                _peak(low, right, log_k, log_s), _peak(left, high, log_k, log_s)
            )
        else:
            scale = _peak(low, high, log_k, log_s)
    diameter = kelvin_diameter * scale
    if not wp.isfinite(diameter) or diameter <= _ZERO:
        return wp.float64(-1.0)
    return diameter
