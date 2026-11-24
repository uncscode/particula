"""Full kappa-Kohler activation at fixed S, with optional inlet water."""

# %%
import numpy as np
import warp as wp
from particula.particles import get_kelvin_radius
from scipy.optimize import brentq, minimize_scalar


@wp.func
def dry_at_growth(x: wp.float64, k: wp.float64, ls: wp.float64):
    """Evaluate critical dry diameter in units of the Kelvin coefficient."""
    one = wp.float64(1)
    water = wp.exp(wp.float64(3) * x) - one
    return wp.exp(-x) / (ls + wp.log(one + k / water))


@wp.kernel
def threshold(
    conditions: wp.array2d(dtype=wp.float64),
    kappas: wp.array(dtype=wp.float64),
    hydrated: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Find critical diameters independently on a scan by kappa grid."""
    i, j = wp.tid()
    one = wp.float64(1)
    k = kappas[j]
    ls = wp.log(one + conditions[i, 0] / wp.float64(100))
    a = wp.float64(0)
    b = wp.log(wp.float64(256))
    for _step in range(52):
        third = (b - a) / wp.float64(3)
        x1 = a + third
        x2 = b - third
        if dry_at_growth(x1, k, ls) < dry_at_growth(x2, k, ls):
            a = x1
        else:
            b = x2
    dry = dry_at_growth((a + b) / wp.float64(2), k, ls)
    ratio = one
    if hydrated == 1:
        a = one
        rh = conditions[i, 1] / wp.float64(100)
        b = wp.pow(one + k * rh / (one - rh), one / wp.float64(3))
        for _step in range(34):
            mid = (a + b) / wp.float64(2)
            water = mid * mid * mid - one
            s = water / (water + k) * wp.exp(one / (dry * mid))
            if s < rh:
                a = mid
            else:
                b = mid
        ratio = (a + b) / wp.float64(2)
    output[i, j] = conditions[i, 2] * dry * ratio


def conditions(data):
    """Return calibrated supersaturation, inlet RH and Kelvin length in nm."""
    a = 2e9 * get_kelvin_radius(0.072, 1000.0, 0.01801528, data["T"].reshape(-1))
    return np.column_stack([data["s"].reshape(-1), data["rh"].reshape(-1), a])


def tables(conditions_array, k, hydrated):
    """Evaluate equilibrium thresholds on the selected CUDA device."""
    c = wp.array(conditions_array, dtype=wp.float64, device="cuda:0")
    grid = wp.array(k, dtype=wp.float64, device="cuda:0")
    out = wp.empty((len(c), len(k)), dtype=wp.float64, device="cuda:0")
    wp.launch(
        threshold,
        dim=out.shape,
        inputs=[c, grid, int(hydrated), out],
        device="cuda:0",
    )
    result = out.numpy()
    if not np.all(np.isfinite(result) & (result > 0)):
        raise ValueError("Invalid activation diameter")
    return result


def oracle(case, k, hydrated):
    """Independent forward maximum in wet diameter inside dry-diameter root."""
    ss, rh, kelvin = case

    def saturation(g, d):
        water = g**3 - 1
        return water / (water + k) * np.exp(kelvin / (d * g))

    def peak(d):
        optimum = minimize_scalar(
            lambda x: -saturation(np.exp(x), d),
            bounds=(0, np.log(256)),
            method="bounded",
        )
        return -optimum.fun - 1 - ss / 100

    d = brentq(peak, 4, 1500)
    if hydrated:
        g = brentq(
            lambda x: saturation(x, d) - rh / 100,
            1.0,
            (1 + k * (rh / 100) / (1 - rh / 100)) ** (1 / 3),
        )
        return d * g
    return d
