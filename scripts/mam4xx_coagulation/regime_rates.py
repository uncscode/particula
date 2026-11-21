"""Size and charge encounter tables for atmospheric regimes."""

from functools import lru_cache

import numpy as np
import warp as wp
from particula import particles
from particula.dynamics import (
    get_hard_sphere_kernel_via_system_state,
    get_turbulent_shear_kernel_st1956_via_system_state,
)
from particula.dynamics.coagulation.brownian_kernel import (
    get_brownian_kernel_via_system_state,
)
from particula.dynamics.coagulation.sedimentation_kernel import (
    get_sedimentation_kernel_sp2016_via_system_state,
)
from particula.dynamics.coagulation.van_der_waals_kernel import (
    COULOMB_LENGTH_T,
    MAX_RADIUS_RATIO,
    get_van_der_waals_kernel_via_system_state,
)
from particula.dynamics.ions.mobility_attachment import mobility_attachment
from particula.gpu.dynamics.van_der_waals import (
    CaptureData,
    capture_pair_wp,
    prepare_capture,
)
from scipy.special import softmax

RHO = 1770.0
P = 101325.0


@wp.kernel
def capture(
    data: CaptureData,
    args: wp.array2d(dtype=wp.float64),
    out: wp.array(dtype=wp.float64),
):
    """Evaluate one admitted pair with Particula's device capture routine."""
    i = wp.tid()
    out[i] = capture_pair_wp(
        data,
        args[i, 0],
        args[i, 1],
        args[i, 2],
        args[i, 3],
        args[i, 4],
        args[i, 5],
        args[i, 6],
        args[i, 7],
    )


@lru_cache(maxsize=12)
def tables(per_decade, cap, temp, hamaker, dielectric):
    """Return kernel matrices in cm³/s and bipolar equilibrium probabilities."""
    # 100 nm is a bin boundary on every grid.
    edge = 100.0 * 10 ** (
        np.arange(-2 * per_decade, 3 * per_decade + 1) / per_decade
    )
    d = np.sqrt(edge[1:] * edge[:-1])
    r0 = d * 0.5e-9
    q = np.arange(-cap, cap + 1)
    r = np.repeat(r0, len(q))
    m = 4 * np.pi / 3 * RHO * r**3
    z = np.tile(q, len(d))
    plus = (
        mobility_attachment(
            r, m, z, temp, P, 100 * 1.66053906660e-27, 1.5e-4, 1
        )
        * 1e6
    )
    minus = (
        mobility_attachment(
            r, m, z, temp, P, 100 * 1.66053906660e-27, 1.5e-4, -1
        )
        * 1e6
    )
    plus = plus.reshape(len(d), -1)
    minus = minus.reshape(len(d), -1)
    # Start at the center to avoid subtracting large rare-state log weights.
    logp = np.zeros_like(plus)
    for k in range(cap, 2 * cap):
        logp[:, k + 1] = logp[:, k] + np.log(
            np.maximum(plus[:, k], 1e-300) / minus[:, k + 1]
        )
        logp[:, 2 * cap - k - 1] = logp[:, k + 1]
    prob = softmax(logp, axis=1)
    base = get_hard_sphere_kernel_via_system_state(r, m, z, temp, P) * 1e6
    i, j = np.triu_indices(len(r))
    ratio = np.maximum(r[i], r[j]) / np.minimum(r[i], r[j])
    coupling = COULOMB_LENGTH_T / temp / (r[i] + r[j])
    ok = (
        (ratio <= MAX_RADIUS_RATIO)
        & (coupling * z[i] * z[j] >= -0.5)
        & (coupling * np.maximum(z[i] ** 2, z[j] ** 2) <= 60)
    )
    i, j = i[ok], j[ok]
    diff = particles.get_diffusion_coefficient_via_system_state(r, temp, P)
    args = np.column_stack(
        [
            r[i],
            r[j],
            m[i],
            m[j],
            z[i],
            z[j],
            diff[i] + diff[j],
            np.full(len(i), temp),
        ]
    )
    a = wp.array(args, dtype=wp.float64, device="cuda:0")
    out = wp.empty(len(i), dtype=wp.float64, device="cuda:0")
    wp.launch(
        capture,
        len(i),
        [prepare_capture(hamaker, dielectric, "cuda:0"), a, out],
        device="cuda:0",
    )
    full = base.copy()
    vals = out.numpy() * 1e6
    if not np.all(np.isfinite(vals) & (vals >= 0)):
        raise ValueError("Invalid admitted capture rate")
    full[i, j] = full[j, i] = vals
    supported = np.zeros_like(full, dtype=bool)
    supported[i, j] = supported[j, i] = True
    # Independent neutral two-radius CPU capture, including diameter contrast.
    take = np.array([len(d) // 3, len(d) // 3 + 1]) * len(q) + cap
    expected = get_van_der_waals_kernel_via_system_state(
        r[take], m[take], z[take], temp, P, hamaker, dielectric
    )
    np.testing.assert_allclose(
        full[np.ix_(take, take)], expected * 1e6, rtol=0.012
    )
    neutral = np.arange(cap, len(r), len(q))
    brown = get_brownian_kernel_via_system_state(r0, m[neutral], temp, P) * 1e6
    settling = (
        get_sedimentation_kernel_sp2016_via_system_state(
            r0,
            np.full_like(r0, RHO),
            temp,
            P,
            calculate_collision_efficiency=False,
        )
        * 1e6
    )
    shear = (
        get_turbulent_shear_kernel_st1956_via_system_state(
            r0, 0.01, temp, P / (287.05 * temp)
        )
        * 1e6
    )
    return (
        edge,
        d,
        q,
        plus,
        minus,
        prob,
        base,
        full,
        supported,
        brown,
        settling,
        shear,
    )
