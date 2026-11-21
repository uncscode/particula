# %%
"""Batch several chamber histories per vector on the local CUDA device."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp
from particula.gas.properties.mean_free_path import get_molecule_mean_free_path
from particula.gpu.dynamics.condensation_funcs import (
    first_order_mass_transport_k_wp,
)
from particula.gpu.properties.particle_properties import (
    vapor_transition_correction_wp,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from chamber import AP as AP  # noqa: E402
from chamber import OM, PRESSURE, RUNS, R  # noqa: E402
from chamber import TREF as TREF  # noqa: E402
from study import FORCING, series  # noqa: E402

ORDER = (1, 4, 5, 6, 2)
CONDITIONS = dict(RUNS) | {2: (309.0, 67.6, 10400.0, 24.81)}
NOX = {1: 40.0, 4: 26.0, 5: 35.0, 6: 18.0, 2: 43.0}
TIMES = np.arange(0.0, 18001.0, 360.0)
RESULT = HERE / "results"
RESULT.mkdir(exist_ok=True)
# Physical columns are f, C*_S, accommodation, lag min, particle loss h^-1,
# vapor loss h^-1, primary low-volatility share, volatility ratio, gas aging
# h^-1, four injection factors plus run 2, and (optional) count retention share.
F64 = wp.float64
MASS = wp.constant(wp.float64(0.168236))
RT = wp.constant(wp.float64(8.314462618))


@wp.func
def exchange(gas: F64, particle: F64, capacity: F64, k: F64, dt: F64):
    """Exchange mass conservatively between vapor and one condensed class."""
    # Integrate linear reversible transfer with geometry/composition denominator
    # fixed over this substep. Both reservoirs remain nonnegative.
    total = gas + particle
    equilibrium = total / (F64(1.0) + capacity)
    moved = (equilibrium - particle) * (
        F64(1.0) - wp.exp(-dt * k * (F64(1.0) + capacity))
    )
    return moved


@wp.kernel
def batch_kernel(
    params: wp.array2d(dtype=F64),
    source: wp.array2d(dtype=F64),
    props: wp.array2d(dtype=F64),
    count_curve: wp.array2d(dtype=F64),
    dt: F64,
    sub: int,
    count_mode: int,
    out: wp.array3d(dtype=F64),
    error: wp.array2d(dtype=F64),
):
    """Advance reaction, reversible condensation and both wall ledgers."""
    i, r = wp.tid()
    odds = (
        params[i, 0]
        / (F64(1.0) - params[i, 0])
        * wp.exp(params[i, 16] * props[r, 4])
    )
    frac = odds / (F64(1.0) + odds)
    sv = params[i, 1] * F64(298.15) / props[r, 2]
    lv = sv / params[i, 7]
    alpha = params[i, 2]
    matured = F64(1.0) - wp.exp(-dt / (F64(60.0) * params[i, 3]))
    age = F64(1.0) - wp.exp(-dt * params[i, 8] / F64(3600.0))
    wall = F64(1.0) - wp.exp(-dt * params[i, 5] / F64(3600.0))
    share = params[i, 6]
    seed = props[r, 0]
    number = props[r, 1]
    temp = props[r, 2]
    mfp = props[r, 3]
    injection = params[i, 9 + r]
    inter = F64(0.0)
    gas_s = F64(0.0)
    gas_l = F64(0.0)
    part_s = F64(0.0)
    part_l = F64(0.0)
    deposits = F64(0.0)
    supplied = F64(0.0)
    worst = F64(0.0)
    out[i, r, 0] = F64(0.0)
    for j in range(50):
        number_keep = wp.exp(-params[i, 4] * F64(0.1))
        mass_keep = number_keep
        if count_mode == 1:
            number_keep = count_curve[r, j + 1] / count_curve[r, j]
            mass_keep = wp.pow(number_keep, params[i, 14])
        for n in range(sub):
            if n == sub / 2:
                deposits += (part_s + part_l) * (F64(1.0) - mass_keep)
                part_s *= mass_keep
                part_l *= mass_keep
                seed *= mass_keep
                number *= number_keep
            addition = source[r, j * sub + n] * frac * injection
            supplied += addition
            # A quadratic reaction consumes two intermediate equivalents and
            # deposits both in the lower-volatility product family.
            midpoint = inter + F64(0.5) * addition
            competition = midpoint * (
                F64(1.0)
                - F64(1.0)
                / (F64(1.0) + midpoint * params[i, 15] * dt / F64(36000.0))
            )
            inter -= competition
            gas_l += competition
            product = (inter + F64(0.5) * addition) * matured
            inter += addition - product
            gas_s += product * (F64(1.0) - share)
            gas_l += product * share
            aged = gas_s * age
            gas_s -= aged
            gas_l += aged
            deposits += (gas_s + gas_l) * wall
            gas_s *= F64(1.0) - wall
            gas_l *= F64(1.0) - wall
            radius = wp.pow(
                F64(3.0e-9)
                / (F64(4.0) * F64(3.141592653589793))
                * (seed / F64(1770.0) + (part_s + part_l) / F64(1400.0))
                / number,
                F64(1.0 / 3.0),
            )
            transport = number * first_order_mass_transport_k_wp(
                radius,
                vapor_transition_correction_wp(mfp / radius, alpha),
                F64(6e-6),
            )
            kelvin = wp.exp(
                F64(0.06) * MASS / (RT * temp * F64(1400.0) * radius)
            )
            denom = seed + part_s + part_l
            ds = exchange(gas_s, part_s, sv * kelvin / denom, transport, dt)
            dl = exchange(gas_l, part_l, lv * kelvin / denom, transport, dt)
            gas_s -= ds
            part_s += ds
            gas_l -= dl
            part_l += dl
        out[i, r, j + 1] = part_s + part_l
        worst = wp.max(
            worst,
            wp.abs(
                inter + gas_s + gas_l + part_s + part_l + deposits - supplied
            ),
        )
    error[i, r] = worst


def observations(run):
    """Read aligned suspended AMS concentrations at their original times."""
    if run == 2:
        d = pd.read_csv(HERE / "data/observations.csv")
        return d.time_s.to_numpy(), d.value.to_numpy()
    return series(run, "soa_ug_m3")


class Engine:
    """Store gas forcing and experimental conditions once per search."""

    def __init__(self, dt=18, device="cuda:0", proxy_hazard=1.0, nox=None):
        """Precompute observed forcing for the requested step size."""
        self.dt = dt
        self.device = device
        self.sub = 360 // dt
        if self.sub * dt != 360 or self.sub % 2:
            raise ValueError("The timestep must divide the two half intervals")
        source = []
        props = []
        count = []
        counts = pd.read_csv(HERE.parent / "results/number_check.csv")
        for run in ORDER:
            ppb, seed, n, tc = CONDITIONS[run]
            temp = tc + 273.15
            f = np.interp(TIMES, *FORCING[1 if run == 2 else run])
            if run == 2:
                f = f**proxy_hazard
            # Piecewise-exponential precursor consumption as in native replay.
            residual = np.exp(
                np.interp(np.arange(0, 18001, dt), TIMES, np.log(f))
            )
            source.append(-np.diff(residual) * ppb * PRESSURE * OM / (R * temp))
            props.append(
                [
                    seed,
                    n * 1e6,
                    temp,
                    get_molecule_mean_free_path(OM, temp, PRESSURE),
                    np.log((nox or NOX)[run] / 30.0),
                ]
            )
            if run == 2:
                d = pd.read_csv(HERE / "data/context.csv")
                d = d[
                    (d.instrument == "WCPC") & (d.channel == "total_NumberConc")
                ]
                from scipy.optimize import isotonic_regression

                count.append(
                    isotonic_regression(
                        np.clip(np.interp(TIMES, d.time_s, d.value) / n, 0, 1),
                        increasing=False,
                    ).x
                )
            else:
                count.append(counts[counts.run == run].conditioned.to_numpy())
        self.inputs = [
            wp.array(np.array(v), dtype=F64, device=device)
            for v in (source, props, count)
        ]

    def __call__(self, params, count_mode=False):
        """Return one suspended mass curve per candidate and run."""
        params = np.atleast_2d(params).astype(float)
        if params.shape[1] != 17 or not np.isfinite(params).all():
            raise ValueError(
                "Expected 17 finite physical parameters per vector"
            )
        arg = wp.array(params, dtype=F64, device=self.device)
        out = wp.empty(
            (len(params), len(ORDER), 51), dtype=F64, device=self.device
        )
        error = wp.empty(
            (len(params), len(ORDER)), dtype=F64, device=self.device
        )
        wp.launch(
            batch_kernel,
            dim=(len(params), len(ORDER)),
            inputs=[
                arg,
                *self.inputs,
                float(self.dt),
                self.sub,
                int(count_mode),
                out,
                error,
            ],
            device=self.device,
        )
        arr = out.numpy()
        if (
            not np.isfinite(arr).all()
            or arr.min() < -1e-10
            or error.numpy().max() > 1e-7
        ):
            raise ArithmeticError("Nonphysical state or product ledger failure")
        return arr
