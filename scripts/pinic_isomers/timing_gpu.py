"""Forward-mode derivatives of normalized gas–particle exchange on CUDA."""

import numpy as np
import warp as wp
from scipy.interpolate import interp1d
from scripts.pinic_isomers.physics import PINIC


@wp.func
def product(a: wp.vec2d, b: wp.vec2d):
    """Multiply two value/directional-derivative pairs."""
    return wp.vec2d(a[0] * b[0], a[1] * b[0] + a[0] * b[1])


@wp.func
def inverse(a: wp.vec2d):
    """Reciprocate a differentiable positive scalar."""
    return wp.vec2d(wp.float64(1.0) / a[0], -a[1] / (a[0] * a[0]))


@wp.func
def exponential(a: wp.vec2d):
    """Exponentiate a dual number."""
    x = wp.exp(a[0])
    return wp.vec2d(x, x * a[1])


@wp.kernel
def integrate(
    parameters: wp.array(dtype=wp.float64),
    bulk: wp.array(dtype=wp.float64),
    beta: wp.array(dtype=wp.float64),
    dt: wp.float64,
    pinic: wp.float64,
    delayed: wp.int32,
    out: wp.array3d(dtype=wp.float64),
):
    """Propagate one directional derivative per lane, storing every endpoint."""
    species, lane = wp.tid()
    one = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
    tau = inverse(exponential(wp.vec2d(parameters[0], wp.float64(lane == 0))))
    cstar = wp.vec2d(pinic, wp.float64(0.0))
    if species == 1:
        cstar = product(
            cstar, exponential(wp.vec2d(parameters[1], wp.float64(lane == 1)))
        )
    kval = inverse(exponential(wp.vec2d(parameters[2], wp.float64(lane == 2))))
    fraction = wp.vec2d(parameters[3], wp.float64(lane == 3))
    primary = one
    precursor = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    part = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    step = exponential(-dt * tau)
    lag_half = exponential(-wp.float64(0.5) * dt * kval)
    lag = product(lag_half, lag_half)
    for i in range(bulk.shape[0]):
        # Midpoint source over each interval, with the precursor inventory
        # retained explicitly so a delayed source never creates extra material.
        generated = product(primary, one - step)
        primary_new = product(primary, step)
        precursor_new = product(precursor, lag) + product(generated, lag_half)
        total = one - wp.float64(0.5) * (primary + primary_new)
        if delayed == 1 and species == 0:
            total = total - product(
                fraction, wp.float64(0.5) * (precursor + precursor_new)
            )
        slope = one + cstar / bulk[i]
        target = product(total, inverse(slope))
        part = target + product(
            part - target, exponential(-dt * beta[i] * slope)
        )
        out[species, i + 1, lane] = part[1]
        if lane == 0:
            out[species, i + 1, 4] = part[0]
        primary = primary_new
        precursor = precursor_new


class Exchange:
    """Cache measured mass and physical transfer coefficients on the device."""

    def __init__(
        self, smps, time, radius_nm, dt=0.25, loading=1.0, early="ramp"
    ):
        """Prepare the loading history and kinetic coefficients."""
        self.dt = dt
        self.time = time
        self.grid = np.arange(0.0, np.ceil(max(time)) + dt / 2, dt)
        tm = 0.5 * (self.grid[:-1] + self.grid[1:])
        mass = loading * np.interp(
            tm,
            np.r_[0.0, smps[:, 0]],
            np.r_[0.0 if early == "ramp" else smps[0, 1], smps[:, 1]],
        )
        r = radius_nm * 1e-9
        temp = 297.65
        diffusivity = 5e-6 / 0.83
        kn = (
            3
            * diffusivity
            / (r * np.sqrt(8 * 8.314462618 * temp / (np.pi * 0.18621)))
        )
        f = (1 + kn) / (1 + 1.71 * kn + 4.0 / 3 * kn * kn)  # unit accommodation
        beta = 60 * 3 * diffusivity * (mass * 1e-9) / (1200 * r * r) * f
        self.cstar = (
            PINIC
            * 10 ** (-5691.7 / temp + 5691.7 / 298.15)
            * 298.15
            / temp
            * np.exp(2 * 0.03 * 0.18621 / (8.314462618 * temp * 1200 * r))
        )
        self.mass = wp.array(mass, dtype=wp.float64, device="cuda:0")
        self.beta = wp.array(beta, dtype=wp.float64, device="cuda:0")
        self.out = wp.zeros(
            (2, len(self.grid), 5), dtype=wp.float64, device="cuda:0"
        )

    def predict(self, p, delayed):
        """Return two channel shapes and their four parameter derivatives."""
        param = wp.array(p, dtype=wp.float64, device="cuda:0")
        wp.launch(
            integrate,
            (2, 4),
            [
                param,
                self.mass,
                self.beta,
                self.dt,
                self.cstar,
                int(delayed),
                self.out,
            ],
            device="cuda:0",
        )
        y = interp1d(self.grid, self.out.numpy(), axis=1)(self.time)
        if (
            not np.all(np.isfinite(y))
            or y[:, :, 4].min() < -1e-9
            or y[:, :, 4].max() > 1 + 1e-9
        ):
            raise ValueError("Invalid tracer inventory")
        return y[:, :, 4], y[:, :, :4]
