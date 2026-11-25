"""CUDA reverse-mode derivatives for constrained daily fits and an empirical κ(D) comparison."""

from functools import lru_cache

import numpy as np
import warp as wp
from observations import SET
from particula.particles import get_kelvin_radius
from scipy.optimize import least_squares, minimize
from scipy.special import expit

KNOTS = np.log([30.0, 60.0, 100.0, 160.0, 300.0])
WIDTH = 0.2572663929304284


@wp.kernel
def combine(
    design: wp.array2d(dtype=wp.float64),
    weight: wp.array(dtype=wp.float64),
    prediction: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    val = wp.float64(0.0)
    for k in range(weight.shape[0]):
        val += design[i, k] * weight[k]
    prediction[i] = val


@wp.kernel
def squared_residual(
    prediction: wp.array(dtype=wp.float64),
    obs: wp.array(dtype=wp.float64),
    loss: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    error = prediction[i] - obs[i]
    wp.atomic_add(loss, 0, error * error)


class AutoLoss:
    """Weighted response matrix contracts particle shares with AIOMFAC activation."""

    def __init__(self, matrix, response):
        self.matrix = wp.array(matrix, dtype=wp.float64, device="cuda:0")
        self.obs = wp.array(response, dtype=wp.float64, device="cuda:0")
        self.z = wp.zeros(
            matrix.shape[1],
            dtype=wp.float64,
            device="cuda:0",
            requires_grad=True,
        )
        self.loss = wp.zeros(
            1, dtype=wp.float64, device="cuda:0", requires_grad=True
        )
        self.prediction = wp.zeros(
            len(response), dtype=wp.float64, device="cuda:0", requires_grad=True
        )
        with wp.Tape() as self.tape:
            self.launch()

    def launch(self):
        wp.launch(
            combine,
            len(self.obs),
            [self.matrix, self.z, self.prediction],
            device="cuda:0",
        )
        wp.launch(
            squared_residual,
            len(self.obs),
            [self.prediction, self.obs, self.loss],
            device="cuda:0",
        )

    def __call__(self, z):
        self.z.assign(np.ascontiguousarray(z))
        self.loss.zero_()
        self.tape.zero()
        self.launch()
        self.tape.backward(loss=self.loss)
        return self.loss.numpy()[0], self.z.grad.numpy()


def basis(x, knots):
    return np.column_stack(
        [np.interp(x, knots, col) for col in np.eye(len(knots))]
    )


def fit_pop(pop, x, j, y, scale):
    X = pop.design(x, j)
    M = X * scale[:, None]
    t = y * scale
    ad = AutoLoss(M, t)

    # Scale the squared error to improve solver termination while retaining fractional units in scores.
    def fun(z):
        v, g = ad(z)
        return v * 100, g * 100

    eq = dict(type="eq", fun=lambda z: pop.A @ z - pop.b, jac=lambda z: pop.A)
    opts = dict(
        method="SLSQP",
        bounds=[(0.0, 1.0)] * M.shape[1],
        options=dict(maxiter=500, ftol=2e-11),
    )
    test = np.linspace(0.1, 0.9, M.shape[1])
    v, g = ad(test)
    residual = M @ test - t
    direct = 2 * M.T @ residual
    delta = np.linspace(-0.9, 0.8, len(test))
    h = 1e-5
    fd = (ad(test + h * delta)[0] - ad(test - h * delta)[0]) / (2 * h)
    err = max(
        abs(v - residual @ residual),
        np.max(abs(g - direct)),
        abs(fd - g @ delta),
    )
    assert err < 2e-8, (
        err,
        v,
        residual @ residual,
        g[:5],
        direct[:5],
        fd,
        g @ delta,
    )
    if pop.N == 1:
        z = np.ones(len(pop.knots))
    else:
        sol = minimize(fun, pop.seed, jac=True, constraints=[eq], **opts)
        if not sol.success:
            raise RuntimeError(sol.message)
        z = sol.x
    assert max(abs(pop.A @ z - pop.b)) < 2e-8 and z.min() > -1e-10
    return z, X @ z, err


@lru_cache(maxsize=1)
def kappa_table():
    # On each dry-size curve, solve the wet-growth-factor stationarity condition.
    k = np.geomspace(0.0001, 20, 2049)[None, :]
    target = np.log1p(SET[:, None] / 100)
    lo = np.ones((len(SET), k.size))
    hi = np.full_like(lo, 10000.0)
    for _ in range(54):
        x = (lo + hi) * 0.5
        y = x**3 - 1
        slope = target + np.log1p(k / y) - 3 * k * (y + 1) / (y * (y + k))
        lo = np.where(slope < 0, x, lo)
        hi = np.where(slope >= 0, x, hi)
    x = 0.5 * (lo + hi)
    kelvin = 2e9 * get_kelvin_radius(0.072, 997.0, 0.01801528, 298.15)
    return np.log(k[0]), np.log(
        kelvin / (x * (target + np.log1p(k / (x**3 - 1))))
    )


def fit_kappa(x, j, y, scale, power):
    LK, LD = kappa_table()

    def pred(v, xx, jj):
        lk = v[0] + (v[1] if power else 0.0) * np.log(np.exp(xx) / 80)
        dc = np.array([np.interp(a, LK, LD[b]) for a, b in zip(lk, jj)])
        return expit((xx - dc) / WIDTH)

    starts = [[-1.0, s] for s in [-1.0, 0.0, 1.0]] if power else [[-1.0]]
    fits = [
        least_squares(
            lambda v: (pred(v, x, j) - y) * scale,
            v,
            bounds=([-6.0, -4.0], [1.0, 4.0]) if power else ([-6.0], [1.0]),
        )
        for v in starts
    ]
    sol = min(fits, key=lambda u: u.cost)
    assert sol.success
    return pred(sol.x, x, j), lambda a, b: pred(sol.x, a, b), sol.x
