"""Run explicitly composed particles in finite chamber sampling volumes."""

import json
import sys
from pathlib import Path

import numpy as np
import warp as wp
from scipy.special import ndtri

HERE = Path(__file__).resolve().parent
JOINT = HERE.parent
sys.path.insert(0, str(JOINT))
sys.path.insert(0, str(JOINT / "chains"))
from bank import BANK, OUT  # noqa: E402
from kernels import (  # noqa: E402
    F64,
    State,
    collisions,
    geometry,
    nucleation,
    particle_chemistry,
    record,
    shared_gas,
)
from wall_campaign import ISOPRENE_UG_M3_PER_PPB  # noqa: E402

from gpu import Model  # noqa: E402

DEVICE = "cuda:0"
GRID = np.array([75.0, 100.0, 150.0])
ARCHIVE = json.loads((OUT / "resolution_checked_mechanism.json").read_text())


def upload(a, dtype=F64):
    """Upload contiguous host data to the study GPU."""
    return wp.array(np.ascontiguousarray(a), dtype=dtype, device=DEVICE)


def zeros(shape, dtype=F64):
    """Allocate a zeroed CUDA inventory."""
    return wp.zeros(shape, dtype=dtype, device=DEVICE)


def launch(kernel, dim, args):
    """Submit a direct Warp kernel on the selected CUDA device."""
    wp.launch(kernel, dim=dim, inputs=args, device=DEVICE)


class Chamber:
    """Simulate particle ensembles for multiple kinetic parameter vectors."""

    def __init__(self, bank=BANK):
        """Reuse the archived chamber initial states and precursor histories."""
        self.bank = tuple(bank)
        model = Model(self.bank)
        self.rate_end = 16 + len(self.bank)
        self.runs = model.runs
        self.initial = model.initial
        self.instruments = model.instruments
        self.source = model.data[0].numpy()
        self.channels = model.channels
        self.ids = [r[0] for r in self.runs]
        self.counter_data = model.data[1].numpy()

    def baseline(self, fit="mass"):
        """Load the previously calibrated reaction rates and injection doses."""
        prev = ARCHIVE["fits"][fit]["fit"]
        # Archived rates transfer by reaction identity. New channels start off.
        inherited = dict(zip(BANK, prev["x"][16:], strict=True))
        rates = [inherited.get(channel, 0.0) for channel in self.bank]
        return np.r_[prev["x"][:16], rates, prev["dose"], 1.5, 0.0, 0.0]

    def run(
        self,
        vectors,
        run_ids,
        *,
        n=1024,
        dt=30,
        rng=1,
        minutes=300,
        snapshot=False,
        collision_sweeps=4,
        wall_enabled=True,
        coag_enabled=True,
    ):
        """Integrate one or more parameter vectors on independent particles."""
        if 60 % dt or minutes % 5 or n % 2:
            raise ValueError(
                "Use divisors of 60 seconds, even seeds and five-minute windows"
            )
        vectors = np.atleast_2d(vectors)
        indices = np.tile([self.ids.index(k) for k in run_ids], len(vectors))
        v = np.repeat(vectors, len(run_ids), axis=0)
        if v.shape[1] != self.rate_end + len(self.ids) + 3:
            raise ValueError(
                "Inconsistent source, rate, dose or particle parameter count"
            )
        rows = len(indices)
        init = self.initial[indices]
        sample_air = n / init[:, 1]
        cap = 2 * n
        # Match both measured zeroth and third moments in every realization.
        quant = ndtri((np.arange(n) + 0.5) / n)
        seed = np.exp(3 * np.log(v[:, -3, None]) * quant[None, :])
        seed *= init[:, :1] / seed.sum(axis=1, keepdims=True)
        x = np.zeros((rows, cap, 31))
        x[:, :n, 30] = seed
        particles = upload(x, State)
        air = upload(sample_air)
        source = (
            self.source[indices]
            * v[np.arange(rows), self.rate_end + indices, None, None]
        )
        gross = (
            source[:, :, :3] * v[:, None, 3:6]
            + source[:, :, 3:] * v[:, None, 6:9]
        )
        consumed = source[:, :, :3] + source[:, :, 3:]
        iso = consumed[:, :, 2] / float(ISOPRENE_UG_M3_PER_PPB)
        retained = gross.copy()
        retained[:, :, :2] /= 1 + v[:, None, 12:13] * iso[:, :, None]
        diverted = gross - retained + 0.1 * np.maximum(0.0, consumed - gross)
        supplied = (retained + diverted)[:, :minutes, :].sum(axis=(1, 2))
        feed, extra = upload(retained), upload(diverted)
        params, rates, eddy, alloc, jmax = map(
            upload,
            (v[:, :16], v[:, 16 : self.rate_end], v[:, 9], v[:, -2], v[:, -1]),
        )
        props = zeros((rows, cap, 6))
        gas, arrivals = zeros(rows, State), zeros(rows, State)
        feedback = zeros((rows, 31))
        ledger = zeros((rows, 7))
        totals = zeros((rows, 2))
        sub = 60 // dt
        nsteps = minutes * sub
        # Common random numbers across parameter candidates for the same run.
        # Row-index keyed generator lanes are separated below via run IDs.
        perm = np.random.default_rng(rng).permuted(
            np.tile(
                np.arange(cap, dtype=np.int32), (nsteps * collision_sweeps, 1)
            ),
            axis=1,
        )
        order = upload(perm, wp.int32)
        lower = np.array(
            [5.0 if self.instruments[i] == "WCPC" else 40.0 for i in indices]
        )
        window = upload(
            np.column_stack([lower, np.where(lower == 5, 1e7, 600.0)])
        )
        grid = upload(GRID)
        obs = zeros((rows, minutes // 5 + 1, 42))
        dist = zeros((rows, minutes // 5 + 1, 120))
        gas_history = []

        def measure(k):
            if snapshot:
                gas_history.append(gas.numpy() + feedback.numpy())
            launch(
                record,
                (rows, cap),
                [particles, air, window, grid, obs, dist, k],
            )

        measure(0)
        for s in range(nsteps):
            totals.zero_()
            launch(
                nucleation,
                rows,
                [
                    particles,
                    gas,
                    air,
                    jmax,
                    float(dt),
                    s,
                    3000001 + rng * 4099,
                    len(run_ids),
                    ledger,
                ],
            )
            launch(
                geometry,
                (rows, cap),
                [
                    particles,
                    air,
                    eddy,
                    int(wall_enabled),
                    init[0, 5],
                    init[0, 6],
                    float(dt),
                    s,
                    1100003 + rng * 4099,
                    len(run_ids),
                    ledger,
                    props,
                    totals,
                ],
            )
            launch(
                shared_gas,
                rows,
                [
                    gas,
                    feedback,
                    feed,
                    extra,
                    params,
                    self.channels,
                    rates,
                    totals,
                    arrivals,
                    s,
                    sub,
                    float(dt),
                ],
            )
            launch(
                particle_chemistry,
                (rows, cap),
                [
                    particles,
                    arrivals,
                    feedback,
                    totals,
                    props,
                    self.channels,
                    rates,
                    float(dt),
                    alloc,
                ],
            )
            for sweep in range(collision_sweeps if coag_enabled else 0):
                launch(
                    collisions,
                    (rows, cap // 2),
                    [
                        particles,
                        air,
                        props,
                        order,
                        s * collision_sweeps + sweep,
                        2100007 + rng * 4099,
                        len(run_ids),
                        dt / collision_sweeps,
                        init[0, 5],
                        init[0, 6],
                        ledger,
                    ],
                )
            if (s + 1) % (5 * sub) == 0:
                measure((s + 1) // (5 * sub))
        history = obs.numpy()
        events = ledger.numpy()
        stock = (
            history[:, -1, 0]
            + gas.numpy().sum(axis=1)
            + feedback.numpy().sum(axis=1)
            + events[:, 0]
        )
        if (
            not np.isfinite(history).all()
            or events[:, 4].max() >= 1
            or events[:, 6].max() > 0
        ):
            raise ArithmeticError(
                f"Pair probability {events[:, 4].max()}, "
                f"unplaced births {events[:, 6].max()}"
            )
        np.testing.assert_allclose(stock, supplied, atol=1e-7, rtol=1e-9)
        np.testing.assert_allclose(
            history[:, -1, 1] + events[:, 1], init[:, 0], atol=1e-8
        )
        np.testing.assert_array_equal(
            history[:, -1, 2], n + events[:, 5] - events[:, 2] - events[:, 3]
        )
        final = particles.numpy() if snapshot else None
        if snapshot and (not np.isfinite(final).all() or final.min() < -1e-9):
            raise ArithmeticError("Invalid particle species inventory")
        self.gas_history = np.stack(gas_history, axis=1) if snapshot else None
        return (
            history.reshape(len(vectors), len(run_ids), -1, 42),
            dist.numpy(),
            events,
            final,
            sample_air,
        )
