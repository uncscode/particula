"""Resolve atmospheric two-mode boxes with native CUDA particle collisions."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import particula
import warp
from particula.dynamics import BrownianCoagulationStrategy, Coagulation
from particula.gpu.kernels.coagulation import CoagulationSubcyclingConfig
from particula.particles import get_binned_number_spectrum
from scipy.special import ndtri
from scipy.stats import t as student
from scripts.mam4xx_coagulation.matched import matched
from scripts.mam4xx_coagulation.particles import DENSITY, WIDTH, mode_population
from scripts.mam4xx_coagulation.regime_box import MODES


def ensemble(modes, slots, dt, seeds, max_subcycles=4096):
    """Compute intervals across independent stochastic trajectories."""
    diameter, sigma, number = np.array(modes).T
    counts = np.rint(slots * number / number.sum()).astype(int)
    np.testing.assert_allclose(
        counts, slots * number / number.sum(), atol=1e-10
    )
    d = []
    for dg, s, n in zip(diameter, sigma, counts, strict=True):
        samples = dg * np.exp(np.log(s) * ndtri((np.arange(n) + 0.5) / n))
        d.extend(
            samples
            * np.cbrt(
                dg**3 * np.exp(4.5 * np.log(s) ** 2) / np.mean(samples**3)
            )
        )
    volume = np.asarray(d) ** 3 * np.pi / 6  # micrometres cubed per particle
    edges = 100 * 10 ** (np.arange(-16, 25) / 8)
    process = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    traces = []
    spectra = []
    for seed in seeds:
        box = mode_population(60, 1, slots)
        box.particles.masses[0, :, 0] = volume * 1e-18 * DENSITY
        box.particles.volume[0] = slots / (number.sum() * 1e6)
        metrics = []
        hist = []
        with process.prepare(
            box,
            device="cuda:0",
            time_step=dt,
            root_seed=seed,
            coagulation_subcycling=CoagulationSubcyclingConfig(
                0.01, max_subcycles, 64 * 1024**2
            ),
        ) as owner:
            for duration in (0, 3600, 18000, 64800):
                for _ in range(round(duration / dt)):
                    owner.step()
                p = owner.snapshot().particles
                live = p.concentration[0] > 0
                v = p.masses[0, live, 0] / DENSITY * 1e18
                count = p.concentration[0, live] / p.volume[0] / 1e6
                size = 1e3 * np.cbrt(v * 6 / np.pi)
                np.testing.assert_allclose(
                    v @ count, volume.sum() * number.sum() / slots, rtol=2e-12
                )
                big = size >= 100
                metrics.append(
                    [count[~big].sum(), count[big].sum(), v[big] @ count[big]]
                )
                bins, low, high = get_binned_number_spectrum(
                    size * 1e-9,
                    count * 1e6,
                    edges * 1e-9,
                    include_upper_edge=False,
                )
                hist.append(np.r_[low, bins, high] * 1e-6)
        traces.append(metrics)
        spectra.append(hist)

    def stats(x):
        return dict(
            mean=np.mean(x, axis=0).tolist(),
            ci95=(
                np.std(x, axis=0, ddof=1)
                * student.ppf(0.975, len(seeds) - 1)
                / np.sqrt(len(seeds))
            ).tolist(),
        )

    return dict(
        metrics=stats(traces),
        spectrum=stats(spectra),
        interior_edges_nm=edges.tolist(),
        slots=slots,
        dt=dt,
        seeds=list(seeds),
    )


def checkpointed_ensemble(regime, variant, modes, slots, dt, seeds):
    """Optionally resume exact cases after an interrupted workstation run."""
    key = {
        "modes": np.array(modes).tolist(),
        "slots": slots,
        "dt": dt,
        "max_subcycles": 4096 if regime == "polluted" else 128,
        "seeds": list(seeds),
        "particula": particula.__version__,
        "warp": warp.__version__,
        "device": warp.get_device("cuda:0").name,
        "source": hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(__file__).with_name("particles.py").read_bytes()
        ).hexdigest(),
    }
    folder = os.environ.get("PARTICULA_MAM4XX_CHECKPOINTS")
    destination = None
    if folder:
        directory = Path(folder)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{regime}-{variant}.json"
        if destination.exists():
            record = json.loads(destination.read_text())
            if record["inputs"] == key:
                return record["result"]
    start = time.monotonic()
    result = ensemble(modes, slots, dt, seeds, key["max_subcycles"])
    result["max_subcycles"] = key["max_subcycles"]
    result["elapsed_seconds"] = time.monotonic() - start
    if destination:
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps({"inputs": key, "result": result}))
        temporary.replace(destination)
    return result


def comparison():
    """Match number and volume across archived, modal and native boxes."""
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/data/mam4xx-coagulation/regional-modal.csv"
    )
    archive = np.genfromtxt(path, delimiter=",", names=True)
    result = []
    for i, (reg, modes) in enumerate(MODES.items()):
        dg, sw, n = np.array(modes[:2]).T
        # Project measured number/volume moments to archived geometric widths.
        adjusted = dg * np.exp(1.5 * (np.log(sw) ** 2 - np.log(WIDTH) ** 2))
        fixed = list(zip(adjusted, WIDTH, n, strict=True))
        central = matched(1000 * adjusted[1], 32, fixed)
        fine = [
            archive[(archive["regime"] == i) & (archive["dt_s"] == dt)]
            for dt in (10, 30)
        ]
        for arr in fine:
            v = arr["v_ait_um3_cm3"] + arr["v_acc_um3_cm3"]
            np.testing.assert_allclose(v / v[0], 1.0, rtol=1e-11)

        def total(arr):
            return arr["n_ait_cm3"] + arr["n_acc_cm3"]

        slots = [1520, 1800, 414][i]
        dt = 600 if reg == "polluted" else 300
        seeds = range(201, 205) if reg == "polluted" else range(201, 207)
        refined_seeds = (
            range(401, 405) if reg == "polluted" else range(301, 305)
        )
        result.append(
            dict(
                regime=reg,
                modal=central,
                archive_cm3=total(fine[0]).tolist(),
                archive_dt_delta_cm3=(total(fine[1]) - total(fine[0])).tolist(),
                fixed=checkpointed_ensemble(
                    reg, "fixed", fixed, slots, dt, seeds
                ),
                measured=checkpointed_ensemble(
                    reg, "measured", modes[:2], slots, dt, seeds
                ),
                refinement=checkpointed_ensemble(
                    reg,
                    "refinement",
                    modes[:2],
                    2 * slots,
                    dt // 2,
                    refined_seeds,
                ),
            )
        )
    return result
