"""Run particle-resolved SMPS pulses for the extended collision laws."""

import json
import platform
from argparse import Namespace

import numpy as np
import warp as wp
from chamber_gpu import COLLECTORS, N100, simulate
from instrument import HERE, TARGETS
from pulse_pairs import MODELS
from study import revision

SEEDS = (103, 227, 431, 653)
REFINED = ("bare", "dispersion_only", "full", "full_settling_shear_10", "gatti")
REPS = 2048


def cases():
    """Enumerate time-step and model combinations."""
    for dt, models in [(5.0, MODELS), (2.5, REFINED)]:
        for model in models:
            yield dt, model, False
            yield dt, model, True


def label(dt, model, seed, conditioned):
    """Identify a unique pulse output."""
    source = "smps" if conditioned else "reference"
    return f"pulse-{source}-{model}-dt{dt:g}-{seed}"


def main():
    """Require native CUDA and complete every prescribed run."""
    runs = []
    for dt, model, conditioned in cases():
        for seed in SEEDS:
            name = label(dt, model, seed, conditioned)
            simulate(
                Namespace(
                    reps=REPS,
                    seed=seed,
                    dt=dt,
                    duration=300,
                    ion=2.0,
                    tube=0.1,
                    bound="upper",
                    parent=1.0,
                    ideal=False,
                    vdw=False,
                    label=name,
                    conditioned=conditioned,
                    model=model,
                )
            )
            meta = json.loads((HERE / f"{name}.json").read_text())
            np.testing.assert_equal(meta["mass_charge_number_checks"], True)
            runs.append(meta)
    record = dict(
        revision=revision(),
        device=wp.get_device("cuda:0").name,
        warp=wp.__version__,
        numpy=np.__version__,
        python=platform.python_version(),
        models=MODELS,
        refined=REFINED,
        seeds=SEEDS,
        reps_per_condition=REPS,
        targets_nm=TARGETS,
        slots=COLLECTORS + 1,
        loaded_cm3=N100,
        duration_s=300,
        steps_s=[5.0, 2.5],
        origin="source-off pulse",
        fitted_tracer_log_slope=next(
            x["tracer_prior_log_slope"] for x in runs if x["conditioned"]
        ),
        runs=len(runs),
        box_histories=sum(x["boxes"] for x in runs),
        accepted_collisions=sum(x["accepted_collisions"] for x in runs),
        seconds=sum(x["seconds"] for x in runs),
        mass_charge_number_checks=all(
            x["mass_charge_number_checks"] for x in runs
        ),
    )
    (HERE / "pulse-kernels-run.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
