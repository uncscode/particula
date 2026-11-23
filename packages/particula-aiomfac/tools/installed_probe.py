"""Exercise installed wheel entry points from outside the checkout."""

import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]


def base_activities():
    from particula.particles.activity_factories import ActivityFactory
    from particula.particles.activity_strategies import (
        ActivityBAT,
        ActivityIdealMass,
        ActivityKappaParameter,
    )

    class Custom(ActivityIdealMass):
        def partial_pressure(self, pure_vapor_pressure, mass_concentration):
            return mass_concentration * 0.25

    masses = np.array([[0.75, 0.25]])
    for model in (
        ActivityFactory().get_strategy("mass_ideal"),
        ActivityKappaParameter(
            np.array([0.0, 0.1]),
            np.array([1000.0, 1200.0]),
            np.array([0.018, 0.15]),
        ),
        ActivityBAT(0.15, 0.5, 1200.0),
        Custom(),
    ):
        expected = model.partial_pressure(np.ones(2), masses)
        np.testing.assert_allclose(
            model.neutral_activities(masses, 298, [1, 0], [True]),
            np.broadcast_to(expected, masses.shape)[:, [1, 0]],
        )
    try:
        ActivityFactory().get_strategy("aiomfac")
    except ImportError as error:
        assert 'pip install "particula[aiomfac]"' in str(error)
    else:
        raise AssertionError("unexpected optional backend")


def calculate(case_id, execution):
    from particula_aiomfac import AIOMFACModel, MixtureSpec

    ref = PACKAGE / "reference"
    groups = json.loads((ref / "cases/components.json").read_text())
    case = json.loads((ref / f"cases/{case_id}.json").read_text())
    expected = json.loads((ref / f"expected/{case_id}.json").read_text())
    mixture = MixtureSpec(
        {
            k: {int(g): v for g, v in groups[k].items()}
            for k in case["components"]
        }
    )
    model = AIOMFACModel.prepare(mixture, execution=execution)
    r = model.evaluate(case["fractions"], case["temperature_K"])
    np.testing.assert_allclose(
        r.neutral_activities[0],
        expected["neutrals"]["water"]["activity"],
        atol=1e-10,
        rtol=1e-8,
    )
    if case_id != "mixed":
        assert r.solver_status == 1
        return
    np.testing.assert_allclose(
        r.mean_electrolyte_coefficients[0],
        expected["salts"]["202/242"]["gamma_mean"],
        atol=1e-10,
        rtol=1e-8,
    )


def installed_sparse(cuda):
    """Build with the installed entry point outside the tree, with no compiler on PATH."""
    from particula_aiomfac.sparse import ActivityTable

    devices = ("cpu", "cuda:0") if cuda else ("cpu",)
    with tempfile.TemporaryDirectory() as tmp:
        cli = Path(sys.executable).with_name("particula-aiomfac-table")
        for build_device in devices:
            artifact = Path(tmp) / "mixed.npz"
            subprocess.run(
                [
                    cli,
                    "build",
                    "mixed-water",
                    "--device",
                    build_device,
                    "--output",
                    artifact,
                ],
                cwd=tmp,
                env=dict(os.environ, PATH=str(cli.parent), PYTHONPATH=""),
                check=True,
            )
            for run_device in devices:
                model = ActivityTable.load(artifact).prepare(device=run_device)
                masses = np.tile([0.977, 0.001, 0.002, 0.020], (2, 1))
                masses[1, 1] *= 2
                a, _, reason = model.evaluate(masses, 298.0)
                assert abs(a[0, 0] - 0.9878567852129456) < 1e-4
                np.testing.assert_array_equal(reason, [0, 2])


if __name__ == "__main__":
    assert shutil.which("gfortran") is None
    if sys.argv[1] == "base":
        base_activities()
    else:
        installed_sparse("cuda" in sys.argv[2:])
        for device in ("cpu", "cuda") if "cuda" in sys.argv[2:] else ("cpu",):
            for case in ("mixed", "sulfate", "coupled-carbonate-sulfate"):
                calculate(case, device)
        if sys.argv[1] == "combined":
            workflow = runpy.run_path(str(PACKAGE / "examples/closed_water.py"))
            for saturation in (0.98, 1.02):
                workflow["trajectory"](saturation)
            workflow["trajectory"](1.02, permuted=True)
            examples = files("particula").joinpath(
                "docs/studies/examples/thermodynamics"
            )
            cuda = "cuda" in sys.argv[2:]
            box = runpy.run_path(
                str(examples.joinpath("activity/aiomfac_closed_box.py"))
            )
            box["simulate"](cuda=cuda)
            parcel = runpy.run_path(
                str(
                    examples.joinpath(
                        "parcel-experiments/aiomfac_multivapor_parcel.py"
                    )
                )
            )
            parcel["simulate"](device="cuda:0" if cuda else "cpu")
            from dataclasses import replace

            from particula_aiomfac.adapter import AIOMFACActivity

            initial = parcel["initial_parcel"]()
            hybrid = AIOMFACActivity(
                initial.activity_strategy.mixture,
                dict(enumerate(initial.material_names)),
                evaluation="hybrid",
                table="mixed-two-vapor",
            )
            initial = replace(initial, activity_strategy=hybrid)
            reference = parcel["run_parcel"](
                initial, timestep=1.0, updraft=1.0, steps=150
            )
            b = box["make_box"](table="mixed-two-vapor")
            box["compare"](
                box["native"](b, 8), box["radau"](box["make_box"]()), b.targets
            )
            from particula.execution.condensation_ode import CondensationODE

            b = box["make_box"](table="mixed-two-vapor", permuted=True)
            ode = CondensationODE(
                b.aerosol,
                b.process,
                method="ros3",
                rtol=1e-8,
                particle_atol=1e-30,
                gas_atol=1e-16,
                first_step=1e-5,
            )
            rows = [box["sample"](b, ode.snapshot())]
            for stop in box["TIMES"][1:]:
                while ode.time < stop:
                    ode.step(stop)
                rows.append(box["sample"](b, ode.snapshot()))
            box["compare"](
                rows, box["radau"](box["make_box"](permuted=True)), b.targets
            )
            if cuda:
                for method in ("rk23", "ros3"):
                    box["qualify_restart"](method, table="mixed-two-vapor")
                    solver = parcel[
                        "ParcelRK23" if method == "rk23" else "ParcelROS3"
                    ]
                    parcel["cuda_comparison"](
                        reference, solver(rtol=2e-6), "cuda:0", restart=True
                    )
