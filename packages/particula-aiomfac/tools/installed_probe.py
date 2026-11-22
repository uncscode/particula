"""Exercise installed wheel entry points from outside the checkout."""

import json
import runpy
import sys
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


def mixed():
    from particula_aiomfac import AIOMFACModel, MixtureSpec

    ref = PACKAGE / "reference"
    groups = json.loads((ref / "cases/components.json").read_text())
    case = json.loads((ref / "cases/mixed.json").read_text())
    expected = json.loads((ref / "expected/mixed.json").read_text())
    mixture = MixtureSpec(
        {
            k: {int(g): v for g, v in groups[k].items()}
            for k in case["components"]
        }
    )
    model = AIOMFACModel.prepare(mixture)
    r = model.evaluate(case["fractions"], case["temperature_K"])
    np.testing.assert_allclose(
        r.neutral_activities[0],
        expected["neutrals"]["water"]["activity"],
        atol=1e-10,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        r.mean_electrolyte_coefficients[0],
        expected["salts"]["202/242"]["gamma_mean"],
        atol=1e-10,
        rtol=1e-8,
    )


if __name__ == "__main__":
    if sys.argv[1] == "base":
        base_activities()
    else:
        mixed()
        if sys.argv[1] == "combined":
            workflow = runpy.run_path(str(PACKAGE / "examples/closed_water.py"))
            for saturation in (0.98, 1.02):
                workflow["trajectory"](saturation)
            workflow["trajectory"](1.02, permuted=True)
