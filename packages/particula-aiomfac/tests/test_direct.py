"""Parity with independent, full-precision Fortran activity calculations."""

import json
import os
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from particula_aiomfac import AIOMFACModel, ConvergenceError, MixtureSpec
from particula_aiomfac._terms import electrolyte_terms
from particula_aiomfac.model import _prepare_fixed

REF = Path(__file__).resolve().parents[1] / "reference"


def read(directory, name):
    return json.loads((REF / directory / (name + ".json")).read_text())


def spec(names):
    groups = read("cases", "components")
    return MixtureSpec(
        {k: {int(i): v for i, v in groups[k].items()} for k in names}
    )


class TestDirect(unittest.TestCase):
    def test_fortran(self):
        self.compare_fortran("cpu")

    @unittest.skipUnless(
        os.environ.get("AIOMFAC_TEST_CUDA") == "1", "requires NVIDIA device"
    )
    def test_fortran_cuda(self):
        self.compare_fortran("cuda")

    def compare_fortran(self, execution):
        for path in sorted((REF / "cases").glob("*.json")):
            name = path.stem
            if name == "components":
                continue
            with self.subTest(case=name):
                case, expected = read("cases", name), read("expected", name)
                mixture = spec(case["components"])
                if "expected_solver_error" in case:
                    model = AIOMFACModel.prepare(mixture, execution=execution)
                    with self.assertRaises(ConvergenceError):
                        model.evaluate(case["fractions"], case["temperature_K"])
                    continue
                if expected["status"].startswith("upstream_error"):
                    pattern = {
                        "missing-pair": "221/249",
                        "invalid-subgroup-pair": "13",
                    }[name]
                    with self.assertRaisesRegex(ValueError, pattern):
                        AIOMFACModel.prepare(mixture)
                    continue
                model = AIOMFACModel.prepare(mixture)
                if name == "analog":
                    self.assertEqual(model.estimated_pairs, ((221, 247),))
                calculator = (
                    model
                    if execution == "cpu"
                    else AIOMFACModel.prepare(mixture, execution=execution)
                )
                if name == "negative-input":
                    with self.assertRaisesRegex(ValueError, "batch index"):
                        calculator.evaluate(
                            case["fractions"], case["temperature_K"]
                        )
                    continue
                mass = np.array(case["fractions"])
                if case["basis"] == "mole":
                    mw = np.empty(model.ncomponents + model.virtual_count)
                    mw[model.neutral_indices] = model.neutral_mw
                    mw[model.salt_indices] = model.salt_mw
                    mass *= mw[: model.ncomponents]
                result = calculator.evaluate(
                    mass, case["temperature_K"], diagnostics=True
                )
                if model.sulfate or model.carbonate:
                    self.assertEqual(result.solver_status, 1)
                    errors = result.diagnostics["equilibrium_residuals"]
                    self.assertLessEqual(errors[0], 1e-8)
                    self.assertLessEqual(max(errors[1:]), 1e-10)
                    for j, key in enumerate(result.ion_ids):
                        oracle = expected["ions"][str(key)]["molality"]
                        if oracle > 0:
                            assert_allclose(
                                np.log(result.ion_molalities[j]),
                                np.log(oracle),
                                atol=1e-8,
                                rtol=0,
                            )
                    if model.carbonate:
                        actual = result.neutral_mole_fractions[model.co2]
                        target = expected["neutrals"][
                            model.neutrals[model.co2]
                        ]["x"]
                        assert_allclose(
                            np.log(actual), np.log(target), atol=1e-8, rtol=0
                        )
                self.assertEqual(result.warnings, expected["warning"])
                for scope, ids, defined, quantities in [
                    (
                        "neutrals",
                        result.neutral_ids,
                        result.neutral_defined,
                        {
                            "activity": result.neutral_activities,
                            "x": result.neutral_mole_fractions,
                            "ln_gamma": result.neutral_log_coefficients,
                        },
                    ),
                    (
                        "ions",
                        list(map(str, result.ion_ids)),
                        result.ion_defined,
                        {
                            "ln_gamma": result.ion_log_coefficients,
                            "molality": result.ion_molalities,
                        },
                    ),
                    (
                        "salts",
                        [f"{c}/{a}" for c, a in result.electrolyte_ids],
                        result.electrolyte_defined,
                        {
                            "iap": result.ion_activity_products,
                            "gamma_mean": result.mean_electrolyte_coefficients,
                        },
                    ),
                ]:
                    self.assertEqual(set(ids), set(expected[scope]))
                    for j, key in enumerate(ids):
                        record = expected[scope][key]
                        self.assertEqual(defined[j], record["defined"])
                        for label, values in quantities.items():
                            if record[label] is None:
                                self.assertTrue(np.isnan(values[j]))
                            else:
                                log = label == "ln_gamma"
                                assert_allclose(
                                    values[j],
                                    record[label],
                                    atol=1e-8 if log else 1e-10,
                                    rtol=0 if log else 1e-8,
                                    err_msg=f"{name}.{key}.{label}",
                                )
                records = [expected["neutrals"][k] for k in result.neutral_ids]
                records.extend(expected["ions"][str(k)] for k in result.ion_ids)
                for k in ("sr", "mr", "lr"):
                    assert_allclose(
                        result.diagnostics[k],
                        [r[k] for r in records],
                        atol=1e-8,
                        rtol=0,
                    )
                for key, oracle in (
                    ("ionic_strength", "ionic_strength_mol_per_kg"),
                    ("molality_conversion", "molality_conversion"),
                ):
                    assert_allclose(
                        result.diagnostics[key],
                        expected[oracle],
                        atol=1e-10,
                        rtol=1e-8,
                    )
                assert_allclose(
                    result.ion_molalities
                    @ model.charge
                    / max(1, result.ion_molalities @ abs(model.charge)),
                    0,
                    atol=1e-10,
                )
                mask = result.neutral_defined
                assert_allclose(
                    result.neutral_activities[mask],
                    result.neutral_mole_fractions[mask]
                    * np.exp(result.neutral_log_coefficients[mask]),
                )

    def test_three_ion_fixed_state(self):
        state = read("expected", "sulfate")
        mixture = MixtureSpec(
            {
                "water": {16: 1},
                "NH4HSO4": {204: 1, 248: 1},
                "H2SO4": {205: 2, 261: 1},
            }
        )
        fixed = _prepare_fixed(mixture.components)
        mol = np.array([state["ions"][str(i)]["molality"] for i in fixed.ions])
        mr, _, _, _ = electrolyte_terms(
            fixed,
            np.array([[1.0]]),
            mol[None, :],
            np.array([read("cases", "sulfate")["temperature_K"]]),
        )
        assert_allclose(
            mr[0],
            [
                state["neutrals"]["water"]["mr"],
                *(state["ions"][str(i)]["mr"] for i in fixed.ions),
            ],
            atol=1e-8,
            rtol=0,
        )

    def test_batch_and_failures(self):
        case = read("cases", "mixed")
        model = AIOMFACModel.prepare(spec(case["components"]))
        masses = np.array(case["fractions"])
        batch = (
            np.tile(masses, (2, 3, 1)) * np.array([1e-23, 1e6])[:, None, None]
        )
        temperatures = [288, 298, 308]
        result = model.evaluate(batch, temperatures)
        for i, t in enumerate(temperatures):
            assert_allclose(
                result.neutral_activities[:, i],
                np.tile(model.evaluate(masses, t).neutral_activities, (2, 1)),
                rtol=1e-12,
            )
        for t in (0, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, r"\(1, 0\)"):
                model.evaluate(batch, [[298], [t]])
        for bad in (np.nan, -1, np.inf, 0):
            broken = batch.copy()
            broken[1, 2] = bad
            with self.assertRaisesRegex(ValueError, r"\(1, 2\)"):
                model.evaluate(broken, 298)
        self.assertTrue(
            AIOMFACModel.prepare(spec(["water", "Na2SO4", "HCl"])).sulfate
        )
        self.assertFalse(
            AIOMFACModel.prepare(spec(["water", "Na2SO4"])).sulfate
        )
        for execution in ("resident", "hybrid"):
            with self.assertRaisesRegex(ValueError, "execution"):
                AIOMFACModel.prepare(spec(["water"]), execution=execution)
