"""Closed analytical inventories, independent rows, and failure propagation."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from particula_aiomfac import AIOMFACModel, ConvergenceError
from test_direct import read, spec


class TestEquilibrium(unittest.TestCase):
    def test_inventory_and_order(self):
        self.check_order("cpu")

    @unittest.skipUnless(
        os.getenv("AIOMFAC_TEST_CUDA") == "1", "requires NVIDIA device"
    )
    def test_device_inventory_and_order(self):
        self.check_order("cuda")

    def check_order(self, execution):
        case = read("cases", "coupled-carbonate-sulfate")
        names = case["components"]
        cpu = AIOMFACModel.prepare(spec(names))
        model = AIOMFACModel.prepare(spec(names), execution=execution)
        reverse = AIOMFACModel.prepare(spec(names[::-1]), execution=execution)
        rows = np.array(
            [
                case["fractions"],
                [0.96, 0.03, 0.01],
                [1.0, 0.0, 0.0],
                [1.0, 1.0e-7, 1.0e-7],
            ]
        )
        ts = np.array([298.15, 310.0, 285.0, 295.0])
        result = model.evaluate(rows, ts, diagnostics=True)
        other = reverse.evaluate(rows[::-1, ::-1], ts[::-1], diagnostics=True)
        for j, key in enumerate(result.neutral_ids):
            np.testing.assert_allclose(
                result.neutral_activities[:, j],
                other.neutral_activities[::-1, other.neutral_ids.index(key)],
                atol=1e-12,
            )
        for j, key in enumerate(result.ion_ids):
            np.testing.assert_allclose(
                result.ion_molalities[:, j],
                other.ion_molalities[::-1, other.ion_ids.index(key)],
                atol=1e-14,
            )
        species = result.species_moles_per_kg_input

        def ion(key):
            return species[
                :, len(result.neutral_ids) + result.ion_ids.index(key)
            ]

        # Recover original analytical inventories independently of model diagnostics.
        amount = rows / rows.sum(axis=1, keepdims=True)
        salt_mass = {
            j: mw for j, mw in zip(cpu.salt_indices, cpu.salt_mw, strict=True)
        }
        np.testing.assert_allclose(
            ion(248) + ion(261), amount[:, 1] / salt_mass[1], rtol=1e-10
        )
        carbon = (
            ion(250)
            + ion(262)
            + species[:, result.neutral_ids.index("CO2(aq)")]
        )
        np.testing.assert_allclose(
            carbon, amount[:, 2] / salt_mass[2], rtol=1e-10
        )
        np.testing.assert_array_equal(result.solver_status, [1, 1, 0, 1])
        broken = rows.copy()
        broken[2, 0] = -1
        with self.assertRaisesRegex(ValueError, r"batch index \(2,\)"):
            model.evaluate(broken, ts)
        repeat = model.evaluate(rows, ts)
        np.testing.assert_array_equal(
            repeat.species_moles_per_kg_input, species
        )

    def test_dilute_sulfate(self):
        self.check_dilute("cpu")

    @unittest.skipUnless(
        os.getenv("AIOMFAC_TEST_CUDA") == "1", "requires NVIDIA device"
    )
    def test_device_dilute_and_exhaustion(self):
        self.check_dilute("cuda")
        with patch(
            "particula_aiomfac._cuda.CARBONATE_EVALUATIONS_PER_DIMENSION", 1
        ):
            model = AIOMFACModel.prepare(
                spec(["water", "NH4HSO4", "KHCO3"]), execution="cuda"
            )
        with self.assertRaises(ConvergenceError) as error:
            model.evaluate([[1.0, 0.0, 0.0], [0.97, 0.02, 0.01]], 298.15)
        self.assertEqual(error.exception.batch_index, (1,))

    def check_dilute(self, execution):
        model = AIOMFACModel.prepare(
            spec(["water", "NH4HSO4"]), execution=execution
        )
        r = model.evaluate(
            [[1.0, x] for x in (0.0, 1e-24, 1e-18, 1e-12)],
            298.15,
            diagnostics=True,
        )
        np.testing.assert_array_equal(r.solver_status, [0, 0, 1, 1])
        self.assertLess(np.max(r.diagnostics["equilibrium_residuals"]), 1e-8)

    def test_precipitated_sulfate_inventory(self):
        case = read("cases", "coupled-calcium")
        p = AIOMFACModel.prepare(spec(case["components"]))
        r = p.evaluate(
            case["fractions"], case["temperature_K"], diagnostics=True
        )
        solid = r.diagnostics["precipitated_caso4_mol_per_kg_input"]
        masses = dict(zip(p.salt_indices, p.salt_mw, strict=True))
        np.testing.assert_allclose(
            solid, case["fractions"][3] / masses[3], rtol=1e-12
        )
        sulfur = sum(
            r.species_moles_per_kg_input[
                len(r.neutral_ids) + r.ion_ids.index(j)
            ]
            for j in (248, 261)
        )
        np.testing.assert_allclose(
            sulfur + solid, case["fractions"][1] / masses[1], rtol=1e-10
        )

    def test_exhaustion(self):
        case = read("cases", "coupled-carbonate-sulfate")
        model = AIOMFACModel.prepare(spec(case["components"]))
        rows = np.array([[1.0, 0.0, 0.0], case["fractions"]])
        with patch(
            "particula_aiomfac._equilibrium.root",
            return_value=SimpleNamespace(x=np.zeros(6)),
        ) as solver:
            with self.assertRaisesRegex(
                ConvergenceError, r"batch index \(1,\)"
            ):
                model.evaluate(rows, 298.0)
            self.assertEqual(solver.call_count, 5)
        np.testing.assert_array_equal(rows[1], case["fractions"])

    @unittest.skipUnless(
        os.getenv("AIOMFAC_TEST_CUDA") == "1", "requires NVIDIA device"
    )
    def test_device_array_status(self):
        import warp as wp

        case = read("cases", "sulfate")
        m = AIOMFACModel.prepare(
            spec(case["components"]), execution="cuda", device="cuda:0"
        )
        stream = wp.Stream(m.device)
        with wp.ScopedStream(stream):
            data = wp.array(
                [case["fractions"], [-1.0, 1.0], case["fractions"]],
                dtype=wp.float64,
                device=m.device,
            )
            ts = wp.array([298.15] * 3, dtype=wp.float64, device=m.device)
            out = wp.empty(
                (3, m.output_width), dtype=wp.float64, device=m.device
            )
            flags = wp.empty(3, dtype=wp.int32, device=m.device)
            scratch = wp.empty(
                (3, m.work_width), dtype=wp.float64, device=m.device
            )
            m.evaluate_into(data, ts, out, flags, scratch, stream=stream)
            np.testing.assert_array_equal(flags.numpy(), [1, -1, 1])
            np.testing.assert_allclose(
                out.numpy()[[0, 2], m.output_slices["activity"]],
                0.9602314418143285,
                atol=1e-10,
                rtol=0,
            )
