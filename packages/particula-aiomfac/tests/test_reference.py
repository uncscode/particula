"""Cheap, compiler-independent fixture contracts for the development release."""

import hashlib
import json
import tomllib
import unittest
from pathlib import Path

from particula_aiomfac import UPSTREAM_REVISION, UPSTREAM_VERSION, __version__

REF = Path(__file__).resolve().parents[1] / "reference"


class TestFixtures(unittest.TestCase):
    def test_identity(self):
        pin = tomllib.loads((REF / "upstream.toml").read_text())
        self.assertRegex(__version__, r"^(0|[1-9]\d*)(\.(0|[1-9]\d*)){2}$")
        self.assertEqual(
            (UPSTREAM_REVISION, UPSTREAM_VERSION),
            (pin["revision"], pin["version"]),
        )

    def test_inventory(self):
        meta = json.loads((REF / "expected/generation.json").read_text())
        self.assertEqual(meta["revision"], UPSTREAM_REVISION)
        cases = set((REF / "cases").glob("*.json"))
        self.assertEqual({p.name for p in cases}, set(meta["case_hashes"]))
        for p in cases:
            self.assertEqual(
                hashlib.sha256(p.read_bytes()).hexdigest(), meta["case_hashes"][p.name]
            )
        ids = {p.name for p in cases if p.stem != "components"}
        self.assertEqual(
            ids,
            {
                p.name
                for p in (REF / "expected").glob("*.json")
                if p.stem != "generation"
            },
        )
        for name in ids:
            case = json.loads((REF / "cases" / name).read_text())
            state = json.loads((REF / "expected" / name).read_text())
            self.assertEqual(case["expected_status"], state["status"], name)

    def test_scientific_limits(self):
        def result(name):
            return json.loads((REF / "expected" / (name + ".json")).read_text())

        water = result("water-limit")["neutrals"]
        self.assertEqual(water["water"]["activity"], 1.0)
        self.assertIsNone(water["butanediol"]["ln_gamma"])
        self.assertGreater(
            result("nacl-dilute")["neutrals"]["water"]["activity"],
            result("nacl")["neutrals"]["water"]["activity"],
        )
        self.assertEqual(
            result("carbonate")["equilibrium"]["carbonate"]["minpack_info"], 1
        )

    def test_source_inventory(self):
        pin = tomllib.loads((REF / "upstream.toml").read_text())
        meta = json.loads((REF / "expected/generation.json").read_text())
        self.assertEqual(
            meta["source_manifest_sha256"],
            hashlib.sha256((REF / "upstream.toml").read_bytes()).hexdigest(),
        )
        for path in ["ModCalcActCoeff.f90", "Mod_MINPACK.f90", "maketemplate.txt"]:
            self.assertIn("FortranCode/" + path, pin["sha256"])

    def test_salt_basis(self):
        import math

        mixed = json.loads((REF / "expected/mixed.json").read_text())
        cation, anion = [mixed["ions"][key] for key in ("202", "242")]
        self.assertAlmostEqual(cation["molality"], anion["molality"], places=12)
        mean = math.exp((cation["ln_gamma"] + anion["ln_gamma"]) / 2)
        salt = mixed["salts"]["202/242"]
        self.assertAlmostEqual(salt["gamma_mean"], mean, places=12)
        self.assertAlmostEqual(
            salt["iap"], mean**2 * cation["molality"] * anion["molality"], places=12
        )
