"""Independent source-rate, binding and analytical tests for NCAR import."""

import json

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]
from particula import ProcessRegistry
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.adapter import GasChemistryAdapter
from particula.dynamics.chemistry.ncar import convert_ncar
from particula.dynamics.chemistry.process import ChemistrySolver, GasChemistry
from particula.dynamics.chemistry.tests.process_test import state_and_process
from scipy.constants import Boltzmann


def source(order):
    """Build the two independent source documents without native helpers."""
    names = ["C", "B", "A"]
    masses = [0.03, 0.02, 0.01] if order == 2 else [0.01, 0.02, 0.01]
    reactants = ["A", "B"] if order == 2 else ["A"]
    return {
        "version": "1.3.0",
        "name": "test",
        "species": [
            {"name": n, "molecular weight [kg mol-1]": m}
            for n, m in zip(names, masses, strict=True)
        ],
        "phases": [{"name": "g", "species": [{"name": n} for n in names]}],
        "reactions": [
            {
                "name": "convert",
                "type": "ARRHENIUS",
                "gas phase": "g",
                "reactants": [{"species name": n} for n in reactants],
                "products": [{"species name": "C"}],
                "A": 1.8,
                "B": 0.7,
                "D": 310,
                **({"C": -145} if order == 2 else {"Ea": 2e-21}),
            }
        ],
    }


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda:0"])
def test_imported_trajectory(tmp_path, order, device):
    """Compare imported rates and inventories with two-temperature solutions."""
    import warp as wp

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    record = source(order)
    if order == 1:
        path = tmp_path / "source.json"
        path.write_text(json.dumps(record))
    else:
        path = tmp_path / "source.yaml"
        path.write_text(yaml.safe_dump(record))
    target = tmp_path / "native.json"
    imported = convert_ncar(path, target)
    state, _ = state_and_process("arrhenius", "rk23")
    gas = state.gas
    gas.name[:] = ["inert", "C", "A", "B"]
    gas.molar_mass[:] = [0.04, 0.03 if order == 2 else 0.01, 0.01, 0.02]
    initial = np.array([1.0, 2.0])
    gas.concentration[:] = (
        np.column_stack((initial / 100, initial * 0, initial, initial))
        * gas.molar_mass
    )
    original = gas.concentration.copy()
    bound = ReactionMechanism.load(target).bind(gas)
    manual = ReactionMechanism.from_dict(
        {
            "version": 1,
            "basis": "mol/m^3-air",
            "species": {f"gas:{n}": {} for n in "ABC"},
            "reactions": [
                {
                    "id": "manual",
                    "reactants": {
                        "gas:A": 1,
                        **({"gas:B": 1} if order == 2 else {}),
                    },
                    "products": {"gas:C": 1},
                    "rate": {
                        "type": "arrhenius",
                        "A": 1.8,
                        "b": 0.7,
                        "T_ref_K": 310,
                        "Ea_J_per_mol": 145 * 8.31446261815324
                        if order == 2
                        else 2e-21 * 6.02214076e23,
                        "units": "m^3/mol/s" if order == 2 else "1/s",
                    },
                }
            ],
        }
    ).bind(gas)
    temperatures = state.environment.temperature
    k = (
        1.8
        * (temperatures / 310) ** 0.7
        * np.exp((-145 if order == 2 else -2e-21 / Boltzmann) / temperatures)
    )
    for temperature, rate in zip(temperatures, k, strict=True):
        for mechanism in (manual, bound):
            np.testing.assert_allclose(
                mechanism.kinetics(temperature).coefficients,
                rate,
                rtol=1e-12,
                atol=0,
            )
    process = GasChemistry(bound, ChemistrySolver(rtol=1e-11, atol=1e-16))
    if device == "numpy":
        process.execute(state, 1.0)
        result = state.gas.concentration
    else:
        registry = ProcessRegistry()
        registry.register(GasChemistry, GasChemistryAdapter())
        with process.prepare(
            state, device=device, time_step=1.0, process_registry=registry
        ) as owner:
            owner.step()
            result = owner.snapshot().gas.concentration
    final = initial / (1 + k * initial) if order == 2 else initial * np.exp(-k)
    expected = (
        np.column_stack(
            (
                initial / 100,
                initial - final,
                final,
                final if order == 2 else initial,
            )
        )
        * gas.molar_mass
    )
    np.testing.assert_allclose(result, expected, rtol=2e-9, atol=1e-20)
    np.testing.assert_array_equal(result[:, 0], original[:, 0])
    volumes = state.particles.volume
    np.testing.assert_allclose(
        result.sum(axis=1) * volumes,
        original.sum(axis=1) * volumes,
        rtol=1e-12,
        atol=1e-22,
    )
    gas.molar_mass[1] *= 2
    with pytest.raises(ValueError, match="gas:C.*molar mass"):
        imported.bind(gas)


@pytest.mark.parametrize(
    ("location", "change", "match"),
    [
        ("root", {"version": "1.4.0"}, "version"),
        (
            "species",
            {"constant mixing ratio [mol mol-1]": 0.1},
            "species 'C'.*mixing ratio",
        ),
        ("reaction", {"E": 0.1}, "convert.*pressure"),
        ("reaction", {"Ea": 1e-21}, "convert.*mutually exclusive"),
        ("reaction", {"type": "TROE"}, "convert.*ARRHENIUS"),
        ("reaction", {"unexpected": 1}, "convert.*unexpected"),
        (
            "reaction",
            {"reactants": [{"species name": "unknown"}]},
            "convert.*unknown species",
        ),
        (
            "reaction",
            {"products": [{"species name": "C", "coefficient": 1.3}]},
            "convert.*coefficient",
        ),
    ],
)
def test_unsupported_preserves_output(tmp_path, location, change, match):
    """Report the source field without changing an existing destination."""
    record = source(2)
    section = {
        "root": record,
        "species": record["species"][0],
        "reaction": record["reactions"][0],
    }[location]
    section.update(change)
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps(record))
    target = tmp_path / "output.yaml"
    target.write_text("existing output")
    with pytest.raises(ValueError, match=match):
        convert_ncar(input_file, target)
    assert target.read_text() == "existing output"
