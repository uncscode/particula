"""Compare the full respiratory illustration at two numerical resolutions.

Run from the repository root with PYTHONPATH=. and --output REPORT.json.
The existing example supplies every physical input and the coarse solver.
This is same-method numerical consistency, not an independent physical oracle.
No particle states are written: the report retains compact observable histories.
"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import matplotlib
import numpy as np
from particula.execution.condensation_ode import CondensationODE
from scripts.docs_examples import GIT, package_source_hash, package_versions

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "docs/examples/simulations/cough_droplets_partitioning.py"
# Prospective consistency targets, not bounds on global or physical error.
TARGETS = {"radius_relative": 1e-3, "phase_inventory_fraction": 1e-3}


def load_setup(source, filename):
    """Execute the example through solver construction, before its time loop."""
    tree = ast.parse(source, filename=filename)
    constructors = [
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "simulation"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "CondensationODE"
    ]
    if len(constructors) != 1:
        raise ValueError("expected one explicit published condensation solver")
    index = constructors[0]
    constructor = tree.body[index].value
    namespace = {"__name__": "respiratory_refinement_setup"}
    matplotlib.use("Agg")
    # The setup's initial histogram has no numerical role. Retain its code,
    # but omit GUI display in this deliberately headless assessment.
    body = [
        node
        for node in tree.body[: index + 1]
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "plt"
            and node.value.func.attr == "show"
        )
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(  # noqa: S102 -- execute this repository's maintained example
            compile(ast.Module(body, []), filename, "exec"),
            namespace,
        )
    namespace["plt"].close("all")
    controls = {
        keyword.arg: eval(  # noqa: S307 -- expressions from that same executed source
            compile(ast.Expression(keyword.value), filename, "eval"), namespace
        )
        for keyword in constructor.keywords
    }
    if controls["method"] != "ros3":
        raise ValueError("this assessment qualifies the ROS3 illustration")
    initial = namespace["aerosol"]
    if initial.particles.masses.shape != (1, 1000, 12):
        raise ValueError(
            "the complete 1000-particle, 12-material case is required"
        )
    np.testing.assert_array_equal(
        namespace["time_array"], np.linspace(0, 20, 201)
    )
    return namespace, controls


def inventory(state):
    """Return particle and gas masses [kg] separately for each species."""
    particles = np.sum(
        state.particles.masses * state.particles.concentration[..., None],
        axis=1,
    )
    gas = state.gas.concentration * state.particles.volume[:, None]
    return particles, gas


def compare(initial, coarse, fine):
    """Use each lane's own scale so water cannot hide a small solute."""
    initial_particle, initial_gas = inventory(initial)
    total = initial_particle + initial_gas
    if np.any(total <= 0) or np.any(initial.particles.masses <= 0):
        raise ValueError(
            "this case requires positive initial material inventories"
        )
    coarse_particle, coarse_gas = inventory(coarse)
    fine_particle, fine_gas = inventory(fine)
    for state, particle, gas in (
        (coarse, coarse_particle, coarse_gas),
        (fine, fine_particle, fine_gas),
    ):
        np.testing.assert_allclose(
            particle + gas, total, rtol=1e-12, atol=1e-30
        )
        np.testing.assert_array_equal(
            state.particles.concentration, initial.particles.concentration
        )
    radii = fine.particles.radii
    if np.any(radii <= 0):
        raise ValueError("a retained droplet has no positive radius")
    return {
        "radius_relative": float(
            np.max(abs(coarse.particles.radii / radii - 1))
        ),
        "particle_phase_inventory_fraction": (
            abs(coarse_particle - fine_particle) / total
        )[0].tolist(),
        "gas_phase_inventory_fraction": (abs(coarse_gas - fine_gas) / total)[
            0
        ].tolist(),
        "slot_material_initial_relative": np.max(
            abs(coarse.particles.masses - fine.particles.masses)
            / initial.particles.masses,
            axis=(0, 1),
        ).tolist(),
        "coarse_radius_quantiles_m": np.quantile(
            coarse.particles.radii, [0.1, 0.5, 0.9]
        ).tolist(),
        "fine_radius_quantiles_m": np.quantile(radii, [0.1, 0.5, 0.9]).tolist(),
        "coarse_particle_mass_kg": coarse_particle[0].tolist(),
        "fine_particle_mass_kg": fine_particle[0].tolist(),
        "coarse_budget_relative": (
            abs(coarse_particle + coarse_gas - total) / total
        )[0].tolist(),
        "fine_budget_relative": (abs(fine_particle + fine_gas - total) / total)[
            0
        ].tolist(),
    }


def advance(owner, bound, label):
    """Advance to a physical boundary, printing bounded live progress."""
    reported = monotonic()
    while owner.time < bound:
        owner.step(bound)
        if monotonic() - reported >= 30:
            print(
                label, owner.time, asdict(owner.solver.statistics), flush=True
            )
            reported = monotonic()


def write_json(path, value):
    """Replace a compact report only after complete JSON serialization."""
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def assess(output):
    """Run both full trajectories and retain all 201 comparison observations."""
    started = monotonic()
    source_path = ROOT / SOURCE
    source = source_path.read_text()
    package_hash = package_source_hash(ROOT)
    namespace, controls = load_setup(source, SOURCE)
    initial = namespace["aerosol"]
    coarse = namespace["simulation"]
    fine_controls = dict(controls)
    for name in ("rtol", "particle_atol", "gas_atol"):
        fine_controls[name] /= 100
    fine_controls["max_step"] /= 2
    fine = CondensationODE(
        initial, namespace["condensation_strategy"], **fine_controls
    )
    report = {
        "schema_version": 1,
        "status": "running",
        "interpretation": (
            "same-method numerical consistency, not independent validation"
        ),
        "source": SOURCE,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "package_source_sha256": package_hash,
        "assessment_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "revision": subprocess.check_output(  # noqa: S603 -- fixed git arguments
            [GIT, "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "versions": package_versions(ROOT),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "requested_material_names": namespace["list_of_chemicals"],
        "resolved_material_names": namespace["short_name"],
        "initial_material_inventory_kg": sum(inventory(initial))[0].tolist(),
        "targets": TARGETS,
        "coarse_controls": controls,
        "fine_controls": fine_controls,
        "coarse_output_interval_s": float(namespace["time_step"]),
        "fine_output_interval_s": float(namespace["time_step"] / 2),
        "observations": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.with_suffix(".progress.json")
    write_json(progress, report)
    previous = 0.0
    for time in namespace["time_array"]:
        bound = float(time)
        advance(coarse, bound, "coarse")
        if bound > previous:
            advance(fine, (previous + bound) / 2, "fine")
        advance(fine, bound, "fine")
        row = compare(initial, coarse.snapshot(), fine.snapshot())
        report["observations"].append({"time_s": bound, **row})
        report["work"] = {
            "coarse": asdict(coarse.solver.statistics),
            "fine": asdict(fine.solver.statistics),
        }
        report["elapsed_seconds"] = monotonic() - started
        write_json(progress, report)
        print(
            f"Compared t={bound:.2f} s; "
            f"radius difference={row['radius_relative']:.6g}",
            flush=True,
        )
        previous = bound
    if (
        source_path.read_text() != source
        or package_source_hash(ROOT) != package_hash
    ):
        raise RuntimeError(
            "source changed during assessment; refusing final report"
        )
    report["max_radius_relative"] = max(
        row["radius_relative"] for row in report["observations"]
    )
    report["max_phase_inventory_fraction"] = max(
        max(row[key])
        for row in report["observations"]
        for key in (
            "particle_phase_inventory_fraction",
            "gas_phase_inventory_fraction",
        )
    )
    report["meets_consistency_targets"] = (
        report["max_radius_relative"] < TARGETS["radius_relative"]
        and report["max_phase_inventory_fraction"]
        < TARGETS["phase_inventory_fraction"]
    )
    report["status"] = "complete"
    write_json(output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "max_radius_relative",
                    "max_phase_inventory_fraction",
                    "meets_consistency_targets",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    assess(parser.parse_args().output)
