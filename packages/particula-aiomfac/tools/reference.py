"""Build and qualify the AIOMFAC developer reference. Run --help for commands."""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import oracle_build as builder
from oracle_build import BUILD, CACHE, REFERENCE
from oracle_cases import crosscheck, execute, parse, read, run_case


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def differences(expected, actual, key):
    """Compare the full structure, with quantity-specific numerical tolerances."""
    if type(expected) is not type(actual):
        yield f"{key}: {expected!r} vs {actual!r}"
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            yield f"{key}: different fields {expected.keys() ^ actual.keys()}"
        for field in expected.keys() & actual.keys():
            yield from differences(expected[field], actual[field], f"{key}.{field}")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            yield f"{key}: different lengths"
        for i, (left, right) in enumerate(zip(expected, actual)):
            yield from differences(left, right, f"{key}[{i}]")
    elif isinstance(expected, float):
        log = key.endswith((".ln_gamma", ".sr", ".mr", ".lr"))
        if not math.isclose(
            expected, actual, rel_tol=0 if log else 1e-8, abs_tol=1e-8 if log else 1e-10
        ):
            yield f"{key}: expected {expected:.17g}, got {actual:.17g}, difference {actual - expected:.4g}"
    elif expected != actual:
        yield f"{key}: expected {expected!r}, got {actual!r}"


def scientific_checks(name, case, state):
    if state["status"] != case["expected_status"]:
        raise ValueError(
            f"{name}: {state['status']} instead of {case['expected_status']}"
        )
    if "expected_solver_error" in case:
        solver=state["equilibrium"]["carbonate"]
        if case["expected_solver_error"] not in state["errors"] or solver["minpack_info"] == 1:
            raise ValueError(f"{name}: expected Fortran convergence failure, got {state['errors']}")
        if max(map(abs, solver["residual_vector"])) <= 1e-8:
            raise ValueError(f"{name}: expected a measurable equilibrium discrepancy")
        return
    if "species" not in state:
        return
    if state["mixed_solvent_reference"]:
        raise ValueError(f"{name}: wrong reference convention")
    if abs(state["relative_charge_residual"]) > 1e-10:
        raise ValueError(f"{name}: charge balance {state['relative_charge_residual']}")
    for key, record in state["equilibrium"].items():
        if abs(record.get("relative_balance", 0)) > 1e-10:
            raise ValueError(f"{name}: {key} balance {record}")
        if abs(record.get("log_residual", 0)) > 1e-8 or any(
            abs(x) > 1e-8 for x in record.get("residual_vector", [])
        ):
            raise ValueError(f"{name}: equilibrium residual {record}")
        if key == "carbonate" and record["minpack_info"] != 1:
            raise ValueError(f"{name}: MINPACK status {record}")


def cases():
    return {
        p.stem: read(p)
        for p in sorted((REFERENCE / "cases").glob("*.json"))
        if p.stem != "components"
    }


def check_build():
    builder.fetch(offline=True)
    meta = read(BUILD / "generation.json")
    if meta["generator_sha256"] != builder.sha(Path(builder.__file__)) or meta[
        "source_manifest_sha256"
    ] != builder.sha(REFERENCE / "upstream.toml"):
        raise ValueError("source manifest or driver changed: rebuild the reference")
    for name in ("standard", "activity"):
        if builder.sha(BUILD / name) != meta["executables"][name]:
            raise ValueError(f"{name} binary changed: rebuild the reference")
    return meta


def expected_state(name, compiler):
    """Select recorded diagnostics for the failed trace solve by toolchain."""
    path = REFERENCE / "expected" / (name + ".json")
    state = read(path)
    if name != "coupled-trace-bound":
        return state
    observation = read(REFERENCE / "compiler-observations" / path.name)
    if compiler != observation["compiler"]:
        return state
    if builder.sha(path) != observation["canonical_sha256"]:
        raise ValueError("Trace fixture changed: review the compiler observation")
    equilibrium = state["equilibrium"]
    equilibrium["carbon"]["relative_balance"] = observation["carbon_relative_balance"]
    for index, value in observation["carbonate_residuals"].items():
        equilibrium["carbonate"]["residual_vector"][int(index)] = value
    return state


def qualify(update=False):
    meta = check_build()
    inputs = cases()
    data = {}
    # This comparison reads the original upstream input rather than the case writer.
    mixed = crosscheck(inputs["mixed"])
    out = CACHE / "candidates"
    for name, case in inputs.items():
        state = mixed if name == "mixed" else run_case(case)
        data[name] = state
        save(out / (name + ".json"), state)
        scientific_checks(name, case, state)

    errors = list(differences(mixed, data["mixed-reordered"], "mixed-reordered"))
    errors.extend(
        differences(data["carbonate"], data["carbonate-implicit"], "implicit_CO2")
    )
    # Repeat both carbonate compositions after different predecessor states.
    for names in (
        ["carbonate", "carbonate-rich", "carbonate"],
        ["carbonate-rich", "carbonate", "carbonate-rich"],
    ):
        points = [inputs[name] for name in names]
        stdout, _ = execute(points)
        for name, state in zip(names, parse(stdout, points[0]), strict=True):
            errors.extend(differences(data[name], state, f"history.{name}"))
    if errors:
        raise ValueError("\n".join(errors))

    stable = {
        k: meta[k]
        for k in (
            "revision",
            "flags",
            "working_precision",
            "generator_sha256",
            "source_manifest_sha256",
        )
    }
    stable["case_hashes"] = {
        p.name: builder.sha(p) for p in sorted((REFERENCE / "cases").glob("*.json"))
    }
    stable["parameter_manifest_sha256"] = builder.sha(REFERENCE / "parameters.toml")
    stable["reader_sha256"] = builder.sha(Path(__file__).with_name("oracle_cases.py"))
    expected = REFERENCE / "expected"
    if update:
        for name in data:
            shutil.copyfile(out / (name + ".json"), expected / (name + ".json"))
        save(expected / "generation.json", {**stable, "compiler": meta["compiler"]})
    else:
        saved = read(expected / "generation.json")
        saved.pop("compiler")
        errors.extend(differences(saved, stable, "generation"))
        for name, state in data.items():
            errors.extend(differences(expected_state(name, meta["compiler"]), state, name))
        if errors:
            raise ValueError("\n".join(errors))
    print(
        f"Passed original CLI, identities, species permutation, history and {len(data)} cases."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch").add_argument("--offline", action="store_true")
    commands.add_parser("build")
    commands.add_parser("run").add_argument("--case", required=True, choices=cases())
    commands.add_parser("verify")
    commands.add_parser(
        "update", help="replace tracked expectations after scientific checks"
    )
    args = parser.parse_args()
    if args.command == "fetch":
        builder.fetch(args.offline)
    elif args.command == "build":
        builder.build()
    elif args.command == "run":
        check_build()
        print(json.dumps(run_case(cases()[args.case]), indent=2, allow_nan=False))
    else:
        qualify(update=args.command == "update")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        sys.exit(str(error))
