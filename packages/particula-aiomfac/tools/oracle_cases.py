"""Run explicit composition cases and decode the reference driver's tagged rows."""

import json
import math
import re
import subprocess
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from oracle_build import BUILD, CACHE, REFERENCE, SOURCE


def read(path):
    return json.loads(path.read_text())


def input_text(points):
    case = points[0]
    groups = read(REFERENCE / "cases/components.json")
    lines = ["Input file for AIOMFAC-web model", "", "mixture components:", "----"]
    for index, name in enumerate(case["components"], 1):
        lines.extend([f"component no.: {index}", f"component name: '{name}'"])
        for group, count in groups[name].items():
            lines.append(f"subgroup no., qty: {group}, {count}")
        lines.append("----")
    mole = int(case["basis"] == "mole")
    lines += [
        "@@@@",
        "calculation options:",
        "smiles-based pure-component method? 0",
        "----",
        "++++",
        "mixture composition and temperature:",
        f"mass fraction? {1 - mole}",
        f"mole fraction? {mole}",
        "----",
        "point, T_K, "
        + ", ".join(f"cp{i:02}" for i in range(2, len(case["components"]) + 1)),
    ]
    for i, state in enumerate(points, 1):
        if state["components"] != case["components"] or state["basis"] != case["basis"]:
            raise ValueError("inconsistent species order or input basis")
        fractions = state["fractions"]
        if (
            case["basis"] not in ("mass", "mole")
            or len(fractions) != len(case["components"])
            or not all(math.isfinite(x) and x >= 0 for x in fractions)
            or not math.isclose(sum(fractions), 1, abs_tol=1e-14, rel_tol=0)
            or not math.isfinite(state["temperature_K"])
            or state["temperature_K"] <= 0
        ):
            raise ValueError("invalid_input")
        lines.append(", ".join(map(str, [i, state["temperature_K"], *fractions[1:]])))
    return "\n".join([*lines, "====", ""])


def execute(points, name="activity", original=False):
    data = input_text(points)
    with TemporaryDirectory(dir=CACHE) as tmp:
        root = Path(tmp)
        (root / "Inputfiles").mkdir()
        (root / "Outputfiles").mkdir()
        file = root / "Inputfiles/input_0008.txt"
        if original:
            data = (SOURCE / "Examples_Input/input_0008.txt").read_text()
            # Keep original component definitions and row 1 without later points.
            data = data[: re.search(r"(?m)^2\s+298", data).start()] + "====\n"
            (root / "Auxiliary").mkdir()
            table = "Auxiliary/Pure_component_smiles_table.csv"
            (root / table).write_bytes((SOURCE / table).read_bytes())
        file.write_text(data)
        result = subprocess.run(
            [BUILD / name, "./Inputfiles/input_0008.txt"],
            cwd=root,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=45,
        )
        output = root / "Outputfiles/AIOMFAC_output_0008.txt"
        code = re.search(r"final error indicator:\s*(\d+)", result.stdout)
        if not result.returncode and code and int(code[1]) and not output.exists():
            return result.stdout, ""
        if result.returncode or not output.exists():
            log = root / "Outputfiles/Errorlog_0008.txt"
            raise RuntimeError(
                result.stderr
                + result.stdout
                + (log.read_text() if log.exists() else "")
            )
        return result.stdout, output.read_text()


def fields(names, values):
    return dict(zip(names.split(), map(float, values), strict=True))


def parse(stdout, case):
    groups = read(REFERENCE / "cases/components.json")
    neutrals = [
        name for name in case["components"] if all(int(g) < 200 for g in groups[name])
    ]
    # SetSystem appends implicit aqueous CO2 to the neutral block.
    if "CO2(aq)" not in neutrals and any(
        "250" in groups[n] or "262" in groups[n] for n in case["components"]
    ):
        neutrals.append("CO2(aq)")
    points = []
    for line in stdout.splitlines():
        row = line.split()
        if row[:1] != ["REF"]:
            continue
        kind, *v = row[1:]
        if kind == "point":
            state = {
                "neutrals": {},
                "ions": {},
                "salts": {},
                "species": {},
                "equilibrium": {},
            }
            points.append(state)
        elif kind == "scale":
            state.update(
                molality_conversion=float(v[0]), mixed_solvent_reference=v[1] == "T"
            )
        elif kind == "neutral":
            key = neutrals[int(v[0]) - 1]
            item = fields("x ln_gamma activity sr mr lr mass_fraction", v[1:])
            item["defined"] = item["mass_fraction"] > 0
            if not item["defined"]:
                item["ln_gamma"] = None
            state["neutrals"][key] = item
        elif kind == "ion":
            item = fields("charge molality ln_gamma sr mr lr", v[1:])
            item["defined"] = item["molality"] > 0
            if not item["defined"]:
                item["ln_gamma"] = None
            state["ions"][v[0]] = item
        elif kind == "salt":
            item = fields("cation_count anion_count gamma_mean iap", v[2:])
            item["defined"] = (
                state["ions"][v[0]]["defined"] and state["ions"][v[1]]["defined"]
            )
            if not item["defined"]:
                item["gamma_mean"] = None
            state["salts"]["/".join(v[:2])] = item
        elif kind == "species":
            ion = int(float(v[-1]))
            key = str(ion) if ion else neutrals[int(v[0]) - 1]
            state["species"][key] = fields(
                "mass_fraction x molality gamma activity", v[1:-1]
            )
        elif kind == "status":
            warning = int(v[0])
            state.update(
                warning=warning,
                errors=list(map(int, v[1:])),
                status="upstream_error"
                if v[1:]
                else "ok"
                if warning in (0, 16)
                else f"warning_{warning}",
            )
        elif kind == "sulfate":
            state["equilibrium"]["sulfate"] = fields(
                "log_residual input_sulfur final_sulfur", v
            )
        elif kind == "carbonate":
            state["equilibrium"]["carbonate"] = {
                "minpack_info": int(v[0]),
                "residual_vector": list(map(float, v[1:])),
            }
        elif kind == "sulfur":
            state["equilibrium"]["sulfur"] = fields("input_sulfur final_sulfur", v)
        elif kind == "carbon":
            state["equilibrium"]["carbon"] = fields("input_carbon final_carbon", v)
        else:
            raise ValueError(f"unknown reference field {kind}")
    for state in points:
        ions = list(state["ions"].values())
        charges = [i["charge"] * i["molality"] for i in ions]
        state["ionic_strength_mol_per_kg"] = 0.5 * math.fsum(
            i["charge"] ** 2 * i["molality"] for i in ions
        )
        state["relative_charge_residual"] = (
            math.fsum(charges) / math.fsum(map(abs, charges)) if any(charges) else 0.0
        )
        for element, system in [("sulfur", "sulfate"), ("sulfur", "sulfur"), ("carbon", "carbon")]:
            if system in state["equilibrium"]:
                record = state["equilibrium"][system]
                initial = record["input_" + element]
                record["relative_balance"] = (
                    record["final_" + element] - initial
                ) / initial
    return points


def run_case(case):
    try:
        stdout, _ = execute([case])
    except ValueError as exc:
        if str(exc) != "invalid_input":
            raise
        return {"status": "invalid_input"}
    if "REF point " not in stdout:
        code = int(re.search(r"final error indicator:\s*(\d+)", stdout)[1])
        return {"status": f"upstream_error_{code}", "mixture_error": code}
    [result] = parse(stdout, case)
    return result


def crosscheck(case):
    direct = run_case(case)
    _, printed = execute([case], "standard", original=True)
    rows = [
        line.split()
        for line in printed.splitlines()
        if re.match(r"^\s*001\s+298\.00", line) and len(line.split()) == 9
    ]
    if len(rows) != len(direct["species"]):
        raise ValueError("standard executable did not report all mixed-case species")
    for (name, species), row in zip(direct["species"].items(), rows, strict=True):
        for key, token in zip(species, row[3:8], strict=True):
            rounded = Decimal(token)
            half_unit = float(Decimal(5).scaleb(rounded.as_tuple().exponent - 1))
            if abs(species[key] - float(rounded)) > half_unit + 2 * math.ulp(
                species[key]
            ):
                raise ValueError(
                    f"standard CLI mismatch: {name}.{key}: {token} vs {species[key]}"
                )
    for name, item in direct["neutrals"].items():
        if not math.isclose(
            item["activity"], item["x"] * math.exp(item["ln_gamma"]), rel_tol=1e-12
        ):
            raise ValueError(f"neutral activity identity failed: {name}")
    ions = direct["ions"]
    mean = math.exp((ions["202"]["ln_gamma"] + ions["242"]["ln_gamma"]) / 2)
    if not math.isclose(mean, direct["salts"]["202/242"]["gamma_mean"], rel_tol=1e-12):
        raise ValueError("NaCl mean coefficient identity failed")
    return direct
