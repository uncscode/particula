# %% Independent native adaptive integration of the fitted chemical channels
"""Audit fixed-forcing rate units and the differentiable sodium-state reduction."""

import argparse
import time
import numpy as np
import pandas as pd
from particula import Aerosol, ProcessRegistry
from particula.particles import ParticleData
from particula.gas import GasData, EnvironmentData
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.resolved import ResolvedChemistry
from particula.dynamics.chemistry.process import ChemistrySolver
from particula.dynamics.chemistry.resolved_adapter import ResolvedChemistryAdapter
from pathways import CANDIDATES, HERE, NAMES, MASSES, FORMULA, catalog


def execute(name, p, d, device):
    bank = catalog()["mechanisms"]
    doc = bank["+".join(sorted(CANDIDATES[name]))]
    n0 = 1.0e-17
    initial = np.array([1, 0, 0, 0, 0, 2, 2, 2, 0], dtype=float) * n0
    mass = initial * MASSES
    # An equal-density closed box keeps material volume constant when reaction mass is conserved.
    # This enforces the normalized-inventory convention used for the second-order fitted rate.
    density = np.full(len(NAMES), 1800.0)
    na_concentration = n0 / np.sum(mass / density)
    size_factor = np.exp(
        -p.small_size_power * max(np.log(200 / d), 0)
        - p.large_size_power * max(np.log(d / 200), 0)
    )
    k = p.peak_k_h * size_factor
    rates = {
        "direct": (1 - p.capture_branch) * k,
        "capture": p.capture_branch * k,
        "oxidize": p.oxidation_or_gate2_h,
        "expose": k * (p.capture_branch if "direct" in CANDIDATES[name] else 1.0),
        "react_exposed": p.oxidation_or_gate2_h,
        "neutralize": p.neutralization_h / na_concentration,
    }
    for rx in doc["reactions"]:
        label = rx["id"]
        # An externally buffered reactant has zero kinetic order; retain its mass/element debit.
        rx["orders"] = {
            key: (0 if key.split(":")[1] in ("H2SO4", "H2SO3", "Oeq") else value)
            for key, value in rx["reactants"].items()
        }
        rx["rate"] = {
            "type": "constant",
            "k": float(rates[label] / 3600),
            "units": "m^3/mol/s" if label == "neutralize" else "1/s",
        }
    state = Aerosol(
        particles=ParticleData(
            mass[None, None, :],
            np.ones((1, 1)),
            np.zeros((1, 1)),
            density,
            np.ones(1),
            distribution_type="particle_resolved",
        ),
        gas=GasData(["inert"], np.array([0.028]), np.zeros((1, 1)), np.array([False])),
        environment=EnvironmentData(
            temperature=np.array([298.15]),
            pressure=np.array([101325.0]),
            saturation_ratio=np.zeros((1, 1)),
        ),
    )
    mech = ReactionMechanism.from_dict(doc)
    bound = mech.bind_resolved(state, {"particle:" + n: i for i, n in enumerate(NAMES)})
    process = ResolvedChemistry(
        bound,
        ChemistrySolver(
            rtol=1e-7, atol=1e-7, first_step=2.0, max_step=180.0, max_trials=1200
        ),
    )
    reg = ProcessRegistry()
    reg.register(ResolvedChemistry, ResolvedChemistryAdapter())
    answer = []
    with process.prepare(
        state, device=device, time_step=1800.0, process_registry=reg
    ) as sim:
        for sample in range(7):
            sim.step()
            mol = sim.snapshot().particles.masses[0, 0] / MASSES / n0
            assert np.max(np.abs((mol - initial / n0) @ FORMULA)) < 1e-9
            answer.append(mol[:5])
    return np.array(answer)


def main(args):
    tic = time.monotonic()
    out = HERE / "native-check"
    out.mkdir(exist_ok=True)
    pars = pd.concat(
        [pd.read_csv(HERE / tag / "scores.csv") for tag in ("primary", "retention")]
    ).set_index("network")
    truth = pd.concat(
        [
            pd.read_csv(HERE / tag / "inventories.csv")
            for tag in ("primary", "retention")
        ]
    )
    rows = []
    for name in args.names.split(","):
        for d in map(int, args.diameters.split(",")):
            x = execute(name, pars.loc[name], d, args.device)
            z = truth.query("network==@name and diameter_nm==@d and time_h<=3.5")[
                [
                    "nacl",
                    "exposed_nacl",
                    "bisulfite",
                    "bisulfate_inventory",
                    "neutral_salt_inventory",
                ]
            ].to_numpy()
            error = abs(z - x).max()
            print(name, d, error, flush=True)
            assert error < 0.001
            rows.append(
                dict(
                    network=name,
                    diameter_nm=d,
                    device=args.device,
                    times=7,
                    max_moles_per_original_na_error=error,
                )
            )
    pd.DataFrame(rows).to_csv(out / (args.tag + ".csv"), index=False)
    print("seconds", time.monotonic() - tic)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--device", default="cuda:0")
    cli.add_argument("--tag", required=True)
    cli.add_argument("--names", default="direct,neutral,siv_lag,parallel_neutral")
    cli.add_argument("--diameters", default="100,200,300")
    main(cli.parse_args())
