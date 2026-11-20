"""Compare native ROS3 chemistry on NumPy and CUDA with independent stiff ODEs.

Run from the checkout with python -m scripts.assess_gas_chemistry. Outputs
are maximum component errors and solver work for independent integrations
from t=0 to each requested time. No timings or cross-backend bitwise
equivalence are assumed.
"""

import json
from dataclasses import replace

import numpy as np
from docs.examples.processes.customization.native_gas_chemistry import chamber
from particula import ProcessRegistry
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.adapter import GasChemistryAdapter
from particula.dynamics.chemistry.process import ChemistrySolver, GasChemistry
from scipy.integrate import solve_ivp
from scripts.assess_numerical_methods import cases


def benchmark():
    """Give the dimensionless Robertson reference explicit benchmark scales."""
    state = chamber()
    gas = state.gas
    gas.name[:] = ["A", "B", "C"]
    gas.molar_mass[:] = 0.1
    gas.concentration[:] = [0.1, 0.0, 0.0]
    a, b, c = ("gas:A", "gas:B", "gas:C")
    reactions = []
    for name, reactants, products, k, unit in (
        ("decay", {a: 1}, {b: 1}, 0.04, "1/s"),
        ("exchange", {b: 1, c: 1}, {a: 1, c: 1}, 1e4, "m^3/mol/s"),
        ("dimer", {b: 2}, {b: 1, c: 1}, 3e7, "m^3/mol/s"),
    ):
        reactions.append(
            dict(
                id=name,
                reactants=reactants,
                products=products,
                rate=dict(type="constant", k=k, units=unit),
            )
        )
    mechanism = ReactionMechanism(
        {
            "version": 1,
            "basis": "mol/m^3-air",
            "species": {a: {}, b: {}, c: {}},
            "reactions": reactions,
        }
    )
    return state, mechanism.bind(gas)


def reference(times):
    """Integrate the independently written Stage 2A right-hand side twice."""
    case = next(c for c in cases() if c["name"] == "robertson")
    results = []
    for method in ("Radau", "BDF"):
        sol = solve_ivp(
            case["rate"],
            (0.0, float(times[-1])),
            case["initial"],
            method=method,
            t_eval=times,
            rtol=1e-11,
            atol=[1e-14, 1e-17, 1e-14],
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        results.append(sol.y.T)
    np.testing.assert_allclose(*results, rtol=1e-8, atol=1e-14)
    return results


def measure(process, state, times, device):
    """Solve each interval from zero and count the work in box zero."""
    rows = []
    work = np.zeros(4, dtype=int)
    registry = ProcessRegistry()
    registry.register(GasChemistry, GasChemistryAdapter())
    for t in times:
        if device == "numpy":
            y, stats = process.integrate(np.array([1.0, 0.0, 0.0]), 298.15, t)
            work += [
                stats.accepted,
                stats.rejected,
                stats.factorizations,
                stats.linear_solves,
            ]
        else:
            with process.prepare(
                state,
                device=device,
                time_step=float(t),
                process_registry=registry,
            ) as sim:
                sim.step()
                y = sim.snapshot().gas.concentration[0] / state.gas.molar_mass
                payload = next(
                    p
                    for p in sim.checkpoint().extension_state
                    if p.role == "work_counts"
                )
                work += np.frombuffer(
                    payload.data, dtype=payload.dtype
                ).reshape(payload.shape)[0]
        rows.append(y)
    return np.array(rows), work


def main():
    """Compare every species, check tighter tolerances and report solve work."""
    state, bound = benchmark()
    times = np.array([1e-4, 1e-2, 1.0, 100.0])
    refs = reference(times)
    base = ChemistrySolver(method="ros3", atol=1e-14, max_step=10)
    results = []
    for device in ("numpy", "cuda:0"):
        previous = None
        for rtol in (1e-5, 1e-8):
            process = GasChemistry(bound, replace(base, rtol=rtol))
            observed, work = measure(process, state, times, device)
            errors = np.max(np.abs(observed - refs[0]), axis=0)
            if previous is not None:
                if not np.all(errors < previous):
                    raise AssertionError(
                        f"{device}: refinement gave {errors}, was {previous}"
                    )
                for ref in refs:
                    np.testing.assert_allclose(
                        observed, ref, rtol=2e-6, atol=1e-12
                    )
            if np.min(observed) < 0:
                raise AssertionError("negative accepted inventory")
            np.testing.assert_allclose(observed.sum(axis=1), 1.0, rtol=1e-12)
            previous = errors
            results.append(
                dict(
                    device=device,
                    rtol=rtol,
                    max_abs=errors.tolist(),
                    accepted=int(work[0]),
                    rejected=int(work[1]),
                    factorizations=int(work[2]),
                    solves=int(work[3]),
                )
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
