"""Independent inventory and population checks for published studies."""

import hashlib

import numpy as np
from particula._cpu_batch import arrays
from scripts.gpu_studies.cases import particle_inventory


def require(condition, message):
    """Keep scientific gates active even when Python optimization is enabled."""
    if not condition:
        raise AssertionError(message)


def state_hash(aerosol):
    """Hash all native array bytes and scientific metadata in stable order."""
    digest = hashlib.sha256()
    digest.update(
        repr((aerosol.gas.name, aerosol.particles.distribution_type)).encode()
    )
    for name, value in sorted(arrays(aerosol).items()):
        digest.update(repr((name, value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def valid_state(aerosol):
    """Require finite physical state after the complete workload."""
    for name, value in arrays(aerosol).items():
        require(np.all(np.isfinite(value)), f"nonfinite {name}")
    for carrier, name in (
        (aerosol.particles, "masses"),
        (aerosol.particles, "concentration"),
        (aerosol.gas, "concentration"),
    ):
        require(np.all(getattr(carrier, name) >= 0), f"negative {name}")
    require(np.all(aerosol.particles.volume > 0), "nonpositive volume")
    require(
        np.all(aerosol.environment.temperature > 0), "nonpositive temperature"
    )
    require(np.all(aerosol.environment.pressure > 0), "nonpositive pressure")


def validate_evolution(case, initial, result):
    """Check individual species and gas lanes; never hide drift in total gas."""
    valid_state(result)
    old_mass, old_number = particle_inventory(initial)
    mass, number = particle_inventory(result)
    old_charge = np.sum(
        initial.particles.charge * initial.particles.concentration, axis=1
    )
    charge = np.sum(
        result.particles.charge * result.particles.concentration, axis=1
    )
    factor = np.exp(-0.02 * case.time_step * case.steps)
    error = 0.0
    if case.family == "brownian":
        np.testing.assert_allclose(mass, old_mass, rtol=2e-12, atol=1e-30)
        np.testing.assert_allclose(charge, old_charge, rtol=2e-12, atol=1e-12)
        np.testing.assert_array_equal(
            result.gas.concentration, initial.gas.concentration
        )
        require(np.all(number <= old_number), "coagulation increased number")
        require(np.any(number < old_number), "Brownian study had no collisions")
        error = float(np.max(np.abs(mass / old_mass - 1)))
    elif case.family in ("condensation", "dilution"):
        np.testing.assert_allclose(
            number, old_number * factor, rtol=2e-12, atol=1e-30
        )
        expected, actual = old_mass.copy(), mass.copy()
        if case.family == "condensation":
            expected[:, 0] += (
                initial.gas.concentration[:, 2] * initial.particles.volume
            )
            actual[:, 0] += (
                result.gas.concentration[:, 2] * result.particles.volume
            )
            # Check particle change independently of the gas reservoir.
            require(
                np.any(result.particles.masses != initial.particles.masses),
                "condensation was inactive",
            )
            lanes = [i for i in range(case.gases) if i != 2]
        else:
            np.testing.assert_array_equal(
                result.particles.masses, initial.particles.masses
            )
            lanes = list(range(case.gases))
        np.testing.assert_allclose(
            actual, expected * factor, rtol=2e-12, atol=1e-30
        )
        np.testing.assert_allclose(
            result.gas.concentration[:, lanes],
            initial.gas.concentration[:, lanes] * factor,
            rtol=2e-12,
            atol=1e-30,
        )
        error = float(np.max(np.abs(actual / (expected * factor) - 1)))
    else:
        # Internal source and condensation move mapped gas mass into particles;
        # wall loss, dilution and the explicit precursor removal are sinks.
        expected, actual = old_mass.copy(), mass.copy()
        for gas, material in ((1, 1), (2, 0)):
            expected[:, material] += (
                initial.gas.concentration[:, gas] * initial.particles.volume
            )
            actual[:, material] += (
                result.gas.concentration[:, gas] * result.particles.volume
            )
        require(
            np.all(actual <= expected * (1 + 2e-12) + 1e-30),
            "full-cycle inventory grew",
        )
        require(np.any(number != old_number), "full cycle was inactive")
        error = None
    return {
        "inventory_max_relative_error": error,
        "initial_active_slots": int(
            np.count_nonzero(initial.particles.concentration)
        ),
        "final_active_slots": int(
            np.count_nonzero(result.particles.concentration)
        ),
        "initial_represented_number": float(old_number.sum()),
        "final_represented_number": float(number.sum()),
    }


def native_cpu_steps(case, aerosol, processes):
    """Use matching NumPy dilution and four equal condensation substeps only."""
    from particula.dynamics import MassCondensation

    if case.family not in ("condensation", "dilution"):
        raise ValueError(
            "native CPU timing is limited to equivalent deterministic physics"
        )
    for _ in range(case.steps):
        for process in processes:
            process.execute(
                aerosol,
                case.time_step,
                sub_steps=4 if isinstance(process, MassCondensation) else 1,
            )
    return aerosol


def mapped_inventory(aerosol):
    """Track water and precursor in their declared particle material lanes."""
    result = particle_inventory(aerosol)[0]
    for gas, material in ((2, 0), (1, 1)):
        result[:, material] += (
            aerosol.gas.concentration[:, gas] * aerosol.particles.volume
        )
    return result


def validate_process_cycle(case, initial, processes):
    """Independently account for every source/sink in one full process cycle.

    Each existing process is prepared separately with the same logical box/RNG
    identities. Intermediate CPU states permit per-box, per-material checks;
    this observation-heavy reference is never timed as resident execution.
    """
    from particula.dynamics import (
        Coagulation,
        Dilution,
        MassCondensation,
        Nucleation,
        WallLoss,
    )

    current = initial
    ledger = []
    for process in processes:
        before = current
        with process.prepare(
            before,
            device="cuda:0",
            time_step=case.time_step,
            root_seed=case.seed,
        ) as owner:
            owner.step()
            current = owner.snapshot()
        before_mass, before_number = particle_inventory(before)
        mass, number = particle_inventory(current)
        family = type(process).__name__
        expected = mapped_inventory(before)
        removed = np.zeros_like(before_mass)
        if isinstance(process, WallLoss):
            lost = (before.particles.concentration > 0) & (
                current.particles.concentration == 0
            )
            removed = np.sum(
                before.particles.masses
                * before.particles.concentration[..., None]
                * lost[..., None],
                axis=1,
            )
            np.testing.assert_allclose(
                mass, before_mass - removed, rtol=2e-12, atol=1e-30
            )
            np.testing.assert_array_equal(
                current.particles.concentration[~lost],
                before.particles.concentration[~lost],
            )
            np.testing.assert_array_equal(
                current.particles.masses[~lost], before.particles.masses[~lost]
            )
            np.testing.assert_array_equal(
                current.gas.concentration, before.gas.concentration
            )
            expected -= removed
        elif isinstance(process, Dilution):
            factor = np.exp(-0.02 * case.time_step)
            np.testing.assert_allclose(
                mass, before_mass * factor, rtol=2e-12, atol=1e-30
            )
            np.testing.assert_allclose(
                number, before_number * factor, rtol=2e-12, atol=1e-30
            )
            np.testing.assert_allclose(
                current.gas.concentration,
                before.gas.concentration * factor,
                rtol=2e-12,
                atol=1e-30,
            )
            removed = expected * (1 - factor)
            expected *= factor
        elif isinstance(process, Nucleation):
            np.testing.assert_array_equal(number - before_number, 2)
            precursor = (
                before.gas.concentration[:, 1] - current.gas.concentration[:, 1]
            ) * before.particles.volume
            from particula.util.constants import AVOGADRO_NUMBER

            np.testing.assert_allclose(
                precursor,
                2 * before.gas.molar_mass[1] / AVOGADRO_NUMBER,
                rtol=2e-12,
                atol=0,
            )
        elif isinstance(process, Coagulation):
            np.testing.assert_allclose(
                mass, before_mass, rtol=2e-12, atol=1e-30
            )
            require(
                np.all(number <= before_number), "coagulation increased number"
            )
        elif isinstance(process, MassCondensation):
            uptake = (
                before.gas.concentration[:, 2] - current.gas.concentration[:, 2]
            ) * before.particles.volume
            np.testing.assert_allclose(
                mass[:, 0] - before_mass[:, 0], uptake, rtol=2e-10, atol=1e-30
            )
            np.testing.assert_allclose(
                mass[:, 1:], before_mass[:, 1:], rtol=2e-12, atol=1e-30
            )
        actual = mapped_inventory(current)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-30)
        valid_state(current)
        ledger.append(
            {
                "process": family,
                "number_before": float(before_number.sum()),
                "number_after": float(number.sum()),
                "sink_mass_kg_by_material": removed.sum(axis=0).tolist(),
                "max_box_species_relative_error": float(
                    np.max(np.abs(actual / expected - 1))
                ),
            }
        )
    return current, ledger
