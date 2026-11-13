"""Variability, forcing, stochastic and large-particle lifecycle studies."""

import numpy as np
from particula.util.constants import GAS_CONSTANT
from scripts.gpu_studies.cases import (
    particle_inventory,
    radius_histogram,
    water_pressure,
)
from scripts.gpu_studies.validation import (
    mapped_inventory,
    require,
    state_hash,
    validate_evolution,
    validate_process_cycle,
)


def _frame(owner, step):
    """Reduce diagnostics and retain complete per-box histogram observations."""
    data = owner.diagnostics()
    snapshot = owner.snapshot()
    edges, histogram = radius_histogram(snapshot)
    mass, number = particle_inventory(snapshot)
    return {
        "step": step,
        "time_s": step * owner.time_step,
        "temperature_K": data.temperature.tolist(),
        "water_kg_m3": data.gas_concentration[:, 2].tolist(),
        "water_saturation": data.saturation_ratio[:, 2].tolist(),
        "represented_number": number.tolist(),
        "particle_mass_kg": mass.tolist(),
        "radius_edges_m": edges.tolist(),
        "histogram_number_m3": histogram.tolist(),
    }, snapshot


def evolution(case, initial, process):
    """Follow heterogeneous response or a prescribed cooling/heating pulse."""
    frames = []
    expected = mapped_inventory(initial)
    external = np.zeros_like(expected)
    max_error = 0.0
    with process.prepare(
        initial, device="cuda:0", time_step=case.time_step, root_seed=case.seed
    ) as owner:
        owner.capture()
        frame, _ = _frame(owner, 0)
        frames.append(frame)
        for step in range(case.steps):
            if case.experiment == "forcing":
                phase = step / case.steps
                temperature = np.interp(
                    phase, [0, 1 / 3, 2 / 3, 1], [298.15, 275, 310, 298.15]
                )
                temperature += np.linspace(-4, 4, case.boxes)
                owner.update_environment(temperature=temperature)
                if step in (case.steps // 3, 2 * case.steps // 3):
                    observed = owner.diagnostics()
                    gas = observed.gas_concentration.copy()
                    humidity = 1.6 if step == case.steps // 3 else 0.8
                    gas[:, 2] = (
                        humidity
                        * water_pressure(temperature)
                        * initial.gas.molar_mass[2]
                        / (GAS_CONSTANT * temperature)
                    )
                    change = (
                        gas[:, 2] - observed.gas_concentration[:, 2]
                    ) * observed.volume
                    external[:, 0] += change
                    expected[:, 0] += change
                    owner.update_gas(gas)
            owner.step()
            expected *= np.exp(-0.02 * case.time_step)
            if (step + 1) % 10 == 0 or step + 1 == case.steps:
                frame, current = _frame(owner, step + 1)
                frames.append(frame)
                actual = mapped_inventory(current)
                np.testing.assert_allclose(
                    actual, expected, rtol=2e-12, atol=1e-30
                )
                max_error = max(
                    max_error, float(np.max(np.abs(actual / expected - 1)))
                )
        final = owner.snapshot()
    if case.experiment != "forcing":
        validate_evolution(case, initial, final)
    require(
        np.any(final.particles.masses != initial.particles.masses),
        "no condensation response",
    )
    return {
        "frames": frames,
        "result_hash": state_hash(final),
        "external_mass_kg_by_box_material": external.tolist(),
        "max_box_species_relative_error": max_error,
    }


def stochastic(case, initial, process):
    """Keep the same initial aerosol and change only the process stream seed."""
    histories, histograms, hashes = [], [], []
    seeds = list(range(16))
    for seed in seeds:
        with process.prepare(
            initial, device="cuda:0", time_step=case.time_step, root_seed=seed
        ) as owner:
            owner.capture()
            history = [owner.diagnostics().number_concentration.tolist()]
            for _ in range(case.steps):
                owner.step()
                history.append(
                    owner.diagnostics().number_concentration.tolist()
                )
            final = owner.snapshot()
        validate_evolution(case, initial, final)
        edges, histogram = radius_histogram(final)
        histories.append(history)
        histograms.append(histogram.tolist())
        hashes.append(state_hash(final))
    require(
        len(set(hashes)) > 1,
        "random stream changes produced no stochastic variation",
    )
    return {
        "seeds": seeds,
        "time_s": (np.arange(case.steps + 1) * case.time_step).tolist(),
        "number_m3_by_seed_time_box": histories,
        "radius_edges_m": edges.tolist(),
        "final_histogram_by_seed_box": histograms,
        "result_hashes": hashes,
        "band_definition": (
            "16 independent stream seeds; ensemble spread, "
            "not a confidence interval"
        ),
    }


def _source_off(owner):
    """Remove the remaining precursor and preserve every other gas."""
    data = owner.diagnostics()
    gas = data.gas_concentration.copy()
    removed = gas[:, 1] * data.volume
    gas[:, 1] = 0
    owner.update_gas(gas)
    return removed


def lifecycle(case, initial, process, processes):
    """Check all five processes, source/sink budgets and exact restart."""
    reference, ledger = validate_process_cycle(case, initial, processes)
    with process.prepare(
        initial, device="cuda:0", time_step=case.time_step, root_seed=case.seed
    ) as original:
        original.capture()
        original.step()
        first = original.snapshot()
        for carrier, names in (
            ("particles", ("masses", "concentration", "charge", "volume")),
            ("gas", ("concentration",)),
        ):
            for name in names:
                np.testing.assert_array_equal(
                    getattr(getattr(first, carrier), name),
                    getattr(getattr(reference, carrier), name),
                )
        removed = _source_off(original)
        saved = original.checkpoint()
        # Detached analysis arrays are not continuation authority.
        first.particles.masses[:] = -123
        saved.resident.particles.masses[:] = -456
        original.run(case.steps - 1)
        final = original.snapshot()
        continuation = original.checkpoint().resident.rng_continuation
    with process.resume(saved) as resumed:
        resumed.capture()
        resumed.run(case.steps - 1)
        continued = resumed.snapshot()
        require(
            state_hash(continued) == state_hash(final),
            "restart changed primary continuation bytes",
        )
        require(
            resumed.checkpoint().resident.rng_continuation == continuation,
            "restart changed random continuation",
        )
    with process.prepare(
        initial, device="cuda:0", time_step=case.time_step, root_seed=case.seed
    ) as ordinary:
        ordinary.step()
        _source_off(ordinary)
        ordinary.run(case.steps - 1)
        np.testing.assert_array_equal(
            ordinary.snapshot().particles.masses, final.particles.masses
        )
        require(
            state_hash(ordinary.snapshot()) == state_hash(final),
            "captured and ordinary trajectories differ",
        )
    science = validate_evolution(case, initial, final)
    edges, histogram = radius_histogram(final)
    return {
        "science": science,
        "process_ledger": ledger,
        "result_hash": state_hash(final),
        "removed_precursor_kg": float(removed.sum()),
        "completed_steps": case.steps,
        "restart_exact": True,
        "capture_exact": True,
        "snapshot_isolated": True,
        "radius_edges_m": edges.tolist(),
        "mean_final_histogram_number_m3": histogram.mean(axis=0).tolist(),
    }


def run_trajectory(case, initial, process, processes):
    """Dispatch the declared complete scientific experiment on actual CUDA."""
    import warp as wp

    require(
        wp.is_cuda_available(), "CUDA is required for scientific trajectories"
    )
    if case.experiment == "stochastic":
        return stochastic(case, initial, process)
    if case.experiment == "lifecycle":
        return lifecycle(case, initial, process, processes)
    if case.experiment in ("forcing", "heterogeneous"):
        return evolution(case, initial, process)
    raise ValueError(f"unknown experiment {case.experiment}")
