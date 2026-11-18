"""Check resident stage handoff and exact stochastic continuation on CUDA."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from particula.dynamics import Dilution, DilutionStrategy
from particula.execution.gpu_coupled import GPUCoupledSimulation
from particula.execution.gpu_coupled_restart import (
    load_coupled_checkpoint,
    resume_coupled,
    save_coupled_checkpoint,
)
from particula.execution.tests.gpu_coupled_test import equal_native
from particula.execution.tests.gpu_multimaterial_condensation_test import (
    material_case,
)
from particula.particles import ActivityBAT


@pytest.mark.parametrize("bat", [False, True])
@pytest.mark.parametrize("order", ["events-first", "condensation-first"])
def test_resident_handoff_has_no_physical_download_and_restarts_exactly(
    order, bat, monkeypatch, tmp_path
):
    """Fixed event order and restart survive device-only mass/gas handoff."""
    if not wp.is_cuda_available() or not wp.is_conditional_graph_supported():
        pytest.skip("native conditional CUDA unavailable")
    initial, strategy = material_case(2 if bat else 3)
    if bat:
        strategy.activity_strategy = ActivityBAT(0.2, 0.5, 1400.0)
    events = (Dilution(DilutionStrategy(0.2)),)
    with GPUCoupledSimulation(
        initial,
        events,
        strategy,
        device="cuda:0",
        time_step=0.01,
        order=order,
        resident=True,
    ) as owner:
        buffers = owner._events._session.particles.masses
        original_numpy = wp.array.numpy

        def scalars_only(array, *args, **kwargs):
            assert array.size <= 7, "split stages downloaded a physical array"
            return original_numpy(array, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(wp.array, "numpy", scalars_only)
            owner.step()
        assert owner._events._session.particles.masses is buffers
        assert owner.coupling_uploads == owner.coupling_downloads == 0
        path = tmp_path / "resident-coupled.npz"
        save_coupled_checkpoint(owner.checkpoint(), path)
        owner.run(2)
        final = owner.snapshot()
        for source, target in strategy.species_map.gas_to_particle:
            before = (
                initial.particles.species_mass_density[:, target]
                + initial.gas.concentration[:, source]
            )
            after = (
                final.particles.species_mass_density[:, target]
                + final.gas.concentration[:, source]
            )
            np.testing.assert_allclose(
                after, before * np.exp(-0.2 * 0.03), rtol=3e-13
            )
        saved = load_coupled_checkpoint(path)
        if bat:
            identity, interpolation, digest = saved.configuration
            changed = replace(
                saved, configuration=(identity, interpolation, digest[::-1])
            )
            with pytest.raises(ValueError, match="physics differs"):
                resume_coupled(changed, events, strategy, device="cuda:0")
        with resume_coupled(
            saved, events, strategy, device="cuda:0"
        ) as restored:
            assert restored.resident
            restored.run(2)
            equal_native(final, restored.snapshot())
            assert owner.statistics == restored.statistics
