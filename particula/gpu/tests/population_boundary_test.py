"""Fail-closed interim boundaries for population consumers not yet migrated.

The M3 science guards are replaced by positive scientific admission in their
own phases. Resident/orchestration guards remain for M4. These checks assert
rejection before payload access rather than treating rejection as evidence of
completed numerical migration.
"""

import importlib
import inspect

import pytest


class _TaggedOnly:
    """A valid tag with forbidden payload access at an unsupported boundary."""

    distribution_kind = 3

    def __getattr__(self, name):
        raise AssertionError(f"unsupported consumer accessed {name}")


@pytest.mark.parametrize(
    "module,name",
    [
        ("wall_loss", "_prepare_selected_wall_loss_call"),
        ("wall_loss", "wall_loss_selected_boxes_step_gpu"),
        ("dilution", "_prepare_dilution_step_gpu"),
        ("nucleation", "_preflight_nucleation"),
        ("nucleation", "_prepare_nucleation_step_gpu"),
        ("exhaustion", "_execute_resampling_step_gpu"),
        ("exhaustion", "_execute_representative_volume_scaling_step_gpu"),
        ("exhaustion", "_prepare_resampling_step_gpu"),
        ("exhaustion", "_prepare_representative_volume_scaling_step_gpu"),
        ("communication", "volume_evolution_step_gpu"),
        ("communication", "particle_communication_step_gpu"),
    ],
)
@pytest.mark.parametrize("code", [1, 2, 3])
def test_unmigrated_consumers_reject_before_payload_access(module, name, code):
    """Every direct/prepared route protects the newly transferable tag."""
    pytest.importorskip("warp")
    function = getattr(
        importlib.import_module(f"particula.gpu.kernels.{module}"), name
    )
    kwargs = {
        key: None
        for key, parameter in inspect.signature(function).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    kwargs["particles"] = _TaggedOnly()
    kwargs["particles"].distribution_kind = code
    with pytest.raises(ValueError, match="require a migrated consumer"):
        function(**kwargs)


def test_native_condensation_requires_binding_before_payload_access():
    """A tag alone never implies an identity species map."""
    pytest.importorskip("warp")
    from particula.gpu.kernels.condensation import (
        _prepare_condensation_step_gpu,
    )

    with pytest.raises(ValueError, match="requires species_binding"):
        _prepare_condensation_step_gpu(_TaggedOnly(), None, None, None, 0.1)


@pytest.mark.parametrize("code", [1, 2])
def test_native_coagulation_rejects_unsupported_population_before_payload(code):
    """The resolved migration does not admit PMF or PDF selector algorithms."""
    pytest.importorskip("warp")
    from particula.gpu.kernels.coagulation import _prepare_coagulation_step_gpu

    particles = _TaggedOnly()
    particles.distribution_kind = code
    with pytest.raises(ValueError, match="only particle_resolved"):
        _prepare_coagulation_step_gpu(particles, None, None, 0.1)


@pytest.mark.parametrize("code", [1, 2])
def test_native_wall_loss_rejects_unsupported_population_before_payload(code):
    """A resolved-removal kernel is not a PMF/PDF loss algorithm."""
    pytest.importorskip("warp")
    from particula.gpu.kernels.wall_loss import _prepare_wall_loss_step_gpu

    particles = _TaggedOnly()
    particles.distribution_kind = code
    with pytest.raises(ValueError, match="only particle_resolved"):
        _prepare_wall_loss_step_gpu(particles, None, None, 0.1, config=None)
