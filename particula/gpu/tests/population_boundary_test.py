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
        ("coagulation", "_prepare_coagulation_step_gpu"),
        ("wall_loss", "_prepare_wall_loss_step_gpu"),
        ("wall_loss", "_prepare_selected_wall_loss_call"),
        ("dilution", "_prepare_dilution_step_gpu"),
        ("nucleation", "_preflight_nucleation"),
        ("nucleation", "_prepare_nucleation_step_gpu"),
        ("exhaustion", "_execute_resampling_step_gpu"),
        ("exhaustion", "_execute_representative_volume_scaling_step_gpu"),
        ("exhaustion", "_prepare_resampling_step_gpu"),
        ("exhaustion", "_prepare_representative_volume_scaling_step_gpu"),
    ],
)
def test_unmigrated_consumers_reject_before_payload_access(module, name):
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
