"""Native population and species-map admission before payload access."""

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
        ("nucleation", "_preflight_nucleation"),
    ],
)
@pytest.mark.parametrize("code", [0, 1, 2, 3])
def test_staged_nucleation_requires_metadata_and_binding(module, name, code):
    """Resolved metadata alone cannot supply the staged species binding."""
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
    message = (
        "requires species_binding" if code == 3 else "only particle_resolved"
    )
    with pytest.raises(ValueError, match=message):
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


@pytest.mark.parametrize(
    "name",
    [
        "resampling_step_gpu",
        "_prepare_resampling_step_gpu",
        "representative_volume_scaling_step_gpu",
        "_prepare_representative_volume_scaling_step_gpu",
    ],
)
@pytest.mark.parametrize("code", [None, 0, 1, 2])
def test_exhaustion_requires_native_metadata_before_payload(name, code):
    """Every active exhaustion entry rejects absent or unsupported semantics."""
    pytest.importorskip("warp")
    module = importlib.import_module("particula.gpu.kernels.exhaustion")
    function = getattr(module, name)

    class MetadataOnly:
        def __getattr__(self, field):
            if field == "distribution_kind":
                raise AttributeError(field)
            raise AssertionError(f"payload accessed: {field}")

    particles = MetadataOnly()
    if code is not None:
        particles.distribution_kind = code
    kwargs = {
        key: None
        for key, parameter in inspect.signature(function).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    kwargs["particles"] = particles
    with pytest.raises(ValueError, match="only particle_resolved"):
        function(**kwargs)
