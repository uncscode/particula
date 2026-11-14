"""Charged model formula parity and limiting behavior on actual devices."""

from typing import no_type_check

import numpy as np
import pytest

from particula.dynamics.coagulation import charged_dimensionless_kernel as cpu

wp = pytest.importorskip("warp")

from particula.gpu.dynamics.coagulation_funcs import (  # noqa: E402
    charged_dimensionless_wp,
)


@no_type_check
@wp.kernel
def _evaluate(
    model: wp.int32,
    kn: wp.array(dtype=wp.float64),
    phi: wp.array(dtype=wp.float64),
    hard: wp.array(dtype=wp.float64),
    result: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    result[i] = charged_dimensionless_wp(model, kn[i], phi[i], hard[i])


MODELS = (
    None,
    cpu.get_coulomb_kernel_dyachkov2007,
    cpu.get_coulomb_kernel_gatti2008,
    cpu.get_coulomb_kernel_gopalakrishnan2012,
    cpu.get_coulomb_kernel_chahl2019,
)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("model", range(5))
def test_charged_dimensionless_matches_cpu(device, model):
    """Compare all models over attraction and the transition regime."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    kn, phi = np.meshgrid(np.logspace(-2, 2, 13), [0.1, 0.5, 1.0, 5.0, 10.0])
    kn, phi = kn.ravel(), phi.ravel()
    hard = cpu.get_hard_sphere_kernel(kn)
    expected = hard if model == 0 else MODELS[model](kn, phi)
    result = wp.zeros(kn.size, dtype=wp.float64, device=device)
    wp.launch(
        _evaluate,
        dim=kn.size,
        inputs=[
            model,
            wp.array(kn, dtype=wp.float64, device=device),
            wp.array(phi, dtype=wp.float64, device=device),
            wp.array(hard, dtype=wp.float64, device=device),
            result,
        ],
        device=device,
    )
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize("phi", [0.0, -1.0, -10.0])
def test_dyachkov_neutral_dimensionless_limit(phi):
    """Use the analytic limiting expression, independent of CPU cancellation."""
    kn = np.logspace(-3, 3, 17)
    result = wp.zeros(kn.size, dtype=wp.float64, device="cpu")
    arrays = [
        wp.array(x, dtype=wp.float64, device="cpu")
        for x in (kn, np.full_like(kn, phi), cpu.get_hard_sphere_kernel(kn))
    ]
    wp.launch(_evaluate, dim=kn.size, inputs=[1, *arrays, result], device="cpu")
    expected = 4 * np.pi * kn**2 / (np.sqrt(2 * np.pi) * kn + 1 / (1 + kn))
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-12)


def test_cpu_dyachkov_neutral_limit():
    """The CPU reference must not lose significant digits at zero charge."""
    kn = np.logspace(-3, 3, 17)
    expected = 4 * np.pi * kn**2 / (np.sqrt(2 * np.pi) * kn + 1 / (1 + kn))
    for phi in (0.0, -1.0, -10.0):
        np.testing.assert_allclose(
            cpu.get_coulomb_kernel_dyachkov2007(kn, phi), expected, rtol=1e-12
        )


@pytest.mark.parametrize("model", ["unknown", None, 2, []])
def test_unknown_charged_model_rejects_without_runtime(model):
    """Configuration errors cannot touch device state or initialize RNG."""
    from particula.gpu.kernels._coagulation_config import (
        CoagulationMechanismConfig,
        resolve_coagulation_mechanism_config,
    )

    with pytest.raises(ValueError, match="Unknown charged"):
        resolve_coagulation_mechanism_config(
            CoagulationMechanismConfig(charged_model=model)
        )


def test_charged_model_cannot_silently_change_brownian_selection():
    """Selecting a fit without enabling charged physics rejects explicitly."""
    from particula.gpu.kernels._coagulation_config import (
        CoagulationMechanismConfig,
        resolve_coagulation_mechanism_config,
    )

    with pytest.raises(ValueError, match="requires the charged"):
        resolve_coagulation_mechanism_config(
            CoagulationMechanismConfig(charged_model="gatti2008")
        )
