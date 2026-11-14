"""Conditioned LF2013 GPU evaluation against published decimal polynomials."""

from decimal import Decimal, localcontext
from pathlib import Path
from typing import no_type_check

import numpy as np
import pytest

from particula.dynamics.ions.attachment import attachment_coefficient
from particula.dynamics.ions.attachment_table import centered_attachment_tables

wp = pytest.importorskip("warp")
from particula.gpu.dynamics.ion_attachment import (  # noqa: E402
    classify_ion_particle_wp,
    ion_attachment_wp,
    ion_collision_rate_wp,
)


@no_type_check
@wp.kernel
def evaluate_attachment(
    radius: wp.array(dtype=wp.float64),
    charge: wp.array(dtype=wp.float64),
    polarity: wp.array(dtype=wp.int32),
    policy: wp.int32,
    tables: wp.array3d(dtype=wp.float64),
    result: wp.array(dtype=wp.float64),
):
    """Evaluate a vector of independent fit queries."""
    i = wp.tid()
    result[i] = ion_attachment_wp(
        radius[i], charge[i], polarity[i], policy, tables
    )


def decimal_reference_grid():
    """Evaluate original unshifted coefficients, independent of transformation."""
    directory = (
        Path(__file__).resolve().parents[3] / "util" / "lf2013_coagulation"
    )
    radii, charges, signs, expected = [], [], [], []
    with localcontext() as context:
        context.prec = 70
        for sign, name in ((-1, "S1.txt"), (1, "S2.txt")):
            for line in (directory / name).read_text().splitlines()[1:]:
                row = [Decimal(x) for x in line.split()]
                for radius in np.geomspace(float(row[-2]), float(row[-1]), 9):
                    x = Decimal(str(radius)).log10()
                    exponent = sum(c * x**i for i, c in enumerate(row[1:-4]))
                    radii.append(radius)
                    charges.append(int(row[0]))
                    signs.append(sign)
                    expected.append(float(Decimal(10) ** exponent))
    return tuple(np.asarray(x) for x in (radii, charges, signs, expected))


@pytest.fixture(scope="module")
def reference_grid():
    """Retain independent reference values for both execution targets."""
    return decimal_reference_grid()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_all_charge_domains_match_decimal(device, reference_grid):
    """Both polarities and all 201 charges agree over their published domains."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    radius, charge, polarity, expected = reference_grid
    tables = wp.array(
        centered_attachment_tables(), dtype=wp.float64, device=device
    )
    result = wp.empty(radius.size, dtype=wp.float64, device=device)
    wp.launch(
        evaluate_attachment,
        dim=radius.size,
        device=device,
        inputs=[
            wp.array(radius, dtype=wp.float64, device=device),
            wp.array(charge, dtype=wp.float64, device=device),
            wp.array(polarity, dtype=wp.int32, device=device),
            0,
            tables,
            result,
        ],
    )
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-10, atol=0)
    np.testing.assert_allclose(
        attachment_coefficient(radius, charge, polarity),
        expected,
        rtol=1e-10,
        atol=0,
    )


@pytest.mark.parametrize("policy,expected", [(0, -1.0), (1, 0.0), (2, 1e-15)])
def test_omitted_and_invalid_domains_are_distinct(policy, expected):
    """An explicit bound never permits nonintegral charge or invalid radii."""
    radius = [1.5e-9, 1e-10, 1e-8, 1e-8, 2e-5]
    charge = [1.0, 0.0, 0.5, 0.0, 0.0]
    polarity = [1, 1, 1, 0, 1]
    device = "cpu"
    result = wp.empty(5, dtype=wp.float64, device=device)
    wp.launch(
        evaluate_attachment,
        dim=5,
        device=device,
        inputs=[
            wp.array(radius, dtype=wp.float64, device=device),
            wp.array(charge, dtype=wp.float64, device=device),
            wp.array(polarity, dtype=wp.int32, device=device),
            policy,
            wp.array(
                centered_attachment_tables(), dtype=wp.float64, device=device
            ),
            result,
        ],
    )
    np.testing.assert_array_equal(result.numpy(), [expected, -1, -1, -1, -1])


@no_type_check
@wp.kernel
def evaluate_ion_pairs(
    masses: wp.array3d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    radius: wp.array(dtype=wp.float64),
    pairs: wp.array2d(dtype=wp.int32),
    positive_mass: wp.float64,
    negative_mass: wp.float64,
    recombination: wp.float64,
    tables: wp.array3d(dtype=wp.float64),
    kinds: wp.array2d(dtype=wp.int32),
    rates: wp.array(dtype=wp.float64),
):
    """Probe classified ion interactions without a collision scheduler."""
    index = wp.tid()
    first, second = pairs[index, 0], pairs[index, 1]
    first_kind = classify_ion_particle_wp(
        masses, charge, 0, first, 0, 1, positive_mass, negative_mass
    )
    second_kind = classify_ion_particle_wp(
        masses, charge, 0, second, 0, 1, positive_mass, negative_mass
    )
    kinds[index, 0] = first_kind
    kinds[index, 1] = second_kind
    rates[index] = ion_collision_rate_wp(
        first_kind,
        second_kind,
        radius[first],
        radius[second],
        charge[0, first],
        charge[0, second],
        recombination,
        0,
        tables,
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_ion_pair_classification_and_physical_rates(device):
    """Distinguish recombination, attachment, inert products and invalid clusters."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    positive, negative, alpha = 2e-25, 1e-25, 1.7e-12
    masses = [
        [
            [positive, 0, 0],
            [0, negative, 0],
            [positive, negative, 0],
            [positive, negative, 1e-22],
            [2 * positive, 0, 0],
            [0, 0, 1e-22],
        ]
    ]
    charges = [[1.0, -1.0, 0.0, 0.0, 2.0, 1.0]]
    radius = [2.5e-10, 2.5e-10, 3.15e-10, 1e-8, 3.15e-10, 1.5e-9]
    pairs = np.array(
        [
            [0, 1],
            [0, 0],
            [1, 1],
            [2, 3],
            [0, 3],
            [3, 0],
            [0, 5],
            [4, 0],
            [3, 5],
        ],
        dtype=np.int32,
    )
    kinds = wp.empty((len(pairs), 2), dtype=wp.int32, device=device)
    rates = wp.empty(len(pairs), dtype=wp.float64, device=device)
    wp.launch(
        evaluate_ion_pairs,
        dim=len(pairs),
        device=device,
        inputs=[
            wp.array(masses, dtype=wp.float64, device=device),
            wp.array(charges, dtype=wp.float64, device=device),
            wp.array(radius, dtype=wp.float64, device=device),
            wp.array(pairs, dtype=wp.int32, device=device),
            wp.float64(positive),
            wp.float64(negative),
            wp.float64(alpha),
            wp.array(
                centered_attachment_tables(), dtype=wp.float64, device=device
            ),
            kinds,
            rates,
        ],
    )
    labels = np.array([1, -1, 2, 0, 3, 0])
    np.testing.assert_array_equal(kinds.numpy(), labels[pairs])
    attachment = float(attachment_coefficient(1e-8, 0, 1))
    np.testing.assert_allclose(
        rates.numpy(),
        [alpha, 0, 0, 0, attachment, attachment, -1, -1, -1],
        rtol=1e-10,
        atol=0,
    )
