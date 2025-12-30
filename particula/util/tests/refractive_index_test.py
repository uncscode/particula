"""Edge-case coverage for refractive index mixing helper."""

import cmath

import pytest

from particula.util.refractive_index_mixing import (
    get_effective_refractive_index,
)


def _expected_effective_index(m_zero, m_one, volume_zero, volume_one):
    volume_total = volume_zero + volume_one
    r_effective = volume_zero / volume_total * (m_zero - 1) / (
        m_zero + 2
    ) + volume_one / volume_total * (m_one - 1) / (m_one + 2)
    return (2 * r_effective + 1) / (1 - r_effective)


def test_effective_refractive_index_matches_formula_for_real_inputs():
    """Real inputs should follow the published mixing rule."""
    m_zero = 1.5
    m_one = 1.33
    volume_zero = 10.0
    volume_one = 5.0
    expected = _expected_effective_index(m_zero, m_one, volume_zero, volume_one)
    assert get_effective_refractive_index(
        m_zero=m_zero,
        m_one=m_one,
        volume_zero=volume_zero,
        volume_one=volume_one,
    ) == pytest.approx(expected)


def test_effective_refractive_index_handles_complex_inputs():
    """Complex refractive indices should remain finite and follow the rule."""
    m_zero = 1.5 + 0.5j
    m_one = 1.3 + 0.2j
    volume_zero = 2.0
    volume_one = 3.0
    expected = _expected_effective_index(m_zero, m_one, volume_zero, volume_one)
    result = get_effective_refractive_index(
        m_zero=m_zero,
        m_one=m_one,
        volume_zero=volume_zero,
        volume_one=volume_one,
    )
    assert result == pytest.approx(expected)
    assert cmath.isfinite(result)


def test_zero_volumes_raise_zero_division():
    """Document current behavior when no volume is provided."""
    with pytest.raises(ZeroDivisionError):
        get_effective_refractive_index(1.5, 1.33, 0.0, 0.0)


def test_near_zero_volumes_remain_finite():
    """Tiny volumes should not result in infinities or NaNs."""
    result = get_effective_refractive_index(1.5, 1.33, 1e-9, 2e-9)
    assert cmath.isfinite(result)


def test_volume_fractions_are_automatically_normalized():
    """Volume fractions not summing to one are normalized by total volume."""
    m_zero = 1.45
    m_one = 1.25
    volume_zero = 0.2
    volume_one = 0.9
    expected = _expected_effective_index(m_zero, m_one, volume_zero, volume_one)
    result = get_effective_refractive_index(
        m_zero=m_zero,
        m_one=m_one,
        volume_zero=volume_zero,
        volume_one=volume_one,
    )
    assert result == pytest.approx(expected)
