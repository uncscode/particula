"""Test the convert module."""

from particula.util.refractive_index_mixing import (
    get_effective_refractive_index,
)


def test_effective_refractive_index():
    """Test the effective_refractive_index function."""

    assert get_effective_refractive_index(
        m_zero=1.5 + 0.5j,
        m_one=1.33,
        volume_zero=10,
        volume_one=5,
    ) == (1.4572585227821824 + 0.3214931829339477j)
