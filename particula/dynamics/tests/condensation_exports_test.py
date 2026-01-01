"""Smoke tests for condensation namespace exports."""

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import particula as par


PAR_DYNAMICS = par.dynamics


def test_imports_from_dynamics_namespace():
    """Staggered condensation exports are available via particula.dynamics."""
    from particula.dynamics import (
        CondensationFactory,
        CondensationIsothermal,
        CondensationIsothermalBuilder,
        CondensationIsothermalStaggered,
        CondensationIsothermalStaggeredBuilder,
    )

    assert CondensationIsothermal is PAR_DYNAMICS.CondensationIsothermal
    assert (
        CondensationIsothermalStaggered
        is PAR_DYNAMICS.CondensationIsothermalStaggered
    )
    assert (
        CondensationIsothermalBuilder
        is PAR_DYNAMICS.CondensationIsothermalBuilder
    )
    assert (
        CondensationIsothermalStaggeredBuilder
        is PAR_DYNAMICS.CondensationIsothermalStaggeredBuilder
    )
    assert CondensationFactory is PAR_DYNAMICS.CondensationFactory


def test_imports_from_condensation_subpackage():
    """Staggered condensation exports are available via subpackage."""
    from particula.dynamics import condensation
    from particula.dynamics.condensation import (
        CondensationFactory,
        CondensationIsothermal,
        CondensationIsothermalBuilder,
        CondensationIsothermalStaggered,
        CondensationIsothermalStaggeredBuilder,
        CondensationStrategy,
    )

    assert CondensationIsothermal is condensation.CondensationIsothermal
    assert (
        CondensationIsothermalStaggered
        is condensation.CondensationIsothermalStaggered
    )
    assert (
        CondensationIsothermalBuilder
        is condensation.CondensationIsothermalBuilder
    )
    assert (
        CondensationIsothermalStaggeredBuilder
        is condensation.CondensationIsothermalStaggeredBuilder
    )
    assert CondensationFactory is condensation.CondensationFactory
    assert CondensationStrategy is condensation.CondensationStrategy


def test_condensation_exports_available_via_par_dynamics():
    """Shorthand namespace exposes staggered condensation symbols."""
    assert hasattr(PAR_DYNAMICS, "CondensationIsothermalStaggered")
    assert hasattr(PAR_DYNAMICS, "CondensationIsothermalStaggeredBuilder")

    strategy = PAR_DYNAMICS.CondensationIsothermalStaggered
    builder = PAR_DYNAMICS.CondensationIsothermalStaggeredBuilder

    assert strategy.__name__ == "CondensationIsothermalStaggered"
    assert builder.__name__ == "CondensationIsothermalStaggeredBuilder"


def test_condensation_export_memberships():
    """Exports are listed in __all__ as expected."""
    from particula.dynamics import condensation

    assert "CondensationIsothermalStaggered" in condensation.__all__
    assert "CondensationIsothermalStaggeredBuilder" in condensation.__all__
    assert "CondensationFactory" in condensation.__all__
    assert "CondensationIsothermal" in condensation.__all__
    assert "CondensationIsothermalBuilder" in condensation.__all__
    assert "CondensationStrategy" in condensation.__all__

    assert "CondensationIsothermalStaggered" in PAR_DYNAMICS.__all__
    assert "CondensationIsothermalStaggeredBuilder" in PAR_DYNAMICS.__all__
    assert "CondensationFactory" in PAR_DYNAMICS.__all__
    assert "CondensationIsothermal" in PAR_DYNAMICS.__all__
    assert "CondensationIsothermalBuilder" in PAR_DYNAMICS.__all__
