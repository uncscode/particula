"""Smoke tests for dilution namespace exports."""

import particula as par
from particula.dynamics import Dilution, DilutionStrategy
from particula.dynamics.dilution import DilutionStrategy as ModuleStrategy
from particula.dynamics.particle_process import Dilution as ModuleDilution


def test_dilution_exports_are_supported_public_symbols():
    """Public dynamics exports retain their implementation identities."""
    assert DilutionStrategy is ModuleStrategy
    assert Dilution is ModuleDilution
    assert par.dynamics.DilutionStrategy is DilutionStrategy
    assert par.dynamics.Dilution is Dilution
    assert "DilutionStrategy" in par.dynamics.__all__
    assert "Dilution" in par.dynamics.__all__


def test_dilution_exports_construct_supported_runnable():
    """The shorthand dynamics namespace constructs the dilution runnable."""
    strategy = par.dynamics.DilutionStrategy(0.25)

    assert isinstance(par.dynamics.Dilution(strategy), Dilution)
