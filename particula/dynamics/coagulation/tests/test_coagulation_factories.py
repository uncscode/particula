import pytest
from particula.dynamics.coagulation.coagulation_factories import (
    CoagulationFactory,
)
from particula.dynamics.coagulation.coagulation_strategy import (
    BrownianCoagulationStrategy,
    ChargedCoagulationStrategy,
    TurbulentShearCoagulationStrategy,
    TurbulentDNSCoagulationStrategy,
    CombineCoagulationStrategy,
)


def test_brownian_coagulation():
    factory = CoagulationFactory()
    strategy = factory.get_strategy(
        "brownian", {"distribution_type": "discrete"}
    )
    assert isinstance(strategy, BrownianCoagulationStrategy)


def test_charged_coagulation():
    factory = CoagulationFactory()
    strategy = factory.get_strategy(
        "charged", {"distribution_type": "discrete"}
    )
    assert isinstance(strategy, ChargedCoagulationStrategy)


def test_turbulent_shear_coagulation():
    factory = CoagulationFactory()
    strategy = factory.get_strategy(
        "turbulent_shear", {"distribution_type": "discrete"}
    )
    assert isinstance(strategy, TurbulentShearCoagulationStrategy)


def test_turbulent_dns_coagulation():
    factory = CoagulationFactory()
    strategy = factory.get_strategy(
        "turbulent_dns", {"distribution_type": "discrete"}
    )
    assert isinstance(strategy, TurbulentDNSCoagulationStrategy)


def test_combine_coagulation():
    factory = CoagulationFactory()
    strategy = factory.get_strategy(
        "combine", {"distribution_type": "discrete"}
    )
    assert isinstance(strategy, CombineCoagulationStrategy)


def test_invalid_coagulation():
    factory = CoagulationFactory()
    with pytest.raises(ValueError) as excinfo:
        factory.get_strategy("nonexistent", {})
    assert "Unknown coagulation strategy type" in str(excinfo.value)
