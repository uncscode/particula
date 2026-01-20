"""Import and integration guards for particula.equilibria."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import particula as par
import pytest
from particula.equilibria import (
    Equilibria,
    EquilibriaFactory,
    EquilibriaStrategy,
    EquilibriumResult,
    LiquidVaporPartitioningBuilder,
    LiquidVaporPartitioningStrategy,
    get_properties_for_liquid_vapor_partitioning,
    liquid_vapor_obj_function,
    liquid_vapor_partitioning,
)

EXPECTED_ALL = {
    "EquilibriaStrategy",
    "LiquidVaporPartitioningStrategy",
    "EquilibriumResult",
    "PhaseConcentrations",
    "LiquidVaporPartitioningBuilder",
    "EquilibriaFactory",
    "Equilibria",
    "liquid_vapor_partitioning",
    "get_properties_for_liquid_vapor_partitioning",
    "liquid_vapor_obj_function",
}


def test_imports_from_module_surface():
    """Surface imports are available from particula.equilibria."""
    assert EquilibriaStrategy is not None
    assert LiquidVaporPartitioningStrategy is not None
    assert Equilibria is not None


def test_imports_via_top_level_package():
    """Top-level particula exposes equilibria surface attributes."""
    assert hasattr(par.equilibria, "Equilibria")
    assert hasattr(par.equilibria, "EquilibriaFactory")
    assert hasattr(par.equilibria, "LiquidVaporPartitioningStrategy")


def test_legacy_imports_continue_to_work():
    """Legacy imports remain available for backward compatibility."""
    assert callable(liquid_vapor_partitioning)
    assert callable(get_properties_for_liquid_vapor_partitioning)
    assert callable(liquid_vapor_obj_function)

    from particula.equilibria.partitioning import (  # local import
        liquid_vapor_partitioning as direct_partitioning,
    )

    assert callable(direct_partitioning)


def test___all___contains_expected_surface():
    """__all__ exports exactly the expected public API surface."""
    from particula import equilibria as eq

    assert set(eq.__all__) == EXPECTED_ALL


@pytest.mark.parametrize(
    ("wrapper_name", "impl_name", "args", "kwargs"),
    [
        (
            "liquid_vapor_partitioning",
            "liquid_vapor_partitioning",
            (1,),
            {"x": 2},
        ),
        (
            "get_properties_for_liquid_vapor_partitioning",
            "get_properties_for_liquid_vapor_partitioning",
            (),
            {"y": 3},
        ),
        (
            "liquid_vapor_obj_function",
            "liquid_vapor_obj_function",
            (np.array([1.0]),),
            {"z": 4},
        ),
    ],
)
def test_deprecation_wrappers_delegate_and_warn(
    monkeypatch,
    wrapper_name: str,
    impl_name: str,
    args: tuple[Any, ...],
    kwargs: dict[
        str,
        Any,
    ],
):
    """Legacy wrappers warn and delegate.

    The wrappers just delegate to the underlying implementations so that we can
    assert the shortcuts still work while warning users to move to strategies.
    """
    sentinel = object()
    calls: list[
        tuple[
            tuple[Any, ...],
            dict[str, Any],
        ]
    ] = []

    def _sentinel(*call_args: Any, **call_kwargs: Any) -> Any:
        calls.append((call_args, call_kwargs))
        return sentinel

    monkeypatch.setattr(par.equilibria._partitioning, impl_name, _sentinel)

    with pytest.warns(DeprecationWarning):
        result = getattr(par.equilibria, wrapper_name)(*args, **kwargs)

    assert result is sentinel
    assert calls == [(args, kwargs)]


def test_integration_strategy_factory_runnable_flow():
    """Strategy → Factory → Runnable wiring executes without heavy compute."""
    factory = EquilibriaFactory()
    strategy = factory.get_strategy(
        "liquid_vapor", parameters={"water_activity": 0.5}
    )
    assert isinstance(strategy, LiquidVaporPartitioningStrategy)

    runnable = Equilibria(strategy=strategy)

    # Minimal aerosol stub carrying required attributes
    aerosol = SimpleNamespace(
        c_star_j_dry=np.array([1e-6, 1e-4]),
        concentration_organic_matter=np.array([1.0, 2.0]),
        molar_mass=np.array([200.0, 200.0]),
        oxygen2carbon=np.array([0.2, 0.3]),
        density=np.array([1200.0, 1200.0]),
    )

    # Solve with small inputs; partition coefficient guess left None
    result = strategy.solve(
        c_star_j_dry=aerosol.c_star_j_dry,
        concentration_organic_matter=aerosol.concentration_organic_matter,
        molar_mass=aerosol.molar_mass,
        oxygen2carbon=aerosol.oxygen2carbon,
        density=aerosol.density,
    )

    assert isinstance(result, EquilibriumResult)

    updated = runnable.execute(aerosol, time_step=1.0, sub_steps=1)
    assert hasattr(updated, "equilibria_result")


def test_builder_creates_strategy():
    """Builder produces a configured strategy."""
    builder = LiquidVaporPartitioningBuilder()
    strategy = builder.set_water_activity(0.75).build()
    assert isinstance(strategy, LiquidVaporPartitioningStrategy)
    assert strategy.water_activity == 0.75
