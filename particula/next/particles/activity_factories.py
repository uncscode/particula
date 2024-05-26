"""Activity strategy factories for calculating activity and partial pressure
of species in a mixture of liquids."""

from particula.next.particles.activity_strategies import (
    IdealActivityMass,
    IdealActivityMolar,
    KappaParameterActivity
)


# Factory function for creating activity strategies
def particle_activity_strategy_factory(
            strategy_type: str = "mass_ideal",
            **kwargs: dict  # type: ignore
        ):
    """
    Factory function for creating activity strategies. Used for calculating
    activity and partial pressure of species in a mixture of liquids.

    Args:
    - strategy_type (str): Type of activity strategy to use. The options are:
        - molar_ideal: Ideal activity based on mole fractions.
        - mass_ideal: Ideal activity based on mass fractions.
        - kappa: Non-ideal activity based on kappa hygroscopic parameter.
    - kwargs: Arguments for the activity strategy."""
    if strategy_type.lower() == "molar_ideal":
        return IdealActivityMolar(**kwargs)
    if strategy_type.lower() == "mass_ideal":
        return IdealActivityMass()
    if strategy_type.lower() == "kappa":
        return KappaParameterActivity(**kwargs)
    raise ValueError(f"Unknown strategy type call: {strategy_type}")
