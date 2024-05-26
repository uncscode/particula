""""Surface tension strategies factory."""



# Surface strategies factory
def surface_strategy_factory(
    strategy_type: str = "default",
    **kwargs: dict  # type: ignore
) -> SurfaceStrategy:
    """
    Factory function to create surface tension strategies.

    Args:
    -----
    - strategy_type (str): Type of strategy to create, options are:
        - "molar": Surface tension and density based on mole fractions.
        - "mass": Surface tension and density based on mass fractions.
        - "volume": Surface tension and density based on volume fractions.
    - **kwargs (dict): Keyword arguments for the strategy.

    Returns:
    --------
    - SurfaceStrategy: Instance of the surface tension strategy.
    """
