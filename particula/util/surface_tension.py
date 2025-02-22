"""
Methods to calculate surface tension for pure water, dry mixtures, and
wet mixtures of solutes and water.

This module includes:
- water : Surface tension of water based on fitted equations.
- dry_mixing : Surface tension of a solute mixture without water.
- wet_mixing : Surface tension of a solute mixture in the presence of water
               using either a film or volume method.

Examples:
    ``` py title="Example Usage"
    import numpy as np
    from particula.util.surface_tension import water, dry_mixing, wet_mixing

    # Water at 298 K
    sigma_water = water(298)
    print(sigma_water)

    # Dry Mixture
    vols = np.array([0.3, 0.7])
    sigs = np.array([0.030, 0.072])
    sigma_dry_mix = dry_mixing(vols, sigs)
    print(sigma_dry_mix)

    # Wet Mixture (film method)
    sigma_wet_mix_film = wet_mixing(
        volume_solute=1e-23,
        volume_water=1e-21,
        wet_radius=1e-7,
        surface_tension_solute=0.030,
        temperature=298,
        method="film"
    )
    print(sigma_wet_mix_film)
    ```

References:
    - Kalová, J., & Mareš, R. (2018). "Surface tension of water."
      Journal of Chemical Physics, 149(8), 084501.
    - AIOMFAC Model: http://aiomfac.caltech.edu/

To be moved to particle.properties. -kyle
"""

# pylint: disable=too-many-positional-arguments, too-many-arguments

import numpy as np


def get_surface_tension_water(temperature, critical_temperature=647.15):
    """
    Calculate the surface tension of water using a fitted equation.

    The equation is adapted from:
    - σ = 241.322 × (τ^1.26) × (1 - 0.0589 × √τ - 0.56917 × τ),
      where τ = 1 - T / Tcrit, and Tcrit is the critical temperature.

    Arguments:
        - temperature : Ambient temperature (K).
        - critical_temperature : Critical temperature of water (K), defaults
          to 647.15 K.

    Returns:
        - Surface tension in N/m at the given temperature.

    Examples:
        ``` py title="Example Usage"
        from particula.util.surface_tension import water
        sigma_298k = water(298)
        print(sigma_298k)
        # ~0.072 N/m
        ```

    References:
        - Kalová, J., & Mareš, R. (2018). "Surface tension of water."
          Journal of Chemical Physics, 149(8), 084501.
    """
    # Dimensionless parameter from fitting equation
    tau = 1 - temperature / critical_temperature

    # Surface tension in mN/m
    sigma = 241.322 * (tau**1.26) * (1 - 0.0589 * (tau**0.5) - 0.56917 * tau)

    return sigma / 1000


def get_surface_tension_volume_mix(volume_fractions, surface_tensions):
    """
    Calculate the effective surface tension of a dry mixture of solutes.

    If the sum of volume_fractions is not 1, they are normalized. The resulting
    surface tension is:
    - σ_mix = ∑ (φ_i × σ_i)
        - σ_mix  : effective surface tension,
        - φ_i    : volume fraction of each solute,
        - σ_i    : surface tension of each solute.

    Arguments:
        - volume_fractions : NDArray of volume fractions for each solute.
        - surface_tensions : NDArray of surface tensions (N/m) for each solute.

    Returns:
        - Effective surface tension of the mixture (N/m).

    Examples:
        ``` py title="Example Usage"
        import numpy as np
        from particula.util.surface_tension import dry_mixing

        vols = np.array([0.3, 0.7])
        sigs = np.array([0.030, 0.072])
        sigma_mix = dry_mixing(vols, sigs)
        print(sigma_mix)
        # ~0.0594 N/m
        ```

    References:
        - AIOMFAC Model: http://aiomfac.caltech.edu/
    """
    if np.sum(volume_fractions) != 1:
        volume_fractions = volume_fractions / np.sum(volume_fractions)

    # Calculate the surface tension of the mixture
    return np.sum(volume_fractions * surface_tensions)


# pylint: disable=too-many-positional-arguments, too-many-arguments
def get_surface_tension_film_coating(
    volume_solute,
    volume_water,
    wet_radius,
    surface_tension_solute,
    temperature,
    method="film",
):
    """
    Calculate the effective surface tension of a mixture with solutes and
    water.

    Two methods are available:
    1) "film": Checks if solute can form a complete monolayer film of thickness
       ~0.3 nm. The surface tension is either that of the solute
       (fully covered) or a weighted combination of solute and water.
    2) "volume": Computes the weighted average based on volumetric
       contributions.

    Arguments:
        - volume_solute : NDArray or float, volume of solute mixture (m³).
        - volume_water : NDArray or float, volume of water (m³).
        - wet_radius : NDArray or float, droplet radius (m).
        - surface_tension_solute : NDArray or float, surface tension of the
            solute (N/m).
        - temperature : float, droplet temperature (K).
        - method : str, either "film" or "volume" method for mixing.

    Returns:
        - Effective surface tension of the droplet (N/m).

    Raises:
        - ValueError : If an invalid method is specified.

    Examples:
        ``` py title="Example Usage - Film Method"
        import numpy as np
        from particula.util.surface_tension import wet_mixing

        sigma_wet_film = wet_mixing(
            volume_solute=1e-23,
            volume_water=1e-21,
            wet_radius=1e-7,
            surface_tension_solute=0.030,
            temperature=298,
            method="film"
        )
        print(sigma_wet_film)
        # ~Some value between 0.030 and water(298)

        # Example with volume method
        sigma_wet_vol = wet_mixing(
            volume_solute=1e-23,
            volume_water=1e-21,
            wet_radius=1e-7,
            surface_tension_solute=0.030,
            temperature=298,
            method="volume"
        )
        print(sigma_wet_vol)
        # Weighted average
        ```

    References:
        - Kalová, J., & Mareš, R. (2018). "Surface tension of water."
          Journal of Chemical Physics, 149(8), 084501.
        - AIOMFAC Model for organic film thickness: http://aiomfac.caltech.edu/
    """

    # Organic-film method
    if method == "film":

        # Characteristic monolayer thickness taken from AIOMFAC
        mono = 0.3e-9

        # Volume of the monolayer
        film_volume = (
            (4 / 3) * np.pi * (wet_radius**3 - (wet_radius - mono) ** 3)
        )

        # Determine if there's enough organics to cover the particle
        surface_coverage = volume_solute / film_volume

        if surface_coverage >= 1:
            # If the particle is completely covered, the surface tension is
            # just the surface tension of the solute
            sigma = surface_tension_solute
        else:
            # If the particle is not completely covered, the surface tension
            # is a weighted average of the surface tension of the solute and
            # water
            sigma = (
                surface_tension_solute * surface_coverage
                + get_surface_tension_water(temperature)
                * (1 - surface_coverage)
            )
        return sigma
    if method == "volume":
        # Volume method
        sigma = (
            volume_solute * surface_tension_solute
            + volume_water * get_surface_tension_water(temperature)
        ) / (volume_solute + volume_water)
        return sigma
    raise ValueError("Invalid method")
