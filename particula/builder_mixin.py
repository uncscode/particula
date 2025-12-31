"""Builder mixin helpers for assigning particle attributes and units.

This module defines mixins that provide builders with fluent setters for
physical quantities such as density, surface tension, concentration, and
chamber geometry while handling unit conversions and input validation.
"""

# pylint: disable=too-few-public-methods

import logging
from typing import Optional, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from particula.util.convert_units import get_unit_conversion
from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")


class BuilderDensityMixin:
    """Mixin that stores a particle density with unit conversion.

    This mixin provides `set_density` so builders can accept density values
    in arbitrary units and normalize them to kilograms per cubic meter.

    Attributes:
        density: Particle density in kg/m^3 after conversion.
    """

    def __init__(self):
        """Initialize density mixin."""
        self.density = None

    @validate_inputs({"density": "positive"})
    def set_density(
        self,
        density: Union[float, NDArray[np.float64]],
        density_units: str,
    ):
        """Set the particle density in kilograms per cubic meter.

        Args:
            density: Positive density value to store.
            density_units: Units for ``density``. Defaults to ``"kg/m^3"`` and
                accepts any unit supported by the converter.

        Returns:
            BuilderDensityMixin: Self for method chaining.
        """
        if density_units == "kg/m^3":
            self.density = density
            return self
        self.density = density * get_unit_conversion(density_units, "kg/m^3")
        return self


class BuilderSurfaceTensionMixin:
    """Mixin that stores surface tension values in N/m.

    This mixin provides `set_surface_tension` so builders can accept surface
    tension inputs in arbitrary units and persist them in newtons per meter.

    Attributes:
        surface_tension: Surface tension in N/m after conversion.
    """

    def __init__(self):
        """Initialize surface tension mixin."""
        self.surface_tension = None

    @validate_inputs({"surface_tension": "positive"})
    def set_surface_tension(
        self,
        surface_tension: Union[float, NDArray[np.float64]],
        surface_tension_units: str,
    ):
        """Set the particle surface tension in newtons per meter.

        Args:
            surface_tension: Positive surface tension value to store.
            surface_tension_units: Units for ``surface_tension``. Defaults to
                ``"N/m"`` and accepts any supported conversion.

        Returns:
            BuilderSurfaceTensionMixin: Self for method chaining.
        """
        if surface_tension_units == "N/m":
            self.surface_tension = surface_tension
            return self
        self.surface_tension = surface_tension * get_unit_conversion(
            surface_tension_units, "N/m"
        )
        return self


class BuilderSurfaceTensionTableMixin:
    """Mixin that stores surface tension lookup tables in N/m.

    The mixin offers `set_surface_tension_table` so builders can normalize
    arrays of surface tension values to newtons per meter and retain them.

    Attributes:
        surface_tension_table: Array of surface tension values in N/m.
    """

    def __init__(self):
        """Initialize surface tension table mixin."""
        self.surface_tension_table = None

    @validate_inputs({"surface_tension_table": "positive"})
    def set_surface_tension_table(
        self,
        surface_tension_table: NDArray[np.float64],
        surface_tension_table_units: str = "N/m",
    ):
        """Normalize a surface tension table to newtons per meter.

        Args:
            surface_tension_table: Array of surface tension values.
            surface_tension_table_units: Units for the values. Defaults to
                ``"N/m"``.

        Returns:
            BuilderSurfaceTensionTableMixin: Self for method chaining.
        """
        table = np.asarray(surface_tension_table, dtype=np.float64)
        if surface_tension_table_units != "N/m":
            table = table * get_unit_conversion(
                surface_tension_table_units, "N/m"
            )
        self.surface_tension_table = table
        return self


class BuilderMolarMassMixin:
    """Mixin that stores molar mass values in kg/mol.

    The mixin provides `set_molar_mass` so builders can accept molar mass inputs
    in arbitrary units and normalize them to kilograms per mole.

    Attributes:
        molar_mass: Stored molar mass in kg/mol.
    """

    def __init__(self):
        """Initialize molar mass mixin."""
        self.molar_mass = None

    @validate_inputs({"molar_mass": "positive"})
    def set_molar_mass(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        molar_mass_units: str,
    ):
        """Set the molar mass in kilograms per mole.

        Args:
            molar_mass: Positive molar mass value to store.
            molar_mass_units: Units for ``molar_mass``. Defaults to
                ``"kg/mol"``.

        Returns:
            BuilderMolarMassMixin: Self for method chaining.
        """
        if molar_mass_units == "kg/mol":
            self.molar_mass = molar_mass
            return self
        self.molar_mass = molar_mass * get_unit_conversion(
            molar_mass_units, "kg/mol"
        )
        return self


class BuilderConcentrationMixin:
    """Mixin that stores concentration values in default units.

    The mixin exposes `set_concentration` so builders can accept concentration
    inputs in arbitrary units and persist them in the configured default units.

    Attributes:
        concentration: Stored concentration in `default_units`.
        default_units: Default units applied when no conversion is needed.
    """

    def __init__(self, default_units: str = "kg/m^3"):
        """Initialize concentration mixin."""
        self.concentration: Optional[Union[float, NDArray[np.float64]]] = None
        self.default_units = default_units if default_units else "kg/m^3"

    @validate_inputs({"concentration": "nonnegative"})
    def set_concentration(
        self,
        concentration: Union[float, NDArray[np.float64]],
        concentration_units: str,
    ):
        """Assign the concentration normalized to the default units.

        Args:
            concentration: Non-negative concentration value to store.
            concentration_units: Units for ``concentration``. Converted to
                ``self.default_units`` before storage.

        Returns:
            BuilderConcentrationMixin: Self for method chaining.
        """
        if concentration_units == self.default_units:
            self.concentration = concentration
            return self
        self.concentration = concentration * get_unit_conversion(
            concentration_units, self.default_units
        )
        return self


class BuilderChargeMixin:
    """Mixin that stores a particle charge count.

    Attributes:
        charge: Assigned charge in elemental charge units (dimensionless).
    """

    def __init__(self):
        """Initialize charge mixin."""
        self.charge = None

    def set_charge(
        self,
        charge: Union[float, NDArray[np.float64]],
        charge_units: Optional[str] = None,
    ):
        """Set the particle charge count.

        Args:
            charge: Numeric value describing elemental charges.
            charge_units: Optional units that are logged and ignored.

        Returns:
            BuilderChargeMixin: Self for method chaining.
        """
        if charge_units is not None:
            logger.warning("Ignoring units for charge parameter.")
        self.charge = charge
        return self


class BuilderPhaseIndexMixin:
    """Mixin that stores phase index assignments for species.

    Attributes:
        phase_index: Array mapping each species to a phase identifier.
    """

    def __init__(self):
        """Initialize phase index mixin."""
        self.phase_index = None

    def set_phase_index(
        self,
        phase_index: Union[Sequence[int], NDArray[np.int_]],
        phase_index_units: Optional[str] = None,
    ):
        """Assign phase membership indices to species.

        Args:
            phase_index: Sequence or array of integers describing phase IDs.
            phase_index_units: Optional units that are logged and ignored.

        Returns:
            BuilderPhaseIndexMixin: Self for method chaining.
        """
        if phase_index_units is not None:
            logger.warning("Ignoring units for phase index parameter.")
        self.phase_index = np.array(phase_index, dtype=int)
        return self


class BuilderMassMixin:
    """Mixin that stores particle mass in kilograms.

    Attributes:
        mass: Particle mass in kg after conversion.
    """

    def __init__(self):
        """Initialize mass mixin."""
        self.mass = None

    @validate_inputs({"mass": "nonnegative"})
    def set_mass(
        self,
        mass: Union[float, NDArray[np.float64]],
        mass_units: str,
    ):
        """Set the particle mass in kilograms.

        Args:
            mass: Non-negative mass value to store.
            mass_units: Units for ``mass``. Defaults to ``"kg"``.

        Returns:
            BuilderMassMixin: Self for method chaining.
        """
        if mass_units == "kg":
            self.mass = mass
            return self
        self.mass = mass * get_unit_conversion(mass_units, "kg")
        return self


class BuilderVolumeMixin:
    """Mixin that stores particle volume in cubic meters.

    Attributes:
        volume: Particle volume in m^3 after conversion.
    """

    def __init__(self):
        """Initialize volume mixin."""
        self.volume = None

    @validate_inputs({"volume": "nonnegative"})
    def set_volume(
        self,
        volume: Union[float, NDArray[np.float64]],
        volume_units: str,
    ):
        """Set the particle volume in cubic meters.

        Args:
            volume: Non-negative volume value to store.
            volume_units: Units for ``volume``. Defaults to ``"m^3"``.

        Returns:
            BuilderVolumeMixin: Self for method chaining.
        """
        if volume_units == "m^3":
            self.volume = volume
            return self
        self.volume = volume * get_unit_conversion(volume_units, "m^3")
        return self


class BuilderRadiusMixin:
    """Mixin that stores particle radius in meters.

    Attributes:
        radius: Particle radius in meters after conversion.
    """

    def __init__(self):
        """Initialize radius mixin."""
        self.radius = None

    @validate_inputs({"radius": "nonnegative"})
    def set_radius(
        self,
        radius: Union[float, NDArray[np.float64]],
        radius_units: str,
    ):
        """Set the particle radius in meters.

        Args:
            radius: Non-negative radius value to store.
            radius_units: Units for ``radius``. Defaults to ``"m"``.

        Returns:
            BuilderRadiusMixin: Self for method chaining.
        """
        if radius_units == "m":
            self.radius = radius
            return self
        self.radius = radius * get_unit_conversion(radius_units, "m")
        return self


class BuilderTemperatureMixin:
    """Mixin that stores temperatures in Kelvin.

    Attributes:
        temperature: Temperature in Kelvin after conversion.
    """

    def __init__(self):
        """Initialize temperature mixin."""
        self.temperature = None

    @validate_inputs({"temperature": "finite"})
    def set_temperature(self, temperature: float, temperature_units: str = "K"):
        """Set the temperature in Kelvin.

        Args:
            temperature: Finite temperature value.
            temperature_units: Units for ``temperature``. Defaults to ``"K"``.
                Accepts ``"degC"``, ``"degF"``, ``"degR"``, or ``"K"``.

        Returns:
            BuilderTemperatureMixin: Self for method chaining.

        Raises:
            ValueError: If the converted temperature is below absolute zero.
        """
        if temperature_units == "K":
            self.temperature = temperature
            return self
        self.temperature = get_unit_conversion(
            temperature_units, "K", temperature
        )
        return self


class BuilderTemperatureTableMixin:
    """Mixin that stores temperature lookup tables in Kelvin.

    Attributes:
        temperature_table: Array of temperatures in Kelvin after conversion.
    """

    def __init__(self):
        """Initialize temperature table mixin."""
        self.temperature_table = None

    @validate_inputs({"temperature_table": "finite"})
    def set_temperature_table(
        self,
        temperature_table: NDArray[np.float64],
        temperature_table_units: str = "K",
    ):
        """Assign a table of temperatures in Kelvin.

        Args:
            temperature_table: Array of temperature values.
            temperature_table_units: Units for the values. Defaults to ``"K"``.

        Returns:
            BuilderTemperatureTableMixin: Self for method chaining.
        """
        temps = np.asarray(temperature_table, dtype=np.float64)
        if temperature_table_units != "K":
            temps = np.array(
                [
                    get_unit_conversion(temperature_table_units, "K", float(t))
                    for t in temps
                ],
                dtype=np.float64,
            )
        self.temperature_table = temps
        return self


class BuilderPressureMixin:
    """Mixin that stores total gas pressure in pascals.

    Attributes:
        pressure: Total pressure in Pa after conversion.
    """

    def __init__(self):
        """Initialize pressure mixin."""
        self.pressure = None

    @validate_inputs({"pressure": "nonnegative"})
    def set_pressure(
        self,
        pressure: Union[float, NDArray[np.float64]],
        pressure_units: str,
    ):
        """Set the total pressure in pascals.

        Args:
            pressure: Non-negative pressure value to store.
            pressure_units: Units for ``pressure``. Defaults to ``"Pa"``.

        Returns:
            BuilderPressureMixin: Self for method chaining.
        """
        if pressure_units == "Pa":
            self.pressure = pressure
            return self
        self.pressure = pressure * get_unit_conversion(pressure_units, "Pa")
        return self


class BuilderLognormalMixin:
    """Mixin that stores lognormal distribution properties for radius.

    Attributes:
        mode: Mode radii in meters.
        number_concentration: Number concentration in 1/m^3.
        geometric_standard_deviation: Geometric standard deviation values.
    """

    def __init__(self):
        """Initialize lognormal distribution mixin."""
        self.mode = None
        self.number_concentration = None
        self.geometric_standard_deviation = None

    @validate_inputs({"mode": "positive"})
    def set_mode(
        self,
        mode: NDArray[np.float64],
        mode_units: str,
    ):
        """Set lognormal mode radii in meters.

        Args:
            mode: Array of modal radius values.
            mode_units: Units for ``mode``. Defaults to ``"m"``.

        Returns:
            BuilderLognormalMixin: Self for method chaining.
        """
        if mode_units == "m":
            self.mode = mode
            return self
        self.mode = mode * get_unit_conversion(mode_units, "m")
        return self

    @validate_inputs({"geometric_standard_deviation": "positive"})
    def set_geometric_standard_deviation(
        self,
        geometric_standard_deviation: NDArray[np.float64],
        geometric_standard_deviation_units: Optional[str] = None,
    ):
        """Set the geometric standard deviation values.

        Args:
            geometric_standard_deviation: Dimensionless geometric std. dev.
            geometric_standard_deviation_units: Optional units that are logged
                and ignored for interface consistency.

        Returns:
            BuilderLognormalMixin: Self for method chaining.
        """
        if geometric_standard_deviation_units is not None:
            logger.warning("Ignoring units for surface strategy parameter.")
        self.geometric_standard_deviation = geometric_standard_deviation
        return self

    @validate_inputs({"number_concentration": "positive"})
    def set_number_concentration(
        self,
        number_concentration: NDArray[np.float64],
        number_concentration_units: str,
    ):
        """Set the number concentration in 1/m^3.

        Args:
            number_concentration: Array of number concentration values.
            number_concentration_units: Units for ``number_concentration``. Must
                be ``"1/m^3"`` or ``"m^-3"`` for direct storage.

        Returns:
            BuilderLognormalMixin: Self for method chaining.
        """
        if number_concentration_units in {"1/m^3", "m^-3"}:
            self.number_concentration = number_concentration
            return self
        self.number_concentration = number_concentration * get_unit_conversion(
            number_concentration_units, "1/m^3"
        )
        return self


class BuilderParticleResolvedCountMixin:
    """Mixin that stores a particle-resolved count.

    Attributes:
        particle_resolved_count: Number of particles to resolve.
    """

    def __init__(self):
        """Initialize particle-resolved count mixin."""
        self.particle_resolved_count = None

    @validate_inputs({"particle_resolved_count": "positive"})
    def set_particle_resolved_count(
        self,
        particle_resolved_count: int,
        particle_resolved_count_units: Optional[str] = None,
    ):
        """Assign the particle-resolved count.

        Args:
            particle_resolved_count: Positive integer number of particles.
            particle_resolved_count_units: Optional units that are logged and
                ignored.

        Returns:
            BuilderParticleResolvedCountMixin: Self for method chaining.
        """
        if particle_resolved_count_units is not None:
            logger.warning("Ignoring units for particle resolved count.")
        self.particle_resolved_count = particle_resolved_count
        return self


class BuilderWallEddyDiffusivityMixin:
    """Mixin for setting wall eddy diffusivity.

    Adds ``set_wall_eddy_diffusivity`` for builders that require wall eddy
    diffusivity and stores the value in 1/s.

    Attributes:
        wall_eddy_diffusivity: Wall eddy diffusivity in 1/s.
    """

    def __init__(self):
        """Initialize wall eddy diffusivity mixin."""
        self.wall_eddy_diffusivity: Optional[float] = None

    @validate_inputs({"wall_eddy_diffusivity": "positive"})
    def set_wall_eddy_diffusivity(
        self,
        wall_eddy_diffusivity: float,
        wall_eddy_diffusivity_units: str = "1/s",
    ) -> "BuilderWallEddyDiffusivityMixin":
        """Set wall eddy diffusivity in 1/s.

        Args:
            wall_eddy_diffusivity: Wall eddy diffusivity value.
            wall_eddy_diffusivity_units: Units for the value. Defaults to
                "1/s".

        Returns:
            BuilderWallEddyDiffusivityMixin: Self for method chaining.

        Raises:
            ValueError: If ``wall_eddy_diffusivity`` is not positive.
        """
        if wall_eddy_diffusivity_units in {"1/s", "s^-1"}:
            self.wall_eddy_diffusivity = wall_eddy_diffusivity
            return self
        self.wall_eddy_diffusivity = cast(
            float,
            get_unit_conversion(
                wall_eddy_diffusivity_units,
                "1/s",
                wall_eddy_diffusivity,
            ),
        )
        return self


class BuilderChamberRadiusMixin:
    """Mixin for setting spherical chamber radius.

    Adds ``set_chamber_radius`` for builders that require spherical chamber
    geometry and stores the value in meters.

    Attributes:
        chamber_radius: Chamber radius in meters.
    """

    def __init__(self):
        """Initialize chamber radius mixin."""
        self.chamber_radius: Optional[float] = None

    @validate_inputs({"chamber_radius": "positive"})
    def set_chamber_radius(
        self,
        chamber_radius: float,
        chamber_radius_units: str = "m",
    ) -> "BuilderChamberRadiusMixin":
        """Set the chamber radius in meters.

        Args:
            chamber_radius: Chamber radius value.
            chamber_radius_units: Units for the value. Defaults to "m".

        Returns:
            BuilderChamberRadiusMixin: Self for method chaining.

        Raises:
            ValueError: If ``chamber_radius`` is not positive.
        """
        if chamber_radius_units in {"m", "meter", "meters"}:
            self.chamber_radius = chamber_radius
            return self
        if chamber_radius_units in {"cm", "centimeter", "centimeters"}:
            self.chamber_radius = chamber_radius * 0.01
            return self
        self.chamber_radius = cast(
            float,
            get_unit_conversion(
                chamber_radius_units,
                "m",
                chamber_radius,
            ),
        )
        return self


class BuilderChamberDimensionsMixin:
    """Mixin for setting rectangular chamber dimensions.

    Adds ``set_chamber_dimensions`` for builders that require rectangular
    chamber geometry and stores dimensions in meters.

    Attributes:
        chamber_dimensions: Tuple of (length, width, height) in meters.
    """

    def __init__(self):
        """Initialize chamber dimensions mixin."""
        self.chamber_dimensions: Optional[Tuple[float, float, float]] = None

    @validate_inputs({"chamber_dimensions": "positive"})
    def set_chamber_dimensions(
        self,
        chamber_dimensions: Tuple[float, float, float],
        chamber_dimensions_units: str = "m",
    ) -> "BuilderChamberDimensionsMixin":
        """Set rectangular chamber dimensions in meters.

        Args:
            chamber_dimensions: Tuple of (length, width, height).
            chamber_dimensions_units: Units for the values. Defaults to "m".

        Returns:
            BuilderChamberDimensionsMixin: Self for method chaining.

        Raises:
            ValueError: If ``chamber_dimensions`` is not length three or any
                value is non-positive.
        """
        if len(chamber_dimensions) != 3:
            raise ValueError(
                "chamber_dimensions must contain three values: "
                "(length, width, height)."
            )
        length, width, height = chamber_dimensions
        if chamber_dimensions_units in {"m", "meter", "meters"}:
            self.chamber_dimensions = (
                length,
                width,
                height,
            )
            return self
        if chamber_dimensions_units in {"cm", "centimeter", "centimeters"}:
            self.chamber_dimensions = (
                length * 0.01,
                width * 0.01,
                height * 0.01,
            )
            return self
        self.chamber_dimensions = (
            cast(
                float,
                get_unit_conversion(
                    chamber_dimensions_units,
                    "m",
                    length,
                ),
            ),
            cast(
                float,
                get_unit_conversion(
                    chamber_dimensions_units,
                    "m",
                    width,
                ),
            ),
            cast(
                float,
                get_unit_conversion(
                    chamber_dimensions_units,
                    "m",
                    height,
                ),
            ),
        )
        return self


class BuilderWallPotentialMixin:
    """Mixin that stores electrostatic wall potential in volts.

    Stores electrostatic wall potential used by charged wall loss strategies.
    A zero potential still permits image-charge enhancement when particle
    charge is non-zero.

    Attributes:
        wall_potential: Electrostatic wall potential in volts.
    """

    def __init__(self):
        """Initialize wall potential mixin."""
        self.wall_potential: float = 0.0

    @validate_inputs({"wall_potential": "finite"})
    def set_wall_potential(
        self, wall_potential: float, wall_potential_units: str = "V"
    ) -> "BuilderWallPotentialMixin":
        """Set wall potential in volts.

        Args:
            wall_potential: Electrostatic wall potential in volts. Zero keeps
                the chamber neutral but still allows image-charge effects for
                charged particles.
            wall_potential_units: Units for ``wall_potential``. Defaults to
                "V".

        Returns:
            BuilderWallPotentialMixin: Self for method chaining.
        """
        if wall_potential_units in {"V", "volt", "volts"}:
            self.wall_potential = wall_potential
            return self
        self.wall_potential = cast(
            float,
            get_unit_conversion(wall_potential_units, "V", wall_potential),
        )
        return self


class BuilderWallElectricFieldMixin:
    """Mixin that stores wall electric field magnitudes in V/m.

    Supports optional drift terms for charged wall loss strategies.
    Accepts a scalar magnitude for spherical chambers or a three-component
    tuple for rectangular geometries. A value of 0.0 disables field-driven
    drift.

    Attributes:
        wall_electric_field: Electric field magnitude (scalar or tuple) in V/m.
    """

    def __init__(self):
        """Initialize wall electric field mixin."""
        self.wall_electric_field: Union[float, Tuple[float, float, float]] = 0.0

    def _validate_wall_electric_field(
        self, wall_electric_field: Union[float, Tuple[float, float, float]]
    ) -> Union[float, Tuple[float, float, float]]:
        """Validate and normalize wall electric field inputs.

        Args:
            wall_electric_field: Scalar magnitude or ``(Ex, Ey, Ez)`` tuple.

        Returns:
            Union[float, Tuple[float, float, float]]: Validated scalar or tuple.

        Raises:
            ValueError: If tuple length is not three or entries are not finite.
        """
        if isinstance(wall_electric_field, tuple):
            if len(wall_electric_field) != 3:
                raise ValueError(
                    "wall_electric_field tuple must be length three for "
                    "rectangular geometries."
                )
            if not np.all(np.isfinite(wall_electric_field)):
                raise ValueError("wall_electric_field entries must be finite.")
            return wall_electric_field
        if not np.isfinite(wall_electric_field):
            raise ValueError("wall_electric_field must be finite.")
        return float(wall_electric_field)

    def set_wall_electric_field(
        self,
        wall_electric_field: Union[float, Tuple[float, float, float]],
        wall_electric_field_units: str = "V/m",
    ) -> "BuilderWallElectricFieldMixin":
        """Set wall electric field magnitude in V/m.

        Args:
            wall_electric_field: Scalar magnitude for spherical chambers or a
                three-element ``(Ex, Ey, Ez)`` tuple for rectangular chambers.
                Use ``0.0`` to disable field-driven drift.
            wall_electric_field_units: Units for ``wall_electric_field``.
                Defaults to "V/m".

        Returns:
            BuilderWallElectricFieldMixin: Self for method chaining.

        Raises:
            ValueError: If the tuple is not length three or any entry is not
                finite.
        """
        validated_field = self._validate_wall_electric_field(
            wall_electric_field
        )
        if isinstance(validated_field, tuple):
            if wall_electric_field_units in {"V/m", "volt/m", "volts/m"}:
                self.wall_electric_field = cast(
                    Tuple[float, float, float], validated_field
                )
                return self
            converted_field = (
                cast(
                    float,
                    get_unit_conversion(
                        wall_electric_field_units, "V/m", validated_field[0]
                    ),
                ),
                cast(
                    float,
                    get_unit_conversion(
                        wall_electric_field_units, "V/m", validated_field[1]
                    ),
                ),
                cast(
                    float,
                    get_unit_conversion(
                        wall_electric_field_units, "V/m", validated_field[2]
                    ),
                ),
            )
            self.wall_electric_field = cast(
                Tuple[float, float, float], converted_field
            )
            return self
        if wall_electric_field_units in {"V/m", "volt/m", "volts/m"}:
            self.wall_electric_field = validated_field
            return self
        self.wall_electric_field = cast(
            float,
            get_unit_conversion(
                wall_electric_field_units, "V/m", validated_field
            ),
        )
        return self


class BuilderDistributionTypeMixin:
    """Mixin for setting distribution type.

    Adds ``set_distribution_type`` for builders requiring a distribution type.
    Defaults to "discrete" and validates against supported types.

    Attributes:
        distribution_type: Distribution type string.
    """

    def __init__(self):
        """Initialize distribution type mixin."""
        self.distribution_type: str = "discrete"

    def set_distribution_type(self, distribution_type: str):
        """Set the distribution type.

        Args:
            distribution_type: One of "discrete", "continuous_pdf", or
                "particle_resolved".

        Returns:
            BuilderDistributionTypeMixin: Self for method chaining.

        Raises:
            ValueError: If ``distribution_type`` is not supported.
        """
        valid_types = ["discrete", "continuous_pdf", "particle_resolved"]
        if distribution_type not in valid_types:
            raise ValueError(
                "distribution_type must be one of "
                f"{valid_types}, got '{distribution_type}'."
            )
        self.distribution_type = distribution_type
        return self
