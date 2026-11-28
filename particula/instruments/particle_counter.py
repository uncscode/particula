"""Import particle detector count rates and associated logger channels."""

from dataclasses import dataclass
from os import PathLike

from particula.data.measurement import ParticleCountRate
from particula.data.profile import ProfileSource
from particula.instruments.instrument import Instrument


@dataclass
class ParticleCounter(Instrument):
    """Read a particle counter export with a configurable profile.

    Args:
        profile: Import profile accepted by Instrument. The default 3010
            logger profile reports detector count rate in counts per second
            and logger temperature in kelvin.
    """

    profile: ProfileSource = "cpc-3010"


def read_particle_counter(
    source: str | PathLike[str],
    *,
    profile: ProfileSource = "cpc-3010",
    **limits: int,
) -> ParticleCountRate:
    """Read a particle counter export and verify its count-rate field.

    Args:
        source: Path to the delimited counter export.
        profile: Built-in profile ID, JSON or YAML file path, or profile
            mapping describing the detector channels and their units.
        **limits: Optional MeasurementReader chunk, record and batch limits.

    Returns:
        Count-rate scans in the profile's output units with source diagnostics.

    Raises:
        ValueError: The profile produces a different measurement type.
    """
    result = ParticleCounter(profile).read(source, **limits)
    if not isinstance(result, ParticleCountRate):
        raise ValueError("The supplied profile did not produce a count rate")
    return result
