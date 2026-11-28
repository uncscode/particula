"""Import particle number spectra resolved by electrical-mobility diameter."""

from dataclasses import dataclass
from os import PathLike

from particula.data.measurement import MobilityNumberSpectrum
from particula.data.profile import ProfileSource
from particula.instruments.instrument import Instrument


@dataclass
class MobilitySizer(Instrument):
    """Read an electrical-mobility sizer export with a configurable profile.

    Args:
        profile: Import profile accepted by Instrument. The default AIM
            number export uses mobility diameter in metres and number
            concentration per decade of diameter in particles per cubic metre.
    """

    profile: ProfileSource = "smps-aim"


def read_mobility_sizer(
    source: str | PathLike[str],
    *,
    profile: ProfileSource = "smps-aim",
    **limits: int,
) -> MobilityNumberSpectrum:
    """Read number spectra and verify the electrical-mobility diameter axis.

    Args:
        source: Path to the delimited sizer export.
        profile: Built-in profile ID, JSON or YAML file path, or profile
            mapping describing the spectrum and diameter axis.
        **limits: Optional MeasurementReader chunk, record and batch limits.

    Returns:
        Number concentration per decade of mobility diameter, with each
        scan linked to its diameter grid and declared output units.

    Raises:
        ValueError: The profile produces a different measurement type.
    """
    result = MobilitySizer(profile).read(source, **limits)
    if not isinstance(result, MobilityNumberSpectrum):
        raise ValueError("Expected a mobility number spectrum")
    return result
