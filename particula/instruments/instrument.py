"""Read measurements from an instrument with a declared import profile."""

from dataclasses import dataclass
from os import PathLike

from particula.data.measurement import Measurement
from particula.data.profile import ProfileSource, describe_profile
from particula.data.storage import read_measurement


@dataclass
class Instrument:
    """Read delimited exports that share one profile.

    Args:
        profile: Built-in profile ID, JSON or YAML file path, or profile
            mapping. Construction validates and stores a copy of the profile.

    Raises:
        ProfileError: The profile cannot be loaded or fails validation.
    """

    profile: ProfileSource

    def __post_init__(self) -> None:
        """Resolve the supplied profile before reading any sources."""
        self.profile = describe_profile(self.profile)

    def read(self, source: str | PathLike[str], **limits: int) -> Measurement:
        """Read one export into memory using this instrument's profile.

        Args:
            source: Path to the delimited instrument export.
            **limits: Optional chunk, record and batch limits accepted by
                MeasurementReader.

        Returns:
            Measurement with scans in the profile's output units, source
            diagnostics and the import manifest.
        """
        return read_measurement(source, profile=self.profile, **limits)
