"""Profile-driven instruments and validated named readers."""

from dataclasses import dataclass

from particula.data import (
    Measurement,
    MobilityNumberSpectrum,
    ParticleCountRate,
    describe_profile,
    read_measurement,
)
from particula.data.profile import ProfileSource


@dataclass
class Instrument:
    """An instrument defined by a built-in or user-supplied profile."""

    profile: ProfileSource

    def __post_init__(self):
        """Resolve and validate the instrument profile."""
        self.profile = describe_profile(self.profile)

    def read(self, source, **limits) -> Measurement:
        """Read the instrument's measurements and source diagnostics."""
        return read_measurement(source, profile=self.profile, **limits)


@dataclass
class ParticleCounter(Instrument):
    """A number-counting instrument, defaulting to the 3010 logger profile."""

    profile: ProfileSource = "cpc-3010"


@dataclass
class MobilitySizer(Instrument):
    """An electrical-mobility sizer, defaulting to the AIM number export."""

    profile: ProfileSource = "smps-aim"


def read_particle_counter(
    source, *, profile="cpc-3010", **limits
) -> ParticleCountRate:
    """Read a count-rate product and validate the measured quantity."""
    result = ParticleCounter(profile).read(source, **limits)
    if not isinstance(result, ParticleCountRate):
        raise ValueError("The supplied profile did not produce a count rate")
    return result


def read_mobility_sizer(
    source, *, profile="smps-aim", **limits
) -> MobilityNumberSpectrum:
    """Read a number spectrum with electrical-mobility diameter coordinates."""
    result = MobilitySizer(profile).read(source, **limits)
    if not isinstance(result, MobilityNumberSpectrum):
        raise ValueError("Expected a mobility number spectrum")
    return result
