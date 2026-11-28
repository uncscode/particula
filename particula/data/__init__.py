"""Import unit-described measurements with the optional data extra."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from particula.data.measurement import (
        Measurement,
        MobilityNumberSpectrum,
        ParticleCountRate,
    )
    from particula.data.migration import migrate_settings
    from particula.data.profile import (
        ProfileError,
        describe_profile,
        validate_profile,
    )
    from particula.data.reader import MeasurementReader, inspect_source
    from particula.data.storage import (
        import_to_parquet,
        open_measurement,
        read_measurement,
    )

_MODULES = {
    "Measurement": "measurement",
    "MobilityNumberSpectrum": "measurement",
    "ParticleCountRate": "measurement",
    "migrate_settings": "migration",
    "ProfileError": "profile",
    "describe_profile": "profile",
    "validate_profile": "profile",
    "MeasurementReader": "reader",
    "inspect_source": "reader",
    "import_to_parquet": "storage",
    "open_measurement": "storage",
    "read_measurement": "storage",
}

__all__ = [
    "Measurement",
    "MeasurementReader",
    "MobilityNumberSpectrum",
    "ParticleCountRate",
    "ProfileError",
    "migrate_settings",
    "describe_profile",
    "import_to_parquet",
    "inspect_source",
    "open_measurement",
    "read_measurement",
    "validate_profile",
]


def __getattr__(name: str):
    """Load a public data feature and its dependencies on first use."""
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
