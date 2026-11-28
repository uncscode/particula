"""Import unit-described measurements with the optional data extra."""

from typing import TYPE_CHECKING

from particula.data._exports import load_export as _load_export

__getattr__ = _load_export

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
