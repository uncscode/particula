"""Import unit-described measurements with the optional data extra."""

from typing import TYPE_CHECKING

from particula.data._exports import load_export as _load_export

__getattr__ = _load_export

if TYPE_CHECKING:
    from particula.data.clock import correct_clock
    from particula.data.collection import CampaignEvent, MeasurementCollection
    from particula.data.deployment import Calibration, Deployment
    from particula.data.measurement import (
        Measurement,
        MobilityNumberSpectrum,
        ParticleCountRate,
    )
    from particula.data.migration import migrate_settings
    from particula.data.process import MeasurementProcess
    from particula.data.product import DerivedMeasurement, FieldContract
    from particula.data.profile import (
        ProfileError,
        describe_profile,
        validate_profile,
    )
    from particula.data.reader import (
        BatchReader,
        MeasurementReader,
        inspect_source,
    )
    from particula.data.recipe import (
        Recipe,
        describe_recipe,
        run_recipe,
        validate_recipe,
    )
    from particula.data.registry import OperationRegistry
    from particula.data.snapshot import save_measurement
    from particula.data.storage import (
        import_to_parquet,
        open_measurement,
        read_measurement,
    )
    from particula.data.windows import WindowComparison

__all__ = [
    "WindowComparison",
    "BatchReader",
    "correct_clock",
    "Calibration",
    "Deployment",
    "DerivedMeasurement",
    "FieldContract",
    "OperationRegistry",
    "MeasurementProcess",
    "Recipe",
    "describe_recipe",
    "run_recipe",
    "validate_recipe",
    "save_measurement",
    "CampaignEvent",
    "MeasurementCollection",
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
