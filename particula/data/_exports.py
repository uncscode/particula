"""Resolve public data imports without loading optional packages at startup."""

from importlib import import_module
from sys import modules
from typing import Any

_MODULES = {
    "WindowComparison": "windows",
    "BatchReader": "reader",
    "correct_clock": "clock",
    "Calibration": "deployment",
    "Deployment": "deployment",
    "DerivedMeasurement": "product",
    "FieldContract": "product",
    "OperationRegistry": "registry",
    "MeasurementProcess": "process",
    "Recipe": "recipe",
    "describe_recipe": "recipe",
    "run_recipe": "recipe",
    "validate_recipe": "recipe",
    "save_measurement": "snapshot",
    "CampaignEvent": "collection",
    "MeasurementCollection": "collection",
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


def load_export(name: str) -> Any:
    """Load a public object on first access and cache it on the data package."""
    if name not in _MODULES:
        raise AttributeError(
            f"module 'particula.data' has no attribute {name!r}"
        )
    value = getattr(import_module(f"particula.data.{_MODULES[name]}"), name)
    setattr(modules["particula.data"], name, value)
    return value
