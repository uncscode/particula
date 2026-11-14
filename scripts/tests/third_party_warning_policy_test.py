"""Keep known Warp ctypes compatibility warnings narrower than scientific ones."""

import runpy
import warnings
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "record", ["APICCudaCompileRecipe", "APICLaunchPtrLocation"]
)
def test_new_warp_ctypes_warnings_are_module_scoped(record):
    """Ignore only the known dependency warning, retaining local deprecations."""
    message = f"Due to '_pack_', the '{record}' Structure will use memory layout compatible with MSVC (Windows)."
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        config = Path(__file__).resolve().parents[2] / "conftest.py"
        runpy.run_path(str(config))["_filter_expected_warnings"]()
        warnings.warn_explicit(
            message,
            DeprecationWarning,
            "types.py",
            1,
            module="warp._src.apic.types",
        )
        with pytest.raises(DeprecationWarning):
            warnings.warn_explicit(
                message,
                DeprecationWarning,
                "local.py",
                1,
                module="particula.local",
            )
        with pytest.raises(DeprecationWarning):
            warnings.warn_explicit(
                "Unrelated deprecation",
                DeprecationWarning,
                "types.py",
                1,
                module="warp._src.apic.types",
            )
