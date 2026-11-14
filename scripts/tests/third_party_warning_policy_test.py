"""Keep dependency compatibility warnings narrowly scoped."""

import runpy
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MESSAGE = (
    "The 'float_precision' argument is deprecated. "
    "Use the default float precision instead."
)


def test_chemical_csv_warning_filter_preserves_other_errors():
    """Only the known warning from the upstream CSV loader is exempted."""
    with warnings.catch_warnings():
        configure = runpy.run_path(str(ROOT / "conftest.py"))[
            "_filter_expected_warnings"
        ]
        warnings.simplefilter("error")
        configure()
        warnings.warn_explicit(
            MESSAGE,
            DeprecationWarning,
            "data_reader.py",
            112,
            module="chemicals.data_reader",
        )
        for module, message in (
            ("particula.example", MESSAGE),
            ("chemicals.data_reader", "A different deprecation"),
        ):
            with pytest.raises(DeprecationWarning):
                warnings.warn_explicit(
                    message,
                    DeprecationWarning,
                    "example.py",
                    1,
                    module=module,
                )


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
