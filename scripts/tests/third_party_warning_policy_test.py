"""Keep the chemicals/pandas compatibility warning exception narrowly scoped."""

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
