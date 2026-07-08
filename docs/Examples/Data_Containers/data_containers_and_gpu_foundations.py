"""Support the data-container guide by delegating to the canonical entrypoint.

The published runnable example lives at
``docs/Examples/data_containers_and_gpu_foundations.py``. This topic-directory
module remains for guide-local context and forwards execution to that
entrypoint.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

CANONICAL_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_containers_and_gpu_foundations.py"
)


def _load_canonical_module():
    """Load the canonical example module from its file path."""
    spec = importlib.util.spec_from_file_location(
        "particula_docs_data_containers_example",
        CANONICAL_EXAMPLE_PATH,
    )
    if spec is None or spec.loader is None:
        msg = f"Unable to load canonical example: {CANONICAL_EXAMPLE_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CANONICAL_MODULE = _load_canonical_module()
main = _CANONICAL_MODULE.main
run_example = _CANONICAL_MODULE.run_example

__all__ = ["main", "run_example"]


if __name__ == "__main__":
    main()
