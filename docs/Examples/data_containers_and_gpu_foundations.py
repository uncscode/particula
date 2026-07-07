"""Run the published data-container and Warp-transfer example.

This wrapper preserves the top-level command path documented for the example
while forwarding execution to the topic-directory implementation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    """Load and run the topic-directory example implementation.

    Raises:
        RuntimeError: If the implementation module cannot be loaded from the
            topic-directory example path.
    """
    example_path = (
        Path(__file__).resolve().parent
        / "Data_Containers"
        / "data_containers_and_gpu_foundations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "data_containers_and_gpu_foundations_impl",
        example_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load example module from {example_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
