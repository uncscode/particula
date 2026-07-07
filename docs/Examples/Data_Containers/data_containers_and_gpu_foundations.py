"""Support the data-container guide by delegating to the canonical entrypoint.

The published runnable example lives at
``docs/Examples/data_containers_and_gpu_foundations.py``. This topic-directory
module remains for guide-local context and forwards execution to that
entrypoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from data_containers_and_gpu_foundations import main, run_example

__all__ = ["main", "run_example"]


if __name__ == "__main__":
    main()
