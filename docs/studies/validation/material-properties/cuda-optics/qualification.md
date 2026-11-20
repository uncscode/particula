# CUDA optics delivery qualification

The [scientific comparison](index.md) describes the workload, assumptions and results supporting these decisions.

## Delivery validation

The pre-review numerical baseline at `baf393b18` passed full source tests
with native CUDA: 10,408 passed, 14 skipped, one expected failure, 91.36% total coverage
(required 80%). Command: `python -m pytest -Werror --cov=particula
--cov-report=term-missing --cov-fail-under=80`. The report includes Python
wrappers and tests; Python line coverage does not instrument compiled Warp
kernel bodies. Device execution is established by the numerical CUDA tests.

The installed-wheel optics suite passed all 251 tests on CPU and native CUDA
in a separate environment with NumPy 2.5.4, SciPy 1.18.1 and Warp 1.19.0.
PyMieScatt, miepython and mpmath were absent. The wheel includes offline
reference fixtures. Documentation bundle verification passed isolated installs
and an offline source-archive rebuild (210 files, 2,514,813 documentation bytes).

Repository-wide Ruff, formatting and mypy passed. Infrastructure tests passed
298 tests; example integration passed 235 with one skip. The registered CPU
example campaign passed, with configured long/CUDA skips; both new optics
examples were separately refreshed in full on their required devices.
These results are development qualification, not release or merge approval.

Review follow-up `4ce93ae09` binds replay to the prepared CUDA stream in both
workspaces. All 255 optics tests passed on CPU/native CUDA, including four
new capture/replay cases with a different current stream. The full CUDA example
was refreshed above after this correction. Mypy passed against Warp 1.19.0;
the earlier installed-wheel and full-suite counts describe the pre-review
baseline, whose scientific formulas are unchanged.

Initialization follow-up `37d197c5c` places workspace buffer fills on the
prepared stream as well. The optics suite passed 259 tests on CPU/native CUDA;
all 11 workspace tests also passed with Warp 1.19.0. Four added cases check
actual fill-stream selection and immediate execution without a device-wide
barrier, including preservation of failure flags. Ruff, formatting and mypy
passed. The complete CUDA example ran successfully again with separate local
outputs; the published timing table above remains the recorded `4ce93ae09`
measurement, not a measurement of this follow-up.
