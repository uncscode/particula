# CUDA optics qualification

Measured 2026-11-15T15:44:41.266102+00:00 on NVIDIA GeForce RTX 5060 (8 GiB),
AMD Ryzen AI 9 HX 370 CPU, Python 3.12.12 and Warp 1.11.1.
Warp reported CUDA Toolkit 12.9 and driver API 13.4. The run used source revision
`d523321593a5669d8eabc4c421ea77eba543be6a` with unchanged package source
SHA-256 `a2faecc9e7fb5cff47d09509cf98cc08b9e69dcda39c0b271c03295e2bee85cb`.
The example source was newly authored; its exact executed bytes, execution
fingerprint, installed distribution metadata and dependency versions are in
`website/results/cuda-optics/manifest.json` and `source.py`.
The installed Particula distribution metadata says 0.2.14; this result uses
the recorded development checkout, not a released 0.2.14 optics API.

## Reproduction and timing boundaries

Run the complete [CUDA optics example](../../../examples/optics/cuda_optics.py):

```bash
python scripts/docs_examples.py refresh cuda-optics
```

Use the selected environment and checkout on PYTHONPATH when reusing another
worktree's environment. This run used a separate empty Warp cache under
`XDG_CACHE_HOME=/tmp/optics-final-evidence-cache`. Other validation workloads
had completed before the retained timing run. Background OS load and frequency
variation were not controlled; this is a local workload measurement, not a
hardware-wide throughput claim.

Module compilation/cache loading is measured explicitly with `wp.load_module`
before preparation. Upload, preparation and download are separate intervals.
CUDA execution uses five warm repeated enqueues, each followed by device
synchronization; status reads and downloads are outside that timing. NumPy CPU
execution uses three complete calls. No graph capture is used. CPU and CUDA
algorithms share physical formulas but are different implementations; no Warp
CPU speed claim is made. Raw repetitions are retained in the saved plot data.

| Cases | Maximum x | NumPy median [ms] | CUDA median [ms] | Upload [ms] | Preparation [ms] | Download [ms] |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 10 | 2.976 | 0.1471 | 0.412 | 0.468 | 0.02611 |
| 512 | 100 | 37.93 | 1.365 | 0.1483 | 0.2006 | 0.02591 |
| 2048 | 1000 | 908.1 | 21.23 | 0.2386 | 0.4668 | 0.05286 |

Module compile/cache-load observations [s]: mie=1.567, angular=0.5666, _bulk=0.5394.

## Numerical qualification

Each batch uses logarithmically spaced x from 0.1 to its maximum, index
1.5+0.1i, float64 and 256-case coefficient tiles. Every returned efficiency,
asymmetry and validity value is compared with NumPy at rtol=1e-8, atol=1e-12.
Output buffer identity is checked across repeated executions. The timing
study is not the x=10000 qualification; see the independent
[reference and angular checks](reference-corpus.md) for that evidence.

The cloud calculation uses 512 radius bins from 0.5 to 20 micrometres,
100 particles/cm³, four wavelengths (450,550,700,900 nm), index 1.33+1e-8i,
61 angles and 128-particle/32-angle tiles. All nine bulk output fields and
every full Mueller element are checked, not only plotted S11. Scalar parity
uses rtol=1e-8, atol=1e-12; angular error is scaled by each case's peak matrix
entry and bounded by 1e-8. The measured maximum scaled Mueller error was
1.09852e-15.

The saved JSON includes only timings, plotted optical summaries and residuals.
It excludes particle state and full validation arrays. Two saved figures were
visually inspected. Optical numerical tests additionally cover Warp CPU,
native CUDA, failures, zero cases, medium normalization, polarization,
chunk independence and independent angular integrals through x=10000.

## Delivery validation

Final numerical source is commit `baf393b18`; subsequent commits before this
measurement only update recorded evidence. Full source tests with native CUDA
passed: 10,408 passed, 14 skipped, one expected failure, 91.36% total coverage
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
