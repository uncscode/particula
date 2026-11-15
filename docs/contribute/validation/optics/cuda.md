# CUDA optics qualification

Measured 2026-11-15T15:25:50.621683+00:00 on NVIDIA GeForce RTX 5060 (8 GiB),
AMD Ryzen AI 9 HX 370 CPU, Python 3.12.12 and Warp 1.11.1.
Warp reported CUDA Toolkit 12.9 and driver API 13.4. The run used source revision
`1b9f4f07967d2daaefa93b68eb096572a7683bee` with unchanged package source
SHA-256 `c8141ca8ddae25d5a2987d44178c80a05fe0817b714350d8246666d5614ca6d1`.
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
`XDG_CACHE_HOME=/tmp/optics-example-isolated-cache`. Other validation workloads
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
| 64 | 10 | 1.987 | 0.144 | 0.4603 | 0.4473 | 0.02318 |
| 512 | 100 | 35.41 | 1.34 | 0.1529 | 0.1745 | 0.02257 |
| 2048 | 1000 | 1045 | 20.95 | 0.2598 | 0.4696 | 0.05348 |

Module compile/cache-load observations [s]: mie=0.6056, angular=0.497, _bulk=0.4513.

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

## Remaining delivery gates

This record does not establish full-package coverage, clean-wheel validation,
completion of every input/failure audit, release readiness or PR approval.
Those gates are tracked separately in the implementation plan and final review.
