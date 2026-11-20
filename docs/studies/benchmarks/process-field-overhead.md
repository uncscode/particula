# Cost of custom processes and particle fields on the RTX 5060

On an NVIDIA GeForce RTX 5060, eight boxes of 256 slots each underwent native dilution at 0.01 s⁻¹ with a 0.001 s timestep. The extension-free workload compares pre-feature revision `a73fa19741885844ffeebbf76723a8b60ab39223` with the feature revision. Each resident measurement used 20 warm-up steps and seven batches of 2,000 steps. The explicit-transfer case used 200 steps per batch. Preparation and compilation were outside the timing boundary.

## Timing results

Measured on revision `8d108f454` with Warp 1.11.1, NumPy 2.3.4 and Python
3.12.12. This revision has the same library bytes as the full package run.
The [complete samples and source hashes](https://github.com/uncscode/particula/blob/main/docs/data/gpu/process-field-extensions.json)
retain seven samples per case and the benchmark script digest. Other GPU tests
and saved-example validation finished before these measurements; each case
checked for competing GPU processes before starting.

| Execution | Median time per step | Sample range | Native overhead |
| --- | ---: | ---: | ---: |
| Pre-feature native | 93.458 µs | 92.809–98.497 µs | Baseline |
| Current native | 99.466 µs | 97.234–102.141 µs | +6.4% |
| Pre-feature native, captured | 124.269 µs | 122.302–126.730 µs | Baseline |
| Current native, captured | 135.344 µs | 134.135–139.587 µs | +8.9% |
| Custom resident + native | 129.603 µs | 129.223–130.583 µs | Different work |
| Custom resident + native, captured | 162.355 µs | 154.531–166.474 µs | Different work |
| Explicit CPU transfer + native | 1,898.879 µs | 1,821.333–2,045.339 µs | Different work |

Both extension-free comparisons pass the predeclared 20% bound. Capture is
slower for this small workload: these are complete host-enqueued step costs,
including lifecycle/capture validation, rather than isolated device-kernel
times. The result does not imply that capture slows larger physics workloads.
The explicit transfer measurement includes the costs of ordinary Python and
transactional staging; it supports the documented convenience/performance
trade-off without claiming universal speedups.
