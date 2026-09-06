# GPU graph-capture profiling record

## Scope and current evidence status

E8-F7/T7 owns CUDA profiling and machine-bounded optimization
recommendations for resident graph capture. E8-F8 is restricted to the
example, limitations, and epic closeout. P1--P4 profiling support is shipped,
but no reviewed normalized artifact or manifest is checked in. Consequently,
results, rankings, and recommendations are unavailable and unshipped.

Do not invent a machine or software identity, result, speedup, raw filename,
byte size, digest, or recommendation. An absent artifact is not measured and
is not zero.

## Frozen matrix and reproduction commands

The fixed matrix uses small `(1, 16, 2)` and medium `(1000, 16, 2)` workloads
with 100% activity, gas communication, and ordered
communication/environment/gas/condensation/coagulation/dilution/wall-loss/
nucleation/diagnostics processes. It records gas/saturation diagnostics, two
warmups, three samples, seed 1582, a duration of 0.5 s, and replay counts
1/10/100/1000. Kernel profiling is limited to compatible complete
captured-replay small evidence.

The first two commands, the documentation test, the repository runner, and
MkDocs build are hardware-free:

```bash
pytest particula/gpu/tests/profiling_support_test.py -q --no-cov
pytest particula/gpu/tests/benchmark_helpers_test.py -q --no-cov
pytest particula/gpu/tests/benchmark_test.py --benchmark -k resident -v -s --no-cov
pytest particula/gpu/tests/profiling_smoke_test.py --benchmark -q --no-cov
pytest particula/tests/gpu_graph_capture_performance_docs_test.py -q --no-cov
.opencode/tools/run_pytest.py
mkdocs build --strict
```

Native collection and smoke require native-CUDA qualification. They must report
unavailable or skip when qualification fails; CPU and Warp-CPU are not
fallbacks.

## Future artifact provenance

Reviewed normalized artifacts and their manifest pointer belong in
`.artifacts/benchmarks/profiling/`; ignored local raw reports belong under
`.artifacts/benchmarks/profiling/raw/`. Future rows must include a machine
identifier/platform, CUDA device/architecture, driver/runtime, Python, Warp,
source revision, selected `nsys`/`ncu` versions, command, timestamp, workload,
mode, method, and raw artifact reference.

`nsys 2026.1.3.425-1` and `ncu 2026.2.1.5-1` are qualification requirements,
not current evidence. Every reviewed executed row must carry a contained
relative raw filename, byte size, and lowercase SHA-256 reference. Never
publish raw reports, absolute paths, credentials, usernames, or device
pointers.

## Methodology and limitations

Reset, warmup, setup, capture construction, and serialization are outside
intervals. Host launch has no in-interval synchronization; synchronized elapsed
has one post-dispatch completion boundary. Graph launch is not device or kernel
duration. Nsight perturbs timing and is not unprofiled throughput. Only a
complete compatible attributed `nsys` duration supports a bounded kernel
conclusion.

This record permits no CPU or Warp-CPU fallback, portability or cross-machine/
workload inference, speed thresholds, public runtime profiler API, or
auto-tuning. It makes no production, scientific, ownership, or RNG optimization
without a separately referenced correctness plan. Missing prerequisites or
counters are unavailable evidence, never favorable evidence; no fabricated
result, speedup, ranking, or recommendation is current evidence.
