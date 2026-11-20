# Prepared simulations: RTX 5060 qualification

This record belongs to the prepared GPU simulation feature branch. It separates
API correctness, scientific validation, performance measurements and publication
checks. Implementation is development work until its PR is merged; this record
does not authorize a release.

## Machine and executable provenance

[Machine and executable provenance in the study](campaign.md#machine-and-executable-provenance)

## Reproduce the campaign

Run on the actual CUDA host, with no competing GPU experiments:

```bash
python -u -m scripts.gpu_studies.campaign \
  --output .artifacts/gpu-campaign --hours 8 --legacy-seconds 2400
python -u -m scripts.validate_gpu_campaign_parity \
  .artifacts/gpu-campaign/cpu-parity.json
python -m scripts.publish_gpu_campaign \
  .artifacts/gpu-campaign docs/data/gpu/rtx5060.json
```

The fixed matrix has 104 jobs: 100 case/mode timing runs and four scientific
histories. Each case runs in a separate process. Ordinary cases have a ten-minute
limit; the specifically approved previous Brownian baseline has forty minutes,
within the same eight-hour campaign budget. Resume preserves the original
deadline and rejects stale executable or mismatched case results. Incomplete
required cases remain incomplete instead of being replaced with smaller jobs.

Timing runs use three warmups and seven repetitions restored to identical
scientific and random state. Restoration and analysis are outside resident
timing; work and completion/status synchronization are inside it. GPU modes
also have seven fresh preparation/work/download runs, except the previous
Brownian baseline, which measures resident work only. JIT/context startup is
warmed. Graph recording and its first replay have distinct costs.

"Direct" refers to retained low-level kernel calls without repeated public
wrapper preflight. The old Brownian implementation is a CUDA comparison, not a
CPU reference. Native NumPy comparisons are limited to matching condensation
and dilution semantics. Condensation workloads include exponential dilution and
four equal coupled condensation substeps. Full-array CPU/CUDA parity is checked
separately after timing. Direct/prepared/captured/legacy GPU results must match
exactly within each common case before publication.

Campaign result hashes cover native aerosol arrays, ordered gas names and the
population tag. They do not hash raw RNG words. Random-state equivalence and
canonical checkpoint continuation are checked separately by the Brownian and
prepared restart/capture regression suites.

Memory admission is the smaller of 6 GiB and 75% of available device memory,
with a conservative primary/scratch/restore estimate. Observed memory is the
Warp memory-pool high-water mark and excludes CUDA context and driver storage.
Both admission and observed-allocation gates must pass.

## Measured cost of residency

[Measured cost of residency in the study](campaign.md#measured-cost-of-residency)

## Scientific acceptance

[Scientific acceptance in the study](campaign.md#scientific-acceptance)

## Profiling findings and limits

[Profiling findings and limits in the study](campaign.md#profiling-findings-and-limits)

## Final validation ledger

- Full library suite before the final review corrections:
  `pytest -q -Werror -m "not slow and not performance"
  --cov=particula --cov-report=term-missing --cov-fail-under=80`: 9,936 passed,
  13 skipped, 57 deselected, one expected failure; **91.91% coverage**.
- Examples and infrastructure before the final review corrections:
  `pytest examples_tests scripts/tests -q -Werror
  -m "not slow and not performance"`: **450 passed, one skipped**.
- Strict RTX gate: `python scripts/qualify_gpu.py --output
  .artifacts/qualification/review/gpu`: **455 passed, zero skips**.
- Final surface-shape and prepared-state regressions: **114 passed** with
  warnings treated as errors. Final review preserves row-shaped surface-tension
  broadcasting and serializes status reads with simulation closure. Publication
  validation also rejects missing timing stages before accepting their samples.
- Hosted library/example/infrastructure tests passed on Python 3.12, 3.13 and
  3.14 after those corrections. Python 3.14 compatibility filters cover only the
  known Warp ctypes layout deprecations in their owning module; regression tests
  confirm unrelated deprecations still fail.
- Full-array CPU/CUDA campaign parity: **17 matching cases passed**,
  `rtol=2e-12`, `atol=1e-30`, against executable digest
  `c853675d08921a1f7eb2c860ca5650fce0458ec4a31743bc8e64666c29a86ef2`.
- Final campaign: **104 of 104 jobs passed**, including 100 timing runs and
  four scientific histories, within the original eight-hour window.
- The quickstart refinement changed final particle mass by at most 0.0545%,
  below its 1% acceptance threshold.
- Ruff and formatting passed; mypy passed on 558 source files; documentation
  structure reported zero errors; all 11 Particula Helper packages validated.

- All 54 CPU-policy documentation examples executed successfully; all saved
  examples passed fingerprint and file-integrity checks. Fourteen CUDA tutorials
  were explicitly refreshed on the RTX GPU. Their execution remains manual.
- Website unit checks and build passed: 432 authored/API pages plus 130
  redirects, with zero link/anchor errors. Existing relocated support anchors
  remain valid. Quickstart and report pages were visually reviewed at desktop
  (1440 pixels) and mobile (390 pixels) widths with loaded figures and no
  page-level horizontal overflow. Scientific plots were also inspected directly.
- Wheel/sdist checks passed for version 0.3.0: 194 documentation files, isolated
  installation checks and an offline sdist rebuild. Build artifacts are local
  validation evidence, not a package release.
- The downloaded scaling ZIP ran outside the checkout using its bundled JSON.
  The multibox ZIP also passed its complete 196,608-active-particle coarse/fine
  CUDA study outside the checkout, with imports verified to resolve to the
  extracted helper files. An initial externally terminated run was not accepted
  as evidence; the independently completed retry exited successfully.

Publication follow-up: the data exporter now selects only the 78 plotted timing
series, their displayed cost/memory summaries and the one small distribution
line. All 104 jobs still pass the same pre-publication gates; raw repetitions
and detailed scientific records are validated locally and omitted from downloads.
