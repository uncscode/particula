# Custom processes and fields: qualification

This development record covers the `codex/process-field-extensions` branch.
Qualification is in progress; this record does not imply merge or release.

## Performance protocol

The comparison baseline is pre-feature revision
`a73fa19741885844ffeebbf76723a8b60ab39223`. Extract it with `git archive` and
use the same interpreter, CUDA device and benchmark script for both revisions.
The script is `scripts/gpu_studies/extension_overhead.py`.

Before collecting results, the acceptance bound for the extension-free workload
is set to **20% median overhead** relative to the baseline, separately for
uncaptured and captured execution. This is a bounded regression check for a
small launch-sensitive workload, not a guarantee for all scientific simulations.
If the bound fails, investigate the cause and report it; do not change the bound
to fit the observed measurements.

Use eight boxes with 256 particle slots, native dilution at 0.01/s, a 0.001 s
timestep, 20 warm-up steps and seven samples of 2,000 steps. Synchronize after
warm-up and each timed batch; exclude compilation and preparation. Run baseline
and current revisions serially without competing GPU tests, and repeat in
reverse order if the result is near the bound or has substantial variability.
The report records every sample, source hashes, revision and runtime versions.

Also measure the published prescribed-warming custom process composed with
native dilution in resident, captured resident and explicit CPU-transfer modes.
Use 200 steps per sample for transfers, whose cost includes downloads, staged
Python callbacks, validation and uploads. Report absolute time per step; these
different execution paths do not establish a general GPU speedup.

Every timing run checks completed steps and the analytical native gas decay.
Custom runs additionally check the integrated time-dependent warming field.
Numerical, failure, ownership, capture and restart tests remain separate gates.

## Results

Measurements and final suite results are pending.
