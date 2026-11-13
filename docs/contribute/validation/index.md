# Validation records

These records explain what evidence supports a scientific or release claim and
what remains unqualified. They are not tutorials or implementation roadmaps.
Results are specific to their recorded revision, workload, and device.

| Question | Record |
| --- | --- |
| Is the native migration approved for release? | [v0.3.0 gates](releases/v0.3.0.md): maintainer/release approval remains pending |
| What supports the mass representation? | [Mass precision](numerics/mass-precision.md) |
| What supports the fixed-four-substep condensation method? | [Stiffness study](numerics/condensation-stiffness.md) and [direct parity](gpu/condensation-parity.md) |
| Which direct coagulation mechanisms were validated? | [Coagulation physics](gpu/coagulation.md) |
| What do historical one-thread-per-box timings show? | [Dated coagulation benchmarks](gpu/coagulation-benchmarks.md) and [legacy viewer](gpu/legacy-benchmarks/gpu_benchmarks.ipynb) |
| Is resident capture performance qualified? | [Qualification gates](gpu/graph-capture-qualification.md): UNSHIPPED/BLOCKED |
| How must resident timing and memory be measured? | [Benchmark/memory record](gpu/resident-benchmarks.md) |
| How must capture profiling be collected? | [Profiling record](gpu/graph-capture-profiling.md) |

Missing reviewed artifacts are unavailable and unmeasured. A passing source
test or closed implementation plan cannot populate a missing measurement row.
