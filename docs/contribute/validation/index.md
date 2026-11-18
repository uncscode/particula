# Validation records

These records explain what evidence supports a scientific or release claim and
what remains unqualified. They are not tutorials or implementation roadmaps.
Results are specific to their recorded revision, workload, and device.

| Question | Record |
| --- | --- |
| How well does direct BAT reproduce the published activity curves? | [Binary BAT reference checks](thermodynamics/bat-activity.md) |
| How well does BAT reproduce measured citric acid water activity? | [Citric acid droplet hygroscopicity](thermodynamics/citric-acid-comparison.md) |
| How well does a CCN-derived κ transfer across sizes and humidities? | [Size-resolved CCN retrieval study](thermodynamics/ccn-retrieval.md) |
| How does CPU/CUDA cost scale for a humidity sensitivity study? | [BAT humidity-pulse ensemble](gpu/bat-humidity-pulse.md) |
| What supports the scientific baseline? | [Scientific correctness](numerics/scientific-correctness.md) |
| Which adaptive method and splitting choices are qualified? | [Adaptive integration assessment](numerics/adaptive-methods.md) |
| What supports recording and exact continuation? | [Simulation workflow](numerics/simulation-workflow.md) |
| What supports independent CPU/CUDA optics? | [Reference corpus](optics/reference-corpus.md) and [CUDA qualification](optics/cuda.md) |
| Is the native migration approved for release? | [v0.3.0 gates](releases/v0.3.0.md): maintainer/release approval remains pending |
| What supports the mass representation? | [Mass precision](numerics/mass-precision.md) |
| What supports the fixed-four-substep condensation method? | [Stiffness study](numerics/condensation-stiffness.md) and [direct parity](gpu/condensation-parity.md) |
| What supports the two-million-particle environmental example? | [Environmental sweep](gpu/environment-sweep.md) |
| Which direct coagulation mechanisms were validated? | [Coagulation physics](gpu/coagulation.md) |
| How closely do published BAT predictions match 30 measured CCN hygroscopicities? | [Cloud activation comparison](thermodynamics/bat-ccn.md) |
| What do historical one-thread-per-box timings show? | [Dated coagulation benchmarks](gpu/coagulation-benchmarks.md) and [legacy viewer](gpu/legacy-benchmarks/gpu_benchmarks.ipynb) |
| Is resident capture performance qualified? | [Qualification gates](gpu/graph-capture-qualification.md): UNSHIPPED/BLOCKED |
| How must resident timing and memory be measured? | [Benchmark/memory record](gpu/resident-benchmarks.md) |
| How must capture profiling be collected? | [Profiling record](gpu/graph-capture-profiling.md) |
| What qualifies custom processes and registered fields? | [Extension qualification](gpu/process-field-extensions.md) |

Missing reviewed artifacts are unavailable and unmeasured. A passing source
test or closed implementation plan cannot populate a missing measurement row.
