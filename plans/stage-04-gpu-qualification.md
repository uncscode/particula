# Stage 4: GPU scientific usefulness and reliability

Status: proposed. Entry dependency: a supported forward workflow with relevant
[verification](stage-01-scientific-correctness.md),
[numerical accuracy](stage-02a-numerical-integration.md), and
[recording](stage-02b-simulation-workflow.md) evidence.

## Outcome

Demonstrate where GPU execution makes a useful aerosol study faster or enables
a study that would otherwise be impractical, while retaining its scientific
accuracy, supported continuation behavior, and reproducibility.

GPU correctness testing accompanies implementation in earlier stages. This
stage adds broad scientific workload qualification and performance evidence;
it does not postpone all CUDA testing until the end of development.

## Existing foundation

Prepared simulations, direct kernels, snapshots/checkpoints, and qualified
CUDA examples already exist. Existing benchmark and qualification records are
revision- and workload-specific. Reuse their methodology where appropriate;
neither historical speedups nor CPU/GPU parity establish accuracy for a new
solver, physics combination, or research case.

## Work packages

1. **Select meaningful workloads.** Use a qualified comparison case, a larger
   population, and/or an ensemble addressing parameter sensitivity or sampling
   uncertainty. State the scientific observable and required accuracy first.
2. **Audit supported compositions.** Check species mappings, population weights,
   source/sink interactions, and continuation. For example, equal-weight
   coagulation requirements must remain satisfied after nucleation/dilution.
   Scope missing capabilities in the owning numerical or physics plan rather
   than benchmarking an invalid substitute problem.
3. **Compare at comparable accuracy.** Match physical assumptions and outputs,
   then establish acceptable errors for each backend. Different valid timestep
   choices are allowed if their accuracy and cost are reported transparently.
   Distinguish deterministic parity from stochastic distribution agreement.
4. **Measure complete cost.** Separate compilation/warmup, preparation,
   transfers, steady stepping, diagnostics, saving, and total runtime. Include
   memory, ensemble size, and resolution. Synchronize actual device work and
   isolate timing from concurrent workloads. Report where the GPU is slower.
5. **Strengthen recurring CUDA validation.** Select a practical real-device
   validation route, separating small regression cases from expensive studies.
   Record hardware availability and actual skips; Warp CPU is not CUDA evidence.
6. **Consolidate based on evidence.** Profile maintainability and runtime paths
   before proposing removal of overlapping infrastructure. Static reference
   counts or line totals alone do not demonstrate that code is dispensable.

## Owning code and evidence

- [Prepared execution](../particula/execution/prepared.py),
  [capture](../particula/execution/prepared_capture.py),
  [checkpoints](../particula/execution/prepared_checkpoint.py), and
  [GPU kernels](../particula/gpu/kernels/).
- [GPU study helpers](../scripts/gpu_studies/) and
  [prepared examples](../docs/examples/gpu/prepared/).
- [Prepared RTX evidence](../docs/contribute/validation/gpu/prepared-rtx5060.md),
  [capture qualification](../docs/contribute/validation/gpu/graph-capture-qualification.md),
  and [GPU support](../docs/guide/gpu/support.md).

## Acceptance and validation

- A complete scientific workload has current actual-CUDA evidence, documented
  accuracy, uncertainty where needed, end-to-end timing, and memory usage.
- The conclusion identifies useful workload sizes and limitations; no speedup
  is generalized beyond the measured cases or obtained by weakening accuracy.
- Supported restart/capture behavior retains its documented guarantees. Exact
  continuation is tested separately from CPU/GPU scientific equivalence.
- Repeated validation can detect numerical and lifecycle regressions on real
  CUDA hardware; unavailable hardware is reported as a qualification gap.
- Claims retain source identity, configuration, device/software information,
  measured timing stages, and reproducible commands according to the
  [validation reference](../plugins/particula-helper/references/repository-validation.md).

The first campaign need not implement every unsupported GPU process. Extension
work is selected by scientific value and tracked in its owning plan.
