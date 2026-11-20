# Stage 4: GPU scientific usefulness and reliability

Status: first campaign qualified on CPU and actual CUDA at 1, 8, 32 and 128 boxes.
[Measurements, figures and reproduction](../docs/studies/benchmarks/bat-humidity-pulse/index.md).
Entry dependency: a supported forward workflow with relevant
[verification](stage-01-scientific-correctness.md),
[numerical accuracy](stage-02a-numerical-integration.md), and
[recording](stage-02b-simulation-workflow.md) evidence.

## Execution and delivery

The pilot used `codex/stage-04-gpu-qualification`. The ensemble campaign used
`/tmp/particula-stage-04-campaign` on `codex/stage-04-campaign`, based on
upstream main at `92615153d2`. Follow the shared
[worktree, branch, and PR workflow](README.md#worktree-branch-and-pr-workflow).

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

## First campaign

**Scientific question.** How does aerosol number concentration affect water
uptake during humidification and water loss after a drying pulse when droplets
compete for a finite vapor reservoir? Use the Stage 3A homogeneous BAT case
at 293 K. Hold dry size, composition and box volume fixed. Measure wet radius
in metres, vapor and condensed water in kg/m³, and signed transfer during each
leg. Record both sides of the vapor reset at 80 ms. The one-box pilot found
that the water-transfer history was more sensitive to timestep than wet radius.
[Pulse implementation](../docs/studies/benchmarks/bat-humidity-pulse/bat_humidity_pulse.py)
· [Pilot measurements](../docs/studies/validation/material-properties/binary-bat.md#one-box-humidity-pulse-timing-and-refinement)

| Candidate | Evidence available | Role in this campaign |
| --- | --- | --- |
| Generic, miscible BAT humidity pulse | Matched CPU and captured CUDA trajectories, explicit forcing and within-leg water budgets | Primary transient and batch-scaling case |
| Citric acid hygroscopicity | Twelve experimental water-activity/composition points with uncertainties, plus a native equilibrium endpoint | Independent context for bulk thermodynamics. The measurements do not resolve the pulse timescale. |
| Cyclic chamber with source pulses and coagulation | Qualified CUDA example and same-device continuation | Follow-on stochastic workload. Recheck equal represented counts at every collision boundary and compare distributions over independent seeds. |

The generic organic uses the miscible Figure 9 descriptors of Gorkowski et al.
The citric acid observations are for a different material. Particle-based
cloud model studies give precedent for measuring cost versus computational
population and backend. Their machine timings do not supply a Particula
speedup or accuracy target.
[BAT paper](https://doi.org/10.5194/acp-19-13383-2019) ·
[Marsh et al.](https://doi.org/10.5194/acp-17-5583-2017) ·
[libcloudph++ performance study](https://doi.org/10.5194/gmd-8-1677-2015)

1. **Population and batch size.** Start with independent boxes at 0.5, 1 and 2
   times the baseline number concentration of 10¹¹ m⁻³. The reference box is
   10⁻⁶ m³ with one resolved slot representing 10⁵ identical droplets.
   For the cost curve use the baseline alone, then 8, 32 and 128
   distinct log-spaced concentrations over this range. Report uptake at 80 ms,
   retained water at 160 ms and the full trajectories. Treat concentration
   as a controlled sensitivity variable. Replicating identical boxes would
   measure throughput only. If varying slots within a box, split its represented count among
   slots while preserving the initial number, dry mass and vapor per volume.
2. **Accuracy before timing.** Proposed maximum errors over all 66 observations
   in each box are 0.1% in relative wet radius and 0.5% in each leg's gas and
   particle water transfer, normalized by peak absolute reference transfer on
   that leg. Refine the direct CPU reference until further halving changes each
   metric by less than one tenth of its target. For a vanishing transfer use
   an explicit absolute tolerance in kg/m³ declared before the run. Keep the
   example's independent closed-leg budget check, unchanged dry mass and an
   explicit external vapor-reset ledger. Check direct BAT against its external
   reference and qualify the table between nodes. A shared transport model
   limits claims about physical accuracy.
3. **Cost and decision.** Compare the native CPU workflow, prepared CUDA and
   captured CUDA at the same observation times and saved output. Include a
   matched-step comparison as well as the fastest tested configuration on each
   backend that meets both accuracy targets. Record CPU thread count and
   batching strategy. The primary ratio compares the direct-BAT CPU and
   tabulated-BAT CUDA workflows. Attribute speed to the hardware alone only
   with an additional CPU control using the same interpolation and solver.
   Report first use with empty and populated compiler
   caches, preparation and upload, capture and first replay, synchronized
   stepping, observations/downloads, saving, and wall time to saved history.
   Record peak host RSS and device pool high-water bytes with context/driver
   exclusions. Reuse the historical three warmups and seven repetitions for
   warmed measurements. Retain individual times and report median and spread.
   Identify the first measured batch size with an advantage and report any
   overlap or slower cases. Profile the largest cost before suggesting
   consolidation. The full curve may favor one backend throughout.
   [Historical RTX campaign](../docs/studies/benchmarks/prepared-scaling/qualification.md)
   · [NVIDIA timing guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#timing)
4. **Repeatable CUDA checks.** On a designated CUDA host, use the strict
   [qualification runner](../scripts/qualify_gpu.py) with explicit BAT test
   selections for changes to BAT, capture, or checkpoint handling. Check
   captured versus ordinary fixed steps and exact same-device continuation
   across the scheduled reset. Retain revision, device, command, counts and
   skip reasons. A skipped or empty selection leaves this gate open. Run the
   full performance matrix separately, on an idle device, for claims about
   speed or memory.

The existing binary BAT capture/restart test passed on the RTX 5060 at
`92615153d2` (one pass, zero skips, 9.72 s of pytest time). It changes
temperature and pressure during replay and resumes a saved checkpoint. The
[ensemble restart check](../docs/studies/benchmarks/bat-humidity-pulse/index.md#restart-and-capture)
subsequently verified exact continuation across the vapor replacement at all four sizes.
Reproduce the short regression from the repository root:

```bash
python scripts/qualify_gpu.py \
  'particula/execution/tests/prepared_restart_capture_test.py::test_capture_records_without_advancing_and_replays_exactly[binary_bat]' \
  --output .artifacts/qualification/stage-04/bat-capture.json
```

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
- [Prepared RTX evidence](../docs/studies/benchmarks/prepared-scaling/qualification.md),
  [capture qualification](../docs/studies/validation/workflows-and-releases/graph-capture/index.md),
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
