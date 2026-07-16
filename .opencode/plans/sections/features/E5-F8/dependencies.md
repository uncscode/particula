# Dependencies

## Upstream

- **E4 (shipped):** Provides `condensation_step_gpu`, fixed-four P2-finalized
  gas coupling, latent-heat correction, signed `energy_transfer`, caller-owned
  sidecars, and existing independent NumPy-oracle tests.
- **E2/E3 foundations (shipped):** Provide explicit CPU/Warp conversion,
  fp64/device ownership, environment/gas schemas, and Warp CPU test policy.
- **Canonical documentation:**
  `docs/Features/condensation_strategy_system.md`,
  `docs/Features/data-containers-and-gpu-foundations.md`, and
  `docs/Features/Roadmap/condensation-stiffness-study.md` define the current
  bounded contract and must remain consistent.
- **Runtime dependencies:** NumPy and pytest are standard project dependencies;
  Warp is optional at installation but Warp CPU is required evidence whenever
  Warp is installed. CUDA is never a prerequisite.

## Downstream

- **E5-F9:** Uses the walkthrough, separate criteria, and owner table as the T8
  closeout artifact and links them in final E5 support/roadmap documentation.
- **Epic F (GPU Process Completeness):** Owns thermal-state/temperature-feedback
  and adaptive process-completeness work once explicit numerical contracts are
  approved.
- **Epic G (Backend Selection and GPU-Resident Simulation):** Owns high-level
  `Aerosol`/`Runnable` integration, backend selection, resident-loop coupling,
  and broader CPU workflow parity.
- **Epic H (Graph Capture and Performance):** Owns graph capture/replay,
  host-validation separation, performance targets, and precision/memory studies.
- **Epic I (Differentiability and Global Optimization):** Owns broad state
  autodiff beyond the shipped raw-rate interior probe.
- **Approved condensation-physics expansion:** Owns phase-aware surface tension,
  BAT activity, and any other unsupported scientific model; it must receive a
  plan ID and validation contract before implementation.

## Phase Ordering

P1 establishes the independent executable walkthrough. P2 adds the three
separate acceptance categories on that fixture. P3 publishes and validates the
ownership record using the stable evidence boundary. P4 integrates canonical
links and must ship last. E5-F8 can run in parallel with E5-F1 through E5-F7,
but E5-F9 must wait for P4.
