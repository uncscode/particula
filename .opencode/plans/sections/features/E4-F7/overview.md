# GPU Condensation Support Contract, Examples, and Troubleshooting

## Problem Statement

Issues #1314 and #1315 completed the first two publication phases. The canonical foundations page states the bounded low-level GPU direct-condensation contract, while the revised quick-start now demonstrates its explicit-transfer workflow. The published material prevents users from inferring unsupported high-level integration, hidden host transfers, or behavior beyond the fixed four-substep direct-kernel path.

## Value Proposition

The completed phases preserve explicit CPU-to-Warp transfer ownership, lazy direct/concrete imports, fixed-shape fp64 caller-owned sidecars, gas coupling, and explicit final checkpoint restoration. The quick-start performs two sequential calls reusing the same scratch, latent-heat, and energy sidecars; regression coverage protects that behavior as well as the no-Warp path and the canonical documentation.

## User Stories

- As a scientific user, I want a precise GPU condensation support matrix so that I can select only verified vapor-pressure, activity, surface-tension, latent-heat, and gas-coupling modes.
- As an adopter, I want a runnable explicit-transfer example so that I can allocate reusable buffers and checkpoint CPU state without accidental per-step host synchronization.
- As a maintainer, I want focused commands and documentation guardrail tests so that published claims remain aligned with E4-F1 through E4-F6 evidence.

Parent epic: [E4](../../epics/E4/vision_problem.md). This is the final publication track after E4-F1 through E4-F6.
