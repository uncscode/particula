# GPU Condensation Support Contract, Examples, and Troubleshooting

## Problem Statement

Issue #1314 completed the first publication phase: the canonical foundations page now states the shipped bounded low-level GPU direct-condensation contract, and migration guidance no longer shows the obsolete return signature. The published contract prevents users from inferring unsupported high-level integration, hidden host transfers, or behavior beyond the fixed four-substep direct-kernel path.

## Value Proposition

The completed phase preserves explicit CPU-to-Warp transfer ownership, direct imports from `particula.gpu.kernels`, fixed-shape fp64 state, and the distinction between bounded low-level support and future high-level backend integration. Text-only publication regression coverage now protects the canonical contract and the concise migration guidance.

## User Stories

- As a scientific user, I want a precise GPU condensation support matrix so that I can select only verified vapor-pressure, activity, surface-tension, latent-heat, and gas-coupling modes.
- As an adopter, I want a runnable explicit-transfer example so that I can allocate reusable buffers and checkpoint CPU state without accidental per-step host synchronization.
- As a maintainer, I want focused commands and documentation guardrail tests so that published claims remain aligned with E4-F1 through E4-F6 evidence.

Parent epic: [E4](../../epics/E4/vision_problem.md). This is the final publication track after E4-F1 through E4-F6.
