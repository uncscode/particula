# GPU Condensation Support Contract, Examples, and Troubleshooting

## Problem Statement

E4-F1 through E4-F6 establish GPU condensation physics, integration, and parity evidence, but users still need one accurate, discoverable contract for the supported low-level path. Existing documentation describes the pre-E4 API and intentionally says that GPU latent-heat feedback is future work. Without a coordinated publication pass, users could infer unsupported high-level integration, perform hidden or repeated host transfers, or miss the fixed-shape and four-substep constraints from issue 1272.

## Value Proposition

E4-F7 turns the verified implementation into an evidence-backed support matrix, runnable Warp CPU example, parity walkthrough, troubleshooting guide, and focused reproduction commands. It preserves explicit CPU-to-Warp transfer ownership, direct imports from `particula.gpu.kernels`, fixed-shape fp64 state, and the distinction between low-level support and future high-level backend integration.

## User Stories

- As a scientific user, I want a precise GPU condensation support matrix so that I can select only verified vapor-pressure, activity, surface-tension, latent-heat, and gas-coupling modes.
- As an adopter, I want a runnable explicit-transfer example so that I can allocate reusable buffers and checkpoint CPU state without accidental per-step host synchronization.
- As a maintainer, I want focused commands and documentation guardrail tests so that published claims remain aligned with E4-F1 through E4-F6 evidence.

Parent epic: [E4](../../epics/E4/vision_problem.md). This is the final publication track after E4-F1 through E4-F6.
