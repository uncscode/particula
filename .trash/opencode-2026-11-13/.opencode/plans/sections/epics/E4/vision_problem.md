# Vision and Problem

The direct Warp condensation kernel is fast and fixed-shape, but it currently
implements only a single-step, isothermal, particle-only approximation. This
creates seven program-level gaps:

1. Pure vapor pressure is caller-supplied and can become stale as temperature changes.
2. Particle activity and composition-dependent surface physics are absent.
3. Production uses one explicit step rather than the selected fixed four-substep method.
4. Latent-heat feedback and signed energy diagnostics are absent.
5. Gas inventory is not mutated or conserved with particle mass.
6. Device-aware parity, reusable-buffer, graph, and bounded-autodiff evidence is incomplete.
7. Users lack a precise support matrix and end-to-end reproduction guidance.

## Vision

After E4 ships, `particula.gpu.kernels` provides a documented, fixed-shape GPU
condensation path that refreshes supported thermodynamics on-device, composes
selected activity and surface models, integrates over four deterministic
substeps, applies latent-heat correction, couples gas and particles
conservatively, and is backed by CPU/Warp parity evidence.

## Why Now

E1-E3 established fixed-shape containers, explicit transfer boundaries, and
low-level kernels. Condensation physics parity is the next dependency before
the direct GPU path can be trusted for coupled scientific simulations.
