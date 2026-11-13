# Appendix

## Primary References

- Issue #1377, `[Planner]: GPU Process Completeness`.
- `docs/Features/Roadmap/data-oriented-gpu.md` -- Epic F scope, slot rules,
  exit bar, and Epic G boundary.
- `particula/dynamics/dilution.py` -- existing CPU dilution equations.
- `particula/dynamics/wall_loss/wall_loss_strategies.py` -- neutral and charged
  CPU physics references.
- `particula/dynamics/particle_process.py` -- strategy/runnable pattern.
- `particula/particles/particle_data.py` and `particula/gpu/warp_types.py` --
  fixed-shape data authority.
- `particula/gpu/conversion.py` -- explicit transfer boundary.
- `particula/gpu/kernels/coagulation.py` -- slot clearing and persistent RNG.
- `docs/Theory/Technical/Dynamics/Nucleation_Equations.md` -- initial source,
  gas-depletion, activation, and exhaustion equations.

## Rejected Approaches

- Scaling per-particle mass to represent dilution: concentration changes, not
  particle composition, are authoritative.
- Dynamic append/resize on GPU: violates fixed-shape storage and future
  resident-loop requirements.
- Hidden CPU fallback or process-level transfer: violates ownership and makes
  parity/performance behavior opaque.
- Silent demand truncation on exhausted slots: violates conservation and
  observability.
- Exact stochastic stream matching: overconstrains implementation; use
  deterministic coefficient and statistical outcome validation instead.

## Planning Diagnostics

Classifier diagnostics: `none`.

The codebase-researcher delegation was unavailable because the workflow was at
its subagent-depth limit. Drafting therefore used the detailed issue research,
repository guidance, templates, and cited codebase references.
