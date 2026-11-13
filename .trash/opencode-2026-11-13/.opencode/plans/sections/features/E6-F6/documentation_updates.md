# Documentation Updates

- [x] Added `docs/Features/slot_exhaustion_policies.md` with the shipped CPU
  planning and direct-Warp primitive contracts, imports, sidecar schemas,
  diagnostics, mutation boundaries, and deferred scope.
- [x] Updated `docs/Features/Roadmap/data-oriented-gpu.md`,
  `docs/Theory/Technical/Dynamics/Nucleation_Equations.md`, and `AGENTS.md` to
  distinguish delivered primitives from deferred discovery, activation,
  source/gas handling, policy composition, and integrated loops.
- [x] Updated `.opencode/guides/architecture/architecture_outline.md` for the
  shipped P4 direct CPU/Warp scaling boundary, caller-owned sidecars,
  concrete-only exports, and its deferred policy/transfer/runnable scope.
- Reconciled E6-F6 records to use the final primitive names and diagnostics:
  P2's tolerance-bounded conservation diagnostic remains distinct from its
  radius-cubed, mean-radius, surface-area, and Riemer-diversity diagnostics;
  Warp P2 records `planning_status`; and P4 records `resolved_scale`.

Documentation states that the CPU resolver defaults to resampling on and volume
scaling off; sufficiently releasable resampling wins before enabled scaling;
no viable policy raises before a plan returns; and no demand is silently
discarded. These are planning-selection rules, not a runtime policy loop.

E6-F5 discovery, free-index classification, and activation, together with
E6-F6-P5 runtime composition, remain deferred. P7 is documentation-only and
does not establish a public policy API or add a pytest module.

Issue #1427 (P6) remains a no-public-contract phase: it adds only conservation
validation in existing CPU and Warp test modules and changes no production,
export, or public API behavior.

Downstream work remains owned by E6-F7 (CPU source and gas inventory), E6-F8
(direct Warp nucleation), and E6-F9 (integrated process sequencing); these
documentation updates do not implement or imply any of those integrations.

P7 validation:

```bash
pytest particula/particles/tests/exhaustion_test.py \
  particula/gpu/kernels/tests/exhaustion_test.py -q -Werror
mkdocs build --strict
```
