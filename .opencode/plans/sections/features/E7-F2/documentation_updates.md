# Documentation Updates

## User-Facing Contract

- Add backend-selected condensation usage to the E7 execution-context guide,
  showing explicit CPU and Warp requests and typed results.
- Update `docs/Features/condensation_strategy_system.md` with the support matrix:
  isothermal and latent-heat direct paths, representable activity/surface modes,
  CPU-only staggered behavior, and unsupported BAT boundaries.
- Cross-reference `docs/Features/data-containers-and-gpu-foundations.md` for
  device ownership, fixed shapes, explicit transfers, synchronization, and
  concrete-only sidecars.

## Semantics to State Explicitly

- CPU and Warp input/state types are intentionally distinct.
- Warp execution mutates particle mass, gas concentration, vapor pressure, and
  documented caller-owned outputs in place; describe transfer-result identity.
- The direct Warp path always executes four equal substeps and refreshes vapor
  pressure before each proposal.
- Normal selected steps do not upload, restore, synchronize, or silently fall
  back. Checkpoints and resident-session lifecycle belong to E7-F4.
- Pre-launch validation is atomic; post-launch partial failures do not promise
  rollback.
- Publish parity cases, numerical tolerances, conservation checks, Warp CPU
  baseline, optional CUDA status, and excluded claims.

## Examples and Validation

Add or update a focused runnable example only after the E7-F1/E7-F6 public API
is final. Guard imports and output assertions with a documentation regression.
Run the example, focused docs test, and `mkdocs build --strict`. Do not present
the existing direct-kernel complete-process example as a production scheduler.
