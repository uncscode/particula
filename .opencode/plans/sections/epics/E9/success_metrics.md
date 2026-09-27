# Success Metrics

All items below are pending acceptance criteria, not checked implementation
claims. Record evidence per track and aggregate it at M6.

- [ ] Six maintenance tracks complete in the fixed validated serial order.
- [ ] Flat `Aerosol` holds all three data containers without legacy facades,
  and supported runnable composition preserves process order and identity.
- [ ] Whole-container getters return held objects; validated individual and
  coordinated replacement retain supplied identity; rejected replacement
  changes neither old references nor data.
- [ ] Tests cover units, non-unit volume normalization, ordered species
  alignment, per-species conservation and documented mutation/failure behavior.
- [ ] Single-box CPU limits still reject unsupported multi-box execution.
- [ ] Dilution handles partitioning and nonpartitioning gas; nucleation retains
  current-gas substeps and per-attempted-substep atomicity.
- [ ] Every enduring scientific assertion formerly using a legacy fixture is
  migrated or has a documented equivalent, never discarded as facade-only.
- [ ] Zero supported executable consumers of removed APIs and zero unremoved
  temporary compatibility items remain. Removed public imports are tested.
- [ ] Historical references are explicitly labeled and confined to appropriate
  historical/migration material; supported examples use replacement APIs.
- [ ] Every supported affected example executes; every affected notebook pair
  is synchronized and executed with both files retained; strict docs pass.
- [ ] Focused assertions, both CPU wall-loss suites, untargeted repository
  coverage runner, required linting/type checking and release docs pass.
- [ ] Existing GPU transfers, exports, ownership and prepared/resident
  boundaries remain regression-covered. Warp CPU evidence is obtained when
  installed; optional CUDA passes or cleanly skips, never falls back.
- [ ] v0.3.0 breaking changes, replacement APIs, initialization/presets and
  migration steps are documented and reviewed.

No throughput target, arbitrary test count, new local coverage threshold or
unmeasured GPU performance claim is an acceptance metric for this migration.
