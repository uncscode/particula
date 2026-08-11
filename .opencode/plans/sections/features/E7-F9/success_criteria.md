# Success Criteria

- [ ] All E7-F1 through E7-F8 contracts are shipped and represented in the
  closeout matrix without contradiction.
- [x] **P1 / issue #1528:** Concrete-only GPU diagnostics report total species
  mass, particle-number concentration, latent energy, and conservation residual
  against co-located contract/oracle coverage; their units and ledger signs are
  defined at the concrete boundary without public-documentation expansion.
- [x] **P2 / issue #1529:** Schema-v3 requires continuation metadata while
  permitting zero current-word payloads; canonical primary bytes and
  registry-owned continuation fields are authoritative, arbitrary caller outputs
  are excluded, and coagulation/wall-loss resource-continuation asymmetries fail
   before restart setup. Focused tests cover primary immutability and schema-v2
   noncommunication restart.
- [x] **P3 / issue #1530:** Real repeated resident-loop regressions cover closed
  GAS and PARTICLES maps, canonical ordinary-node ordering, current derived
  state, one setup upload per CPU container, stable resident identities/schemas,
  and tight closed GAS inventory conservation. A late writer failure closes the
  guard, faults the session, and rejects later dispatch; rollback is not claimed.
  Private nucleation dispatch now executes its adapter and records completion,
   without changing the public API or canonical order.
- [x] **P5 / issue #1532:** Test-only transport/restart evidence covers independent
  NumPy float64 closed-map amount oracles for expansion, mixing, dilution, sparse
  maps, and empty/disabled barriers; exact-device transport restart with fresh
  resource identities and preserved published stream words; and direct
  open-boundary source-minus-sink ledger reconciliation at `rtol=1e-12`,
  `atol=1e-30`. No production API, checkpoint schema, scheduler ordering,
  exports, or user documentation changed.
- [x] **P6 / issue #1533:** The canonical
  `docs/Examples/gpu_resident_multi_timestep.py` uses availability validation and
  the resident scheduler for a three-box source session, one setup upload, two
  source steps, caller-owned diagnostics, manual exact-device restart, and cached
  source finalization. Its documentation regression covers disabled, import,
  failure, and enabled behavior without claiming CPU fallback or required CUDA.
- [ ] At least one condensation and one Brownian coagulation workflow run through
  backend selection and match CPU references within recorded tolerances.
- [ ] A multi-box GPU-resident loop runs all supported processes between explicit
  checkpoints with per-box RNG and no hidden CPU transfer or synchronization.
- [ ] Repeated steps preserve container, array, sidecar, diagnostic, capacity,
  and stream identities/shapes.
- [ ] Environment and derived gas state are refreshed in canonical order before
  consumers; no stale-state fixture passes accidentally.
- [ ] Independent boxes match one-box references; unrelated box additions,
  disabling, and reordering do not perturb enabled logical streams.
- [ ] Prescribed advection, dilution, mixing, expansion, and volume updates match
  CPU oracles and satisfy explicit conservation/source-sink accounting.
- [ ] Same-backend checkpoint/restart preserves required metadata and continues
  deterministic and stochastic state as documented.
- [ ] Missing devices/unsupported physics fail clearly or cross only an explicit
  fallback boundary; no failure triggers silent movement.
- [ ] Warp CPU matrix passes; optional CUDA rows pass where available or skip
  cleanly; no mandatory CUDA CI is introduced.
- [x] The complete example uses explicit caller diagnostic observation and
  checkpoint/restart boundaries, and is protected by an executable documentation
  regression.
- [x] Changed executable modules retain >=80% coverage, repository thresholds are
  not lowered, export tests pass, and `mkdocs build --strict` passes.
- [ ] Epic H performance/graph capture and Epic I autodiff remain deferred.

## P7 Gate Status (2026-08-11 — shipped)

P1--P6 artifacts are verified. P7 is shipped with recorded validation evidence.
Warp CPU was available and the CUDA-guarded rows passed. The focused assertion,
export boundary, resident-fast-suite, full-package-coverage,
changed-module-coverage, strict-`mkdocs build --strict`, and optional-CUDA
command groups all have results recorded.

The changed executable module target list, derived from P1--P6, is
`particula/execution/diagnostics.py`,
`particula/execution/gpu_resources.py`,
`particula/execution/checkpoint.py`, and
`particula/execution/resident_scheduler.py`. Its aggregate >=80% gate passed;
P7 Markdown-only work is excluded. No remediation remains.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Setup uploads per GPU session | Manual/ad hoc | Exactly one per CPU container | Transfer spies |
| Intermediate bulk restores/syncs | Not system-guaranteed | 0 before explicit boundary | Full-loop regressions |
| Supported process order violations | Not centrally guarded | 0 | Scheduler trace tests |
| Independent-box parity failures | No complete E7 matrix | 0 accepted failures | Multi-box tests |
| Closed-system conservation failures | Incomplete full-loop evidence | 0 outside recorded tolerance | Diagnostic/transport oracles |
| Same-backend restart mismatches | No full-loop closeout | 0 | Restart regressions |
| Required Warp CPU matrix pass rate | No E7 closeout matrix | 100% | Test report |
| Documentation example regressions | Illustrative direct sequence only | 1 canonical passing resident example | Docs test |
| Changed-module coverage | Repository threshold | >=80%, never reduced | pytest-cov |

The definitive roadmap exit bar is preserved verbatim in intent: condensation
and coagulation use the selection API and match CPU within recorded tolerance;
the documented multi-box resident loop runs all supported processes between
checkpoints with per-box RNG and no hidden CPU transfers.
