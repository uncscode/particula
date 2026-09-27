# Guidelines and Requirements

## Functional Requirements

1. Accept M3's completed-and-validated handoff, including approved M1 mapping,
   normalization and M2 construction/replacement contracts. Unresolved upstream
   decisions block implementation; drafts are not prerequisite completion.
2. Retain the input `Aerosol` and its held data containers by identity for
   supported in-place processes. Respect M1 array mutation/copy rules; do not
   promise more array identity than the approved scientific operation supports.
3. Read current held state at native CPU entry, validate direct mutations, and
   reject unsupported multi-box or incompatible species layouts before writes.
   Never silently take box zero or infer species order solely from equal widths.
4. Retain `|` ordering: each sequence substep executes every process in order
   with duration `time_step / sub_steps` and child `sub_steps=1`. Preserve
   exception propagation; no whole-sequence rollback, retry or reordering.
5. Condensation/coagulation/wall loss delegate to M3 native strategies. Preserve
   rate units, process-owned configuration, environment temperature/pressure and
   wall-loss per-substep nonnegative clipping under M1 normalization conventions.
6. Dilution applies the same exponential factor to particle concentration and
   every gas species, regardless of partitioning. Preserve masses, charge,
   density, volume, species metadata and environment. Concrete preflight checks
   all sources/storage/candidates before mutation, including zero-time or
   zero-coefficient calls. Unexpected commit/setter failure restores written
   concentration state and re-raises the original error, chaining rollback
   failure where applicable. Do not weaken behavior because legacy setters go.
7. `Dilution.execute` preflights concrete strategy state before its first equal
   substep. Compatible custom strategies retain generic equal-substep delegation
   and own validation/atomicity; their return value remains ignored. Do not
   invent whole-call rollback for a later successful-then-failing substep.
8. Nucleation retains supported native topology, precursor eligibility and
   process configuration without a `MassBasedMovingBin` facade dependency.
   Reuse source finalization and commit primitives; resolve distribution admission
   explicitly from the upstream contract, not from a new state-held strategy.
   Each equal substep reads gas left by preceding commits. An attempted commit
   is atomic; earlier successful substeps persist on later failure. Preserve
   no-admission/zero-time behavior and existing substep limits.
9. `EnvironmentData` on the flat aggregate is authoritative. Resolve migration
   of the existing `Nucleation(environment=...)` signature before P3; any
   temporary accepted argument must not override held state or survive M6.
10. Existing CPU adapters retain original request/configuration/runnable/payload
    identities, profile admission and exactly-once calls. Errors propagate with
    no fallback, retry, implicit conversions or added recovery. CPU replacement
    never automatically rebinds prepared GPU/resident objects.

## Quality Bars

- Every function-changing phase ships adjacent `*_test.py` tests in its PR.
  Preserve enduring assertions when replacing facade fixtures.
- Independent expectations use non-unit volumes (0.25 and 4 m^3), unequal
  particle weights and distinguishable mixed-gas lanes. Keep concentration-
  weighted conservation separate from sink accounting and stochastic bounds.
- Keep typed APIs, unit/shape/ownership docstrings, 80-column style and required
  repository lint/type checks. Do not relax tests or coverage to pass migration.

## Constraints

Execute tracks and P1–P5 serially, with completed validation before advancing.
Aim for reviewable S/XS PRs; re-scope oversized discoveries rather than hiding
them in a physics rewrite. Temporary seams require consumer inventory, an owner
and E9-M6 removal gate. No new dependency, export expansion or diagnostic track.
