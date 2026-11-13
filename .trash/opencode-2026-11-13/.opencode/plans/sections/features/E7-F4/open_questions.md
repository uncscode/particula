# Open Questions

- [x] Is the initial checkpoint contract an on-disk serialization format?
  - Resolved 2026-07-26: No. E7-F4 defines a versioned typed in-memory
    checkpoint/restore boundary. Durable file formats and remote storage are out
    of issue #1451 Track T4 scope.
  - Rationale: Existing conversion helpers establish CPU object restoration but
    no stable storage codec; inventing one would expand compatibility and
    security scope.
  - Evidence: `particula/gpu/conversion.py:422-626`.

- [x] Does a normal checkpoint terminate or replace the live GPU session?
  - Resolved 2026-07-26: No. `checkpoint()` synchronizes and snapshots while the
    resident session remains active; `finalize()` is the explicit terminal
    operation.
  - Rationale: Issue #1451 requires transfers only at explicit checkpoints and
    supports multi-timestep residency; checkpoint inspection must not force a
    hidden backend transition.
  - Evidence: `docs/Examples/gpu_complete_process_sequence.py:485-538` shows the
    current final-only boundary that E7-F4 generalizes.

- [x] Should sidecar concrete records become top-level public APIs?
  - Resolved 2026-07-26: No. The session exposes deliberate resource views while
    concrete records stay in their owning kernel modules unless E7-F6 approves a
    specific stable export.
  - Rationale: Existing kernel exports intentionally expose entry points but not
    scratch internals.
  - Evidence: `particula/gpu/kernels/__init__.py:15-16,24-42`.

- [x] Does T4 define final per-box RNG stream identity and reseeding policy?
  - Resolved 2026-07-26: No. E7-F4 stores and restores opaque mutable RNG
    resources; E7-F8 owns stream identity, box-order invariance, and exact
    stochastic restart semantics.
  - Rationale: This preserves the issue #1451 child-track boundary while making
    the session extensible.
  - Evidence: Issue #1451 dependency chain and Track T8 acceptance criteria.

- [x] Which checkpoint fields should be guaranteed stable for external durable
  serializers after Epic G?
  - Resolved 2026-07-27: Guarantee no external durable-serializer fields in
    E7-F4; stabilize only the versioned typed in-memory checkpoint API and its
    restart behavior.
  - Rationale: A durable codec, migration policy, and long-term compatibility
    lifetime are explicitly outside this track and cannot be inferred from
    runtime Warp layouts.
  - Evidence:
    - `.opencode/plans/sections/features/E7-F4/scope.md:30` - disk/file formats,
      remote checkpoints, and delta checkpoints are out of scope.
    - `particula/gpu/conversion.py:473` - current gas restoration is explicitly
      lossy and therefore is not a durable serialization schema.
  - Resolved by: plan-question-resolver
