# Scope

E8-F3 extends the concrete-only resident resource boundary to prepare one
complete, immutable capture resource set for a fixed session and E8-F2 prepared
timestep. It covers process, communication, diagnostic, control, validation,
and RNG sidecars; exact identity reuse; atomic publication; and overflow-safe
logical byte accounting.

P1 is complete: it delivered inventory/reporting for the six existing manifests
and centralized checked arithmetic only. It did not acquire or preallocate any
new resource, alter configuration payload handling, or implement the later
whole-set preparation work.

P2 is complete: it adds only the descriptor-only dilution family and prepared
view validation/identity-retention plumbing. Its two `(B,)` `wp.float64` roles
are normalized coefficient and factors. It does not preallocate, publish,
reacquire, or initialize/reset RNG resources.

P3 is complete: it registers, rather than acquires, one exact absent/closed
communication selection and ordered diagnostic registrations. It retains their
references and resolved metadata/reports after transactional host-only
validation, including O(R log R) overlap detection. It does not change
checkpoint enumeration, scheduler behavior, allocation, or exports.

P4 is complete: it adds concrete-only `CaptureResourceRequirements` and
`CaptureResourceSet`, atomic nonpublishing staging and publication, and a
metadata-only exact-identity retained-set validator. It initializes only newly
created resident RNG stream sidecars before publication, preserves existing
published stream state, and permits clean retry after candidate failures. Its
optional resident-enqueue reference is validation/retention only; it does not
   impose a capture-set prerequisite or change READY, token, or dispatch policy.

P5 is complete: final request construction requires previously published exact
requirements. CAPTURED and READY admission validate the cached set/report and
freeze the requirements/set/report identity triple in `configurations`; real
GAS/PARTICLES loops and the resident example follow that setup order. Validation
is metadata-only and rejects invalid publication before token entry or dispatch.

## In Scope

- Define canonical capture-resource roles, shapes, dtypes, capacities, and
  deterministic manifest order in `particula/execution/gpu_resources.py`.
- Fill remaining gaps in registry-owned preallocation for prepared process controls,
  validation/status work, selected-lane work, diagnostics, communication, and
  persistent RNG state required by the fixed prepared sequence.
- Add one setup-only capture-set acquisition operation that validates all
  supplied storage, stages omitted storage, and publishes only a complete
  nonaliasing set. **Shipped in P4.**
- Pin the exact session, resource views, native records, arrays, optional
  communication map, capacities, and prepared-resource signature.
- Return established views and arrays by identity on compatible reacquisition,
  with no allocation, reseeding, transfer, synchronization, or payload read.
  **Shipped in P4.**
- Compute deterministic logical bytes from checked shape products and manifest
  dtype sizes, with per-role, per-family, and total summaries.
- Add co-located unit and integration tests, plus developer-facing contract
  documentation.

## Out of Scope

- Native graph capture/replay lifecycle itself (E8-F1) or process enqueue
  refactoring (E8-F2), except for their concrete admission hooks required to
  validate and freeze the already-published capture set.
- CPU/uncaptured/captured physics parity (E8-F4), scaling benchmarks (E8-F5),
  the broader state/checkpoint/tape memory model (E8-F6), profiling (E8-F7), or
  examples, runbook, and closeout (E8-F8).
- Runtime resizing, compaction, automatic recapture, fallback, migration,
  allocator-pool tuning, or claims about allocator overhead/reserved bytes.
- New public package exports or changes to direct-kernel ownership contracts.
- Treating P2's supplied `PreparedResourceViews` as an acquisition or
  publication API; it is a concrete-only, read-only validation and identity
  retention seam.
- Treating optional `CommunicationResources.final_volumes` as a manifest role;
  it remains a configuration binding outside P1 inventory reporting.
