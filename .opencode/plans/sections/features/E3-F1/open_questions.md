# Open Questions

- [ ] What exact API shape should express explicit initialization for
  caller-provided `rng_states`?
  - Options include a keyword-only initialization mode, a boolean flag, or a
    documented separate initializer call. The decision must preserve existing
    positional call compatibility.
- [ ] Should `coagulation_step_gpu` return `rng_states` in addition to existing
  outputs, or should callers continue retaining the buffer they pass?
  - Open: Returning it may improve discoverability but could be a broader API
    change than needed.
- [ ] How should legacy callers that pass zeroed `rng_states` but expect implicit
  seeding be migrated?
  - Open: P1 should decide whether to preserve an explicit opt-in path or update
    documentation with a migration note.
- [ ] Should benchmark code switch immediately to seed-once semantics, or remain
  intentionally varied by `rng_seed` for historical comparability?
  - Open: Decide during P3/P4 after core tests pass.
