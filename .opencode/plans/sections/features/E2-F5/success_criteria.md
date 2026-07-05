# Success Criteria

## Functional Criteria

- Existing scalar GPU API behavior still passes for condensation and coagulation.
- A keyword-only explicit `environment` path is published on both GPU entry
  points without breaking positional scalar callers.
- Mixed scalar-plus-environment calls and pure explicit-environment calls fail
  with clear pre-launch `ValueError` messages in P1.
- The migration path does not move temperature or pressure ownership into
  `GasData`.

## Verification Criteria

- Condensation tests cover keyword-only signature behavior, positional scalar
  compatibility, mixed-input rejection, explicit-environment rejection, and
  helper short-circuiting.
- Coagulation tests cover keyword-only signature behavior, positional scalar
  compatibility, mixed-input rejection, explicit-environment rejection, and
  launch short-circuiting.
- Relevant GPU kernel tests and lint checks pass.

## Documentation Criteria

- API docs and docstrings make scalar compatibility, explicit environment
  conflict rules, temporary P1 rejection behavior, and downstream handoff
  points explicit.
- Later physics kernels can identify one canonical environment-state feed point
  rather than introducing independent scalar migration logic.

## Done Signal

Issue `#1203` completed E2-F5-P1 when scalar callers remained supported,
the reserved explicit-environment API and temporary rejection rules were
documented and tested, and downstream GPU work could reuse one documented
environment feed path in later phases.
