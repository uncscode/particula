# Success Criteria

## Functional Criteria

- Existing scalar GPU API behavior still passes for condensation and coagulation.
- A keyword-only explicit `environment` path is published on both GPU entry
  points without breaking positional scalar callers.
- Valid explicit `environment` inputs, direct `(n_boxes,)` Warp-array inputs,
  and hybrid scalar-plus-array direct inputs execute on both GPU entry points.
- Mixed scalar-plus-environment calls fail with clear pre-launch `ValueError`
  messages.
- The migration path does not move temperature or pressure ownership into
  `GasData`.

## Verification Criteria

- Helper tests cover scalar broadcast, valid direct arrays, valid explicit
  environment arrays, hybrid inputs, wrong shape, wrong `n_boxes`, wrong
  device, mixed-input ambiguity, and missing-input behavior.
- Condensation tests cover keyword-only signature behavior, positional scalar
  compatibility, explicit-environment success, direct-array success,
  hybrid-input success, mismatch failures, helper short-circuiting, scalar
  versus uniform-array equivalence, non-uniform explicit-environment parity,
  rejected-input `mass_transfer` immutability, zero-volume short-circuit
  safety, and the one-time box-property precompute regression.
- Coagulation tests cover keyword-only signature behavior, positional scalar
  compatibility, explicit-environment success, direct-array success,
  hybrid-input success, mismatch failures, launch short-circuiting, and direct
  array reuse into the Brownian kernel launch.
- Relevant GPU kernel tests and lint checks pass.

## Documentation Criteria

- API docs and docstrings make scalar compatibility, explicit environment
  conflict rules, supported direct/environment call forms, and early-failure
  contracts explicit.
- Later physics kernels can identify one canonical environment-state feed point
  rather than introducing independent scalar migration logic.

## Done Signal

Issues `#1203`, `#1204`, and `#1205` completed the compatibility-contract,
shared normalization, and condensation regression-hardening milestones when
scalar callers remained supported, valid explicit environment/direct-array
inputs executed successfully, uniform and non-uniform condensation parity was
covered, invalid domains failed before launch, rejected inputs left caller
buffers unchanged, and downstream GPU work could reuse one shared environment
feed path.
