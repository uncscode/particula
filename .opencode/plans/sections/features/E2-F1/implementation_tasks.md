# Implementation Tasks

## E2-F1-P1: Inventory Current Schemas

- Read `particula/particles/particle_data.py`, `particula/gas/gas_data.py`,
  `particula/gpu/warp_types.py`, and `particula/gpu/conversion.py`, then record
  one row per public field in
  `docs/Features/Roadmap/data-oriented-gpu.md` or the linked decision record.
- Capture constructor validation, dtype coercion, and leading-dimension rules
  for `ParticleData`, `GasData`, `WarpParticleData`, and `WarpGasData`, naming
  the enforcing method or helper when one exists.
- Record current CPU↔GPU transfer behavior from `to_warp_particle_data()`,
  `from_warp_particle_data()`, `to_warp_gas_data()`, and
  `from_warp_gas_data()`, including explicit notes about placeholder names and
  `vapor_pressure` loss on CPU return.
- Cross-check the inventory against
  `particula/particles/tests/`, `particula/gas/tests/`,
  `particula/gpu/tests/warp_types_test.py`, and
  `particula/gpu/tests/conversion_test.py` so the inventory cites executable
  evidence rather than prose-only assumptions.

## E2-F1-P2: Decide Field Ownership

- Create or update the schema decision record with columns for field owner, CPU
  shape, GPU shape, dtype, mutability, round-trip behavior, and downstream
  consumers.
- Record `ParticleData.density` as shared-across-boxes state with shape
  `(n_species,)`, and cite the source module/tests that already rely on that
  contract.
- Record `ParticleData.volume` as the authoritative per-box simulation-volume
  carrier, explicitly stating that `EnvironmentData` must not own or mutate
  simulation volume.
- Record `vapor_pressure` as explicit process or GPU-helper state rather than
  owned CPU `GasData` or `EnvironmentData`, and note the intentional loss or
  sidecar behavior on CPU restoration.
- Record that `WarpGasData` remains numeric-only and that CPU restoration must
  use caller-supplied names or external index-map metadata instead of treating
  names as preserved GPU state.

## E2-F1-P3: Document Shape Conventions

- Publish canonical shape tables covering `ParticleData`, `GasData`, the new
  `EnvironmentData`, and their Warp counterparts, with one explicit row per
  field.
- State that single-box workflows retain a leading box dimension of size `1`
  for per-box arrays, while shared arrays such as `ParticleData.density` remain
  species-only.
- Document particle-resolved and binned concentration/count conventions with
  example shapes that match existing container tests.
- Document the CPU strategy boundary where containers may store multi-box state
  but current condensation/coagulation execution still requires `n_boxes == 1`.

## E2-F1-P4: Publish Handoff Map

- Add one downstream handoff note per sibling feature `E2-F2` through `E2-F9`,
  naming the exact field, shape, or transfer contract each feature is expected
  to inherit.
- Link the final decision record from
  `docs/Features/Roadmap/data-oriented-gpu.md` and
  `docs/Features/particle-data-migration.md` so implementers do not need to
  search plan prose.
- Run the repository documentation or link-validation command that is available
  for touched docs and record the command beside the update notes.
- Fold in accepted reviewer clarifications from the PR thread before closing the
  phase so sibling features inherit the final contract, not an intermediate one.
