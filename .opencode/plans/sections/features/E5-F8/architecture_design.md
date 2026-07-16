# Architecture Design

## High-Level Design

This is an evidence and ownership feature layered on the shipped direct-kernel
API. The walkthrough owns fixture construction and reporting; production
condensation code remains unchanged.

```text
Immutable physical fixture constants
  +--> independent NumPy fixed-four-substep oracle
  |      +--> expected particle mass / gas concentration / transfer / energy
  |
  +--> separately allocated Warp fp64 state and sidecars
         +--> condensation_step_gpu(device="cpu")
                +--> synchronize and copy observations

expected + observations
  +--> PHYSICS: particle mass and gas concentration parity
  +--> CONSERVATION: initial vs final particle-plus-gas inventory
  +--> ENERGY: signed finalized transfer times latent heat

three independent results + support boundary
  +--> published walkthrough
  +--> deferred capability -> owner -> entry gate -> explicit non-claim table
```

The result model should keep category name, observed metric, tolerance, and
pass/fail state separate. The command must fail overall if any required
category fails, while still printing all category diagnostics.

## Data / API / Workflow Changes

- **Data Model:** No production schema changes. The example may use a small
  local immutable fixture/result structure solely for deterministic reporting.
- **API Surface:** No package export. Add a directly runnable documentation
  example and a Markdown evidence/ownership record.
- **Workflow Hooks:** E5-F8 depends only on shipped E4. E5-F9 consumes stable
  artifact links and results. The normal documentation and pytest workflows
  validate the files; CUDA remains optional.
- **Ownership Boundary:** CPU expected arrays must not alias or be generated
  from Warp output arrays. Caller-owned `energy_transfer` is synchronized and
  observed as diagnostic state, not treated as a third return value.

## Acceptance Categories

| Category | Quantity | Planned criterion |
|----------|----------|-------------------|
| Physics | Final particle masses and gas concentrations vs independent oracle | Explicit per-field `rtol`/`atol` recorded beside each result; use the existing deterministic fp64 parity envelope and never conservation tolerance as a substitute. |
| Conservation | Per-box/per-species concentration-weighted particle-plus-gas inventory | `rtol=1e-12`, `atol=1e-30`, reported independently of oracle agreement. |
| Energy | `Q = sum_particles(Delta m_finalized) * L` by box/species | `rtol=1e-12`, `atol=1e-18`; condensation positive, evaporation negative. |

Final P2 physics tolerances must be copied from the current canonical
`condensation_test.py` fixtures during implementation and printed in the
walkthrough rather than invented or silently relaxed.

## Security & Compliance

No permissions, network access, credentials, or external data are added. The
walkthrough must remain deterministic, avoid unsafe deserialization, validate
all shapes/devices before execution, and make optional dependency/device skips
explicit. Documentation must not represent a skipped CUDA path as evidence.
