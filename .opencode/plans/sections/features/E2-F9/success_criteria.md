# E2-F9 Success Criteria

## Documentation Criteria

- A foundation guide is discoverable from `docs/Features/index.md`.
- The guide documents `ParticleData`, `GasData`, Warp data schemas, transfer
  helpers, shape conventions, current limitations, and downstream roadmap
  handoff.
- Current limitations are explicit, including single-box CPU condensation,
  scalar temperature/pressure GPU kernels, absent/planned environment state,
  `WarpGasData` schema drift, and fixed-shape graph-capture constraints.
- Roadmap docs link to the new foundation material rather than duplicating the
  same support-boundary content in multiple places.

## Example Criteria

- Minimal examples are discoverable from `docs/Examples/index.md`.
- Default-path examples run successfully on CPU-only environments or skip
  cleanly when optional Warp support is absent.
- If paired notebooks are added, the `.py` and `.ipynb` artifacts stay aligned
  and validation steps are documented.

## Verification Criteria

- Documentation validation passes, or the authoritative unavailable validation
  command is clearly reported in implementation notes.
- Example smoke validation proves imports, transfer-helper names, and limitation
  notes match the shipped API.
- No runtime API behavior changes are introduced by this docs/examples track.

## Done Signal

Issue `#1172` feature `E2-F9` is complete when users can find one accurate foundation
guide, one validated example path, and one roadmap handoff trail without having
to infer current support boundaries from source code alone.
