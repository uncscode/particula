# Documentation Updates

## Required Updates

- Update the E2 roadmap or GPU feature documentation to describe the scalar
  compatibility bridge and per-box environment path.
- Document accepted call forms for condensation and coagulation GPU APIs:
  scalar temperature/pressure, per-box arrays, and/or explicit environment
  container depending on the final API choice.
- Document `n_boxes` validation requirements and common error messages.
- Note that temperature and pressure remain environment state, not `GasData`
  fields.

## Candidate Files

- `docs/Features/Roadmap/data-oriented-gpu.md`
- Any E2 feature documentation produced by the documentation workflow.
- Docstrings in `particula/gpu/kernels/condensation.py` and
  `particula/gpu/kernels/coagulation.py`.

## Developer Notes

- Explain scalar compatibility as a migration aid for existing callers.
- Explain per-box environment state as the required path for future multi-box
  physics kernels.
- Include a short example showing scalar input and equivalent uniform per-box
  environment input if documentation scope allows.
