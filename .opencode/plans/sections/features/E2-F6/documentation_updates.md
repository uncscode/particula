# E2-F6 Documentation Updates

## Required Documentation

- P1 shipped the baseline roadmap page at
  `docs/Features/Roadmap/mass-precision-study.md`.
- P1 also added roadmap links in `docs/Features/Roadmap/index.md` and
  `docs/Features/Roadmap/data-oriented-gpu.md`.
- Later phases still need to extend the report with candidate comparisons,
  measured tradeoffs, and the final recommendation.

## Shipped P1 Content

- Current baseline: absolute per-species `np.float64` mass storage on CPU and
  `wp.float64` on GPU.
- Deterministic study cases: `npf_cluster`, `five_to_ten_nm`,
  `accumulation_mode`, and `cloud_droplet`.
- Assumptions recorded: radius bands, density choices, species volume
  fractions, spherical mass formula, canonical shapes, and focused
  reproduction command.
- Explicit deferral: no precision recommendation yet; that remains future work
  for later E2-F6 phases.

## Notebook/Example Notes

If the evidence is presented in a notebook under `docs/Examples/` or
`docs/Theory/`, follow the repository Jupytext workflow: edit the paired `.py`
file, sync the notebook, execute it, and commit both files.
