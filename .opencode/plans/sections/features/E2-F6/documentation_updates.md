# E2-F6 Documentation Updates

## Required Documentation

- P1 shipped the baseline roadmap page at
  `docs/Features/Roadmap/mass-precision-study.md`.
- P1 also added roadmap links in `docs/Features/Roadmap/index.md` and
  `docs/Features/Roadmap/data-oriented-gpu.md`.
- P2 expanded the roadmap page into a P1+P2 study record with executed
  candidate descriptions, unsupported-candidate notes, and focused
  reproduction commands.
- Later phases still need to add broader tradeoff evidence and any final
  recommendation.

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

## Shipped P2 Additions

- The roadmap page now documents the executed study-only candidates
  `fp32_absolute_mass`, `mixed_precision_mass_plus_density`, and
  `fp32_total_mass_fp32_mass_fraction`.
- The page records how each candidate reconstructs comparable per-species
  masses and derived radii for tolerance-based comparison against the baseline.
- Unsupported candidates that would require runtime schema expansion remain
  documentation-only in this phase.

## Shipped P3 Additions

- The roadmap page now records executable P3 thresholds for pure
  reconstruction error, mixed-scale smallest-particle mass error, and
  mixed-scale smallest-particle radius error.
- The page now describes CPU-reference mass-transfer comparisons against
  `get_mass_transfer`, including both per-particle deltas and aggregate
  species-total deltas.
- Clamp accounting is documented explicitly with raw updated mass,
  post-clamp mass, clamp delta, and clamp frequency terminology.
- Per-candidate memory-footprint examples now use explicit
  `n_boxes × n_particles × n_species × dtype-size` calculations.
- Optional benchmark reproduction is documented as an explicit opt-in path on
  `particula/gpu/tests/benchmark_test.py`, including the clean-skip behavior on
  unsupported machines.
- The documentation now includes a checklist to verify candidate names,
  thresholds, memory examples, and reproduction commands against the landed
  executable test surface.

## Notebook/Example Notes

If the evidence is presented in a notebook under `docs/Examples/` or
`docs/Theory/`, follow the repository Jupytext workflow: edit the paired `.py`
file, sync the notebook, execute it, and commit both files.
