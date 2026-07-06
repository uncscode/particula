# E2-F6 Documentation Updates

## Required Documentation

- P1 shipped the baseline roadmap page at
  `docs/Features/Roadmap/mass-precision-study.md`.
- P1 also added roadmap links in `docs/Features/Roadmap/index.md` and
  `docs/Features/Roadmap/data-oriented-gpu.md`.
- P2 expanded the roadmap page into a P1+P2 study record with executed
  candidate descriptions, unsupported-candidate notes, and focused
  reproduction commands.
- P4 has now published the final recommendation report that interprets the
  shipped P1-P3 evidence and serves as the canonical downstream citation.

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
- The documentation now matches the landed fast benchmark helper coverage in
  `particula/gpu/tests/benchmark_helpers_test.py` so the opt-in benchmark path
  is backed by a default-runnable validation layer.
- The documentation now includes a checklist to verify candidate names,
  thresholds, memory examples, and reproduction commands against the landed
  executable test surface.

## Shipped P4 Publication Outcome

- `docs/Features/Roadmap/mass-precision-study.md` now publishes the final
  recommendation report title and positions the page as the E2-F6 acceptance
  artifact rather than only a study inventory.
- The report explicitly recommends keeping absolute per-species `np.float64`
  particle masses on CPU and `wp.float64` particle masses on GPU/Warp mirrors
  as the current production policy.
- The report now separates shipped evidence from policy boundaries, executed
  but not recommended candidates, unsupported candidates, deferred follow-up
  investigation, and downstream constraints for any future dtype/schema
  proposal.
- Publication-readiness validation guidance is now documented directly in the
  report, including the focused `mass_precision_cases_test.py`,
  `mass_precision_metrics_test.py`, `benchmark_helpers_test.py`, and
  `pytest -Werror ... mass_precision_metrics_test.py` commands plus the opt-in
  benchmark command.
- `docs/Features/Roadmap/data-oriented-gpu.md` now names the report as the
  canonical downstream reference before any production dtype-or-schema
  migration proceeds.
- `docs/Features/Roadmap/index.md` now keeps the roadmap artifact link text in
  sync with the published report title.
- `docs/Features/particle-data-migration.md` now directs contributors to the
  mass-precision report before changing particle mass dtype/schema behavior.

## Notebook/Example Notes

If the evidence is presented in a notebook under `docs/Examples/` or
`docs/Theory/`, follow the repository Jupytext workflow: edit the paired `.py`
file, sync the notebook, execute it, and commit both files.
