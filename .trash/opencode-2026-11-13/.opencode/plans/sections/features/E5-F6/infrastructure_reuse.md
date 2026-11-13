# Infrastructure Reuse

- E5-F1's `CoagulationMechanismConfig`, canonical bit mask, capability
  validator, shared pair-rate dispatcher, and summed-majorant contract in
  `particula/gpu/kernels/coagulation.py` are the extension seam. Do not create a
  second selector or public step.
- `brownian_coagulation_kernel` in
  `particula/gpu/kernels/coagulation.py:991` contains the baseline compact
  active-index preparation, bounded trial scheduling, rank-based pair proposal,
  one-draw acceptance, and swap-pop removal behavior that E5-F1 generalizes.
- `_bound_scheduled_trials()` at
  `particula/gpu/kernels/coagulation.py:335` and mechanism/constants around
  line 103
  preserve overflow and work bounds for a summed majorant.
- `coagulation_step_gpu()` in `particula/gpu/kernels/coagulation.py` is the only
  orchestration and
  launch boundary. Reuse its fail-before-launch validation, optional output
  buffers, persistent RNG state, and single apply launch.
- E5-F3 supplies charged pair-rate/majorant dispatch and charge-conserving merge
  semantics; E5-F4 supplies SP2016 pair rates and an exhaustive safe majorant;
  E5-F5 supplies ST1956 pair rates, majorant, and normalized dissipation/fluid
  density inputs.
- `_ensure_environment_arrays()` and `_ensure_volume_array()` in the existing
  coagulation entry path establish scalar/direct-Warp normalization. Reuse the
  E5-F5 validators for additive calls requiring turbulent inputs.
- `CombineCoagulationStrategy.kernel()` in
  `particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py:118-154`
  establishes CPU additive semantics. Use independently evaluated component
  formulas as test references, not the new Warp dispatcher as its own oracle.
- `particula/gpu/kernels/tests/coagulation_test.py` provides explicit fp64
  fixtures, Warp CPU/CUDA parameterization, inactive-slot cases, fixed-buffer
  identity checks, conservation assertions, and persistent RNG patterns.
- Sibling plans E5-F3, E5-F4, and E5-F5 define the scientific support boundary
  and all-pairs majorant proofs for their terms; E5-F6 composes those proofs
  without weakening them.
