# Infrastructure Reuse

- `docs/Examples/gpu_direct_kernels_quick_start.py:172` provides the canonical
  lazy-Warp, explicit-conversion, caller-owned-sidecar, and final-restore
  example structure. Extend the pattern to the complete process sequence rather
  than adding hidden orchestration.
- `particula/gpu/tests/gpu_direct_kernels_example_test.py:22` provides example
  path discovery, forced-no-Warp execution, subprocess validation, and identity
  assertions to reuse for the new example regression.
- `particula/gpu/conversion.py` and the public `particula.gpu`
  `to_warp_*`/`from_warp_*` helpers remain the only CPU/Warp transfer boundary.
- `particula/gpu/kernels/condensation.py` supplies gas-coupled finalization,
  fixed-four-substep execution, and reusable scratch-sidecar validation.
- `particula/gpu/kernels/coagulation.py` supplies persistent caller-owned RNG,
  fixed-slot merge behavior, and collision diagnostics.
- The direct dilution, wall-loss, slot-management, and nucleation entry points
  delivered by E6-F2 through E6-F8 are consumed unchanged; E6-F9 must not wrap
  them in a new scheduler API.
- `particula/gpu/kernels/tests/condensation_test.py` and
  `particula/gpu/kernels/tests/coagulation_test.py` establish Warp CPU/CUDA
  marker, device, parity, conservation, and optional-skip conventions.
- `.opencode/plans/sections/epics/E6/success_metrics.md:3` is the authoritative
  E6 exit checklist; `dependency_map.md:21` requires all E6-F1 through E6-F8
  before E6-F9.
- `docs/Features/Roadmap/data-oriented-gpu.md:1191` defines Epic F's public
  scope and exit bar; lines 1239 onward define the excluded Epic G ownership.
- `docs/Features/Roadmap/index.md:64` provides the established epic inventory
  and shipped/active/on-deck status pattern used for closeout.
