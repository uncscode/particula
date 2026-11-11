# P6 integration and final M3 evidence

M2 baseline: `fd114270a3e847a6b7b51a2c2ef2f7e5c27ecb4b` (merged #1630).
P1–P5 implementation checkpoints end at `7540d991e`. This single PR implements
the scientific migration requested under #1605; no separate phase issues,
merge, release or M4 scientific acceptance is implied.

## Native integration and developer contract

`particula/integration_tests/native_scientific_processes_test.py` adds 33
facade-free aggregate-to-strategy cases using M2 constructors/builders:

- All three condensation strategies, full reordered gas mapping and inert
  storage. Particle uptake and mapped gas depletion are compared separately;
  latent energy follows the actual extensive mass change.
- Resolved volume .25/1/4 and common count .25/1/4; actual Brownian selection
  reduces active slots and conserves species/weighted charge before charged
  wall loss removes inventory. Gas and environment remain protected.
- PMF and explicit-radius PDF sequences use independent summation/trapezoidal
  integration. Real Brownian finite-grid species defects are bounded by 1% of
  independently integrated gross loss on this 512-node fixture. This bounded
  integration check supplements P4's independent collision integrals and
  convergence, rather than claiming exact arbitrary-grid conservation.
- Wall loss is an explicit sink; surviving inventory plus removed inventory
  closes at rtol=2e-15, atol=0. CPU lost-slot charge retention is preserved.

Initial focused result: 33 passed in 2.02 s. Published narrow developer contract:
`docs/Features/native_scientific_processes.md`, linked from the condensation
feature. Broad tutorials/notebooks remain M5-owned.

## Assertion migration and temporary compatibility inventory

No existing CPU condensation or coagulation test files were removed. Their
legacy fixture coverage remains alongside independent native successor suites:

| Existing behavior | Native successor / preserved authority |
|---|---|
| Condensation strategy/builder/factory admission, missing physical strategies, gas order, skips, multi-box rejection | `species_mapping_test.py`; `native_step_admission_test.py`; native configuration through existing builders/factory |
| Mass transfer, depletion, evaporation, gas-disabled reservoir, latent heat | `native_step_test.py`; `native_step_admission_test.py`; independent particle/gas assertions |
| Staggered theta/batch/random controls, conservation, stability/current gas | `native_staggered_test.py` scalar oracle and ownership/RNG/repeated-evaporation cases; existing stability/performance suites retained |
| Seven coagulation strategy modules and ABC kernels/rates | `native_rates_test.py` physical inputs, independent simple laws and independent inputs to existing complex laws, representation compatibility |
| Resolved selection/binning/merge and density steps | `native_coagulation_step_test.py`, `native_coagulation_moments_test.py`; species/charge moments supersede invalid number-only mass assertions |
| Both CPU wall-loss suites | All 15 normal-suite and 38 adjacent-suite test functions retained with native fixtures; parameterized counts differ; independent `native_wall_loss_test.py` supplies sinks, statistics, rejection and current-charge evidence |
| GPU turbulent selector complexity diagnostic | Same `test_turbulent_public_dispatch_keeps_linear_majorant_path` assertion now inspects the shared selector implementation used by native and legacy wrappers |
| Former `test_tagged_transfer_rejects_before_warp_probe` | `test_pdf_transfer_rejects_before_warp_probe` retains unsupported PDF provenance rejection; `population_metadata_test.py` adds raw resolved/PMF and explicit legacy round trips, bad-tag rejection and map binding checks |
| Native aggregate transfer-boundary fixture | Supported resolved data reaches explicit Warp probing; unsupported/bad tags retain before-probe rejection elsewhere; real transfers have CPU/CUDA evidence |

An AST name inventory against M2 found no removed test function in either
wall-loss suite, GPU coagulation, GPU session, or native aggregate integration.
The transfer-test rename above is intentional behavior expansion. Name
retention alone is not semantic evidence; the phase records and literal
scientific assertions establish the contracts.

| Temporary seam | Consumers / next owner | Removal owner |
|---|---|---|
| `_legacy_aerosol` and legacy aggregate builder | `particle_process.py`, dilution/equilibria and execution adapters; M4; legacy examples M5 | M6 after downstream migration |
| `ParticleRepresentation`, `GasSpecies` and legacy strategy branches | Existing runnable/adapters and explicitly legacy scientific regressions; M4/M5 | M6 |
| Untagged `ParticleData` / Warp code 0 | Legacy transfer/prepared/resident callers; M4 adds preserved native provenance and schemas | M6 review; explicit transfer APIs themselves remain |
| Selected-box wall loss, dilution/nucleation/exhaustion native guards | M4 must implement full composition contracts before removing guards | M4 integration; M6 obsolete compatibility cleanup |
| Resident session/checkpoint native-tag rejection | M4 schema/provenance, lifecycle and RNG integration | M4 implementation; no automatic native support in M3 |

## M4 handoff contract

The maintainer must review the scientific evidence before marking M3 completed
and admitting M4. M4 must preserve explicit population basis, PDF grids and
full species mapping, algorithm-specific weight restrictions and current-state
preflight. It must decide how CPU lost-slot charge storage interacts with slot
activation, without silently replacing CPU laws with GPU clearing behavior.
Prepared calls require stable valid payloads and re-preparation on semantic or
layout changes; selected/resident/checkpoint support is not inherited from
direct support. GPU persistence must retain the existing explicit reset and
continuation ownership. No cross-backend seeded equality or performance claim
has been established here.

## Final revision gate

Final commands and outcomes will be recorded after the integration checkpoint
is committed and all required checks complete. Phase-local results above are
not substituted for final-revision validation. M4 acceptance remains pending.
