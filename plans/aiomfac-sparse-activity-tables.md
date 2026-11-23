# Milestone 7 — Sparse neutral activity tables

Backend 0.1.4 builds and evaluates tables with native float64 AIOMFAC. The
builder and the simulation each select a device. A simulation retains direct
AIOMFAC for exact work on its own device. The upstream identity is AIOMFAC-web
3.15 at `24041a1b0db090de1285008814a5285b4b9b746b`. Recorded Fortran
fixtures anchor the direct model. Table construction requires only the installed
backend, bundled parameters, and the dependencies for the selected device.

## First review

The canonical mixed water/organic/NaCl sample gave a tabulated water activity
of 0.98785634 at 298 K against 0.9878567852129456 in the pinned fixture.
The mixed-water build evaluated 2,241 distinct states, including construction
probes. Both build devices produced the same refinement topology for this
recipe. On the independent mixed-water states, each artifact had a maximum
absolute water activity error of 4.11e-6 on either device. Median nominal
construction took 20.6 ms on CPU and 20.8 ms on the RTX 5060 before
serialization. With an empty compiler cache, the first GPU build took 7.40 s.
A one-droplet native water-transfer run recovered the warmed build in about
four repeats. The
[scaling study](../docs/studies/validation/condensation/aiomfac-sparse/index.md)
owns the measured build costs, paired trajectory timings and amortization.

## Implementation

| Boundary | Implementation and review evidence |
| --- | --- |
| Standalone build | `particula-aiomfac-table build RECIPE --device cpu\|cuda:0 --output FILE` and `particula_aiomfac.sparse.build_table` accept a shipped recipe or a JSON definition. Both share deterministic host refinement. CUDA runs the native model's `evaluate_into` and returns selected neutrals and row status. |
| Portable artifact | The NPZ contains numeric arrays and JSON. SHA-256 covers model/parameter identity, recipe, generating device, levels, hierarchical coefficients and per-cell log-error estimates. NPZ loading verifies the digest, array layout, dimensions and ancestor closure before device upload. The ordered species names and sorted subgroup counts form the exact analytical mixture fingerprint. Coordinates follow from dyadic levels and ordered basis indices. |
| Sparse approximation | The tensor increments combine endpoint linear functions and interior midpoint hats. Refinement activates admissible multi-indices by dimension; the controller caches direct answers. Construction probes and failed nodes govern admission in each cell. CPU evaluates an equivalent combination of linear grids. CUDA evaluates the sparse coefficients directly. |
| Chemistry | Mixed-water fixes the three dry shares. The 3D table varies butanediol and fixes vanillic acid to salt at 1:20. The 4D recipe varies acid within the remaining dry mass. Binary neutral, shared-ion, ammonium bisulfate and closed potassium bicarbonate recipes also ship. Each equilibrium node solves its own analytical inventories. |
| Exact evaluation | A vacancy skips chemistry. Absent species, incompatible dry ratios, excursions and unqualified cells use the original analytical masses. CPU gathers those rows. CUDA masks the bounded direct kernel on the caller's stream. Error propagation retains the original row or box, slot and solver-stage identity. |
| Condensation | Native and CUDA box solvers and resident parcel solvers refresh requested neutral activities at each stage. Particula tests box flux direction at the interval endpoints. The parcel bounds the change in summed heating, heat-removal coefficient and each signed flux. An uncertain class recomputes all targets directly. Hybrid ROS3 uses direct same-device chemistry for each physical Jacobian sample and the box forcing derivative. Restart checks the artifact digest and provider identity. |
| Packaging | The root extra requires `particula-aiomfac>=0.1.4,<0.2`. Parameter arrays, table recipes, example tables and CUDA lookup kernels ship in the separate GPL distribution. See the [backend usage and installation instructions](../packages/particula-aiomfac/README.md). |

## Acceptance and interpretation

The independent tests compare absolute water activity and differences in
natural log neutral activity at the same analytical composition. At a fixed
direct speciated mole fraction, the latter equals the error in effective
neutral log coefficient. The gates are `1e-4` and `1e-3`, respectively.
Each recipe uses 4,096 independently seeded interior states plus 64 points
on each near-face and 256 concentrated states. Maximum errors across 28 build/lookup combinations were 6.271e-6 in water activity and 8.919e-5 in log neutral activity. All measured errors were inside the reported cell intervals. At common grid nodes, the largest CPU/CUDA direct log difference was 1.07e-14.
The cell intervals are empirical. Ionic and detailed speciation diagnostics use the direct model.

The trajectory gates compare wet radii at `1e-3` relative, inventories at
`1e-3` of initial species total, per-species conservation at `1e-10`,
temperature at 0.01 K, supersaturation at `1e-4` absolute and first-law
closure at 0.01 K equivalent. The long-ascent gate evaluates the critical
Köhler curves directly on each trajectory and allows at most 1 m discrepancy.
Across all boxes, maximum normalized budget, phase and relative radius errors were 2.92e-13, 1.93e-6 and 1.11e-5. In the 1.5 km parcel the maximum direct-versus-hybrid activation shift was 0.000057 m. The hybrid first-law closure was 6.68e-7 K equivalent. The trace-bound, wrong-ratio, vacancy, restart mismatch and near-equilibrium checks exercise the exact path. A failed builder node records its input masses, temperature, unit coordinates, index and direct status. Every incident cell requires exact evaluation.

A throughput claim requires at least 2× warmed activity speedup and a
measurable whole-workload gain on the same device at a stated size. Check the
[paired measurements](../docs/studies/validation/condensation/aiomfac-sparse/)
before choosing hybrid. The 16,384-slot fixed-step CUDA box ran 1.92× faster end to end, with 2.30× host-facing activity speedup. Either build paid for itself in one run. The 1.5 km haze ascent spends almost its entire time
outside the frozen mixed-two-vapor recipe. Direct RK23 reproduced its cloud
activation more quickly on the RTX 5060. The initial development estimate
was 3–5 engineer-weeks plus one week for scaling. Milestone 8 owns release
qualification and public publication.

## Local verification

The installed matrix exercised base Particula, the standalone CPU backend,
`particula[aiomfac]` and the standalone CUDA extra in isolated Python 3.12
environments. Pip resolved the local wheels offline. Each runtime probe had
only its virtual environment on `PATH`. Both archive pairs passed the
payload audit. The combined install built portable tables, ran the box and
parcel, and resumed hybrid RK23 on an RTX 5060 with Warp 1.19.0. The study
used Warp 1.11.1. Public release is a separate milestone.

The device-backed regression run passed 718 tests and the focused sparse
suite passed all 18 CPU/CUDA tests. Nsight measured two resident graph
launches and 24 lookup kernels across the box and parcel, with zero CUDA
copies and zero allocation calls inside the replay range. The study includes
the profiler command, recorded counts, and the recipe and hardware used for
each performance claim. Targeted Ruff and Particula mypy checks passed. A
whole-backend default Ruff scan also reports seven existing import-order and
variable-name findings in the direct backend and its earlier parcel tests.

### Hybrid ROS3 qualification

The native CPU box, both CUDA box controllers and the resident CUDA parcel
accept the existing hybrid provider. Every stage applies the flux-sign guard.
Every Jacobian difference uses direct physical rates on its execution device,
including cross-material terms and time-dependent forcing. Tests place
perturbations across table faces and compare with the direct derivative.

On the RTX 5060 the 16-slot resident ROS3 box took 49 ms hybrid and 74 ms
direct over the warmed measurement window. The two-class 150 m parcel took
3.69 s hybrid and 2.92 s direct. The 1.5 km parcel reproduced direct ROS3
class activation within 0.00005 m and reproduced its own uninterrupted
continuation exactly after restart. The closed-box and parcel replay trace
contained 27 table lookups, zero CUDA copies and zero allocation calls. The
combined local wheel passed both hybrid ROS3 restart probes with Warp 1.19.0
and Fortran unavailable. All 50 backend tests passed with CUDA enabled.
[Measurements, reference errors and reproduction commands](../docs/studies/validation/condensation/aiomfac-sparse/index.md#ros3-stage-fluxes-and-physical-derivatives).
