# AIOMFAC activity package for Particula

The reference, direct activities on CPU and CUDA, closed CPU and CUDA speciation, and two-vapor condensation on CPU and CUDA are implemented locally. [Closed-box and rising-parcel qualification](aiomfac-condensation-integration.md). [Milestones 4 and 5 qualification](aiomfac-speciation-and-cuda.md). [Milestones 2 and 3 implementation and evidence](aiomfac-cpu-activity.md). Build a standalone activity calculator in the Particula repository and expose it through `pip install "particula[aiomfac]"`. Deliver direct float64 CPU and NVIDIA CUDA calculations, integrate them into kinetic condensation including resident execution and restart, and qualify optional sparse tables. [Milestone 7 table generation, empirical domains and measured break-even](aiomfac-sparse-activity-tables.md). Viscosity is deferred.

## Milestone 1 qualification — 2026-11-22

The [standalone package](../packages/particula-aiomfac/README.md) exposes the pinned model identity and the root extra resolves it from local wheels. Particula's built version is 0.3.4 and the backend is 0.1.0. On Linux, clean Python 3.12 environments passed base-only, backend-only and combined installation, import-location checks and pip dependency checks. Both wheels were built from their respective source archives, and both base artifacts passed the GPL payload isolation check. Public availability of the compatible backend is a prerequisite for releasing the extra.

The [regeneration guide](../packages/particula-aiomfac/reference/README.md) covers the verified checkout, stock and activity-only builds, 14 canonical cases, tolerances, and explicit fixture updates. The initial generation used GNU Fortran 16.2.1. For the first row of upstream example 0008 at 298 K, the activity-only result gave water activity 0.9878567852129456 and NaCl mean molal coefficient 0.7037048661941817. The stock CLI matched all five reported quantities for each of five species within its printed precision. A fresh Linux build and a repeat using the offline cache passed, including the implicit aqueous CO₂, reordered species, and shared-process history comparisons. The largest normalized charge residual in the corpus was 1.45e-15. The [CI workflow](../.github/workflows/aiomfac-reference.yml) repeats package and numerical qualification. Milestone 2 still owns the public provider contract.

## Feasibility and scientific scope

AIOMFAC evaluates the non-ideality of a specified liquid mixture from its temperature and composition. Each particle is an independent mixture, so a population supplies the parallel work for CUDA. The main engineering costs are reproducing the reference states and electrolyte equilibria, removing dependencies on Fortran module state, and refreshing activities throughout the condensation solver. The GPU crossover batch size, memory limit, and speedup for iterative mixtures require measurement.

Support the active activity chemistry of [upstream AIOMFAC v3.15 at revision 24041a1](https://github.com/andizuend/AIOMFAC/tree/24041a1b0db090de1285008814a5285b4b9b746b). Combine short-range modified UNIFAC, middle-range ion–organic and ion–ion interactions, and long-range electrostatic contributions. Include neutral activities, ionic coefficients, mean electrolyte coefficients, ion activity products, and the upstream sulfate–bisulfate and closed carbonate–bicarbonate–aqueous CO₂ equilibria. Preserve upstream applicability warnings and distinguish fitted parameters, analog estimates, and unavailable interactions. See the [original model](https://acp.copernicus.org/articles/8/4559/2008/), [extended parameterization](https://acp.copernicus.org/articles/11/9155/2011/), and [carbonate and iodine extension](https://acp.copernicus.org/articles/22/973/2022/).

Evaluate one specified liquid phase, including metastable compositions where supported. Phase separation, precipitation, inversion from RH to equilibrium water content, general organic-acid dissociation, open-system CO₂ exchange, and reactive gas partitioning are outside this release. Explicit subgroup counts are the organic input. Automated molecular structure parsing and lumping are optional future preprocessing.

Particula can offer ideal, κ, BAT, and AIOMFAC through the same model-selection interface. κ describes water uptake, the qualified BAT workflow handles binary water–organic mixtures, and AIOMFAC needs the full dissolved composition, subgroup assignments, and electrolyte definitions. A missing input or unsupported volatile species must produce an actionable error. The [BAT qualification](../docs/studies/validation/material-properties/binary-bat.md) provides a precedent for comparing direct calculations, table interpolation, and condensation trajectories. The existing [citric-acid comparison](../docs/studies/literature/particle-water-uptake/citric-acid/index.md) provides a scientific cross-check.

## Distribution and license boundary

| Installation | Proposed behavior |
| --- | --- |
| `pip install particula` | Installs the MIT Particula distribution with its usual dependencies. |
| `pip install "particula[aiomfac]"` | Also installs the separately distributed GPL activity backend. |
| `pip install particula-aiomfac` | Installs a standalone calculator usable independently of Particula. |

Create `packages/particula-aiomfac/` with its own project metadata, source layout, license and notices, documentation, source-generation tools, reference fixtures, tests, and release process. Use distribution name `particula-aiomfac`, import name `particula_aiomfac`, and initial package version 0.1.0. Record the upstream scientific version separately. The root extra initially depends on `particula-aiomfac>=0.1,<0.2`. The standalone package declares its numerical dependencies directly. Its adapter checks the Particula provider-interface version when selected.

Keep translated equations, CPU and CUDA implementations, upstream parameter data, and derived fixtures in the GPL package. Preserve copyright and file-specific GPL-3.0-or-later notices, modification records, upstream provenance, and applicable third-party solver notices. Put the generic, independently authored provider interface and condensation integration in Particula. Load the adapter only when selected. Keep the AIOMFAC dependency out of unrelated extras.

The extra controls installation. Distribution of an integrated work can carry GPL obligations even with separate wheels in a common repository. Document the licenses, preserve MIT notices, and ship the source, data, and generation instructions needed for the covered backend. Review the actual release artifacts and their notices, including third-party material. See the [GNU GPL FAQ on plugins](https://www.gnu.org/licenses/gpl-faq.en.html#GPLPlugins) and [Python packaging guidance for extras](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#optional-dependencies).

Explicitly exclude the backend directory from the base source archive. Inspect both wheels and both source archives, including Particula’s bundled documentation. Keep backend-derived fixtures and translated examples in the backend distribution. The base docs can describe installation, selection, assumptions, and how to find backend documentation.

## Public interfaces and physical quantities

| Proposed interface | Responsibility |
| --- | --- |
| `MixtureSpec` | Immutable independent components, ordered material identifiers, subgroup counts, electrolyte stoichiometry, and dissolved-phase membership. |
| `AIOMFACModel` | Parameter selection, preparation for a specified mixture and device, and batched direct evaluation. |
| `ActivityResult` | Named neutral, ionic, and speciation outputs with status, definition masks, reference states, and provenance. |
| `ActivityTable` | Immutable interpolation data tied to a validated mixture domain and model revision. |
| `ActivityAIOMFAC` | Particula adapter supplied by the backend. |

Accept temperature in kelvin and nonnegative component masses along the final array dimension. Each leading index denotes an independent mixture. Normalize standalone mass amounts internally. In Particula, read native masses in kg per particle across the full material axis, then select the explicitly dissolved components. Insoluble material still contributes to the particle properties that Particula owns. Preserve active-slot masks, box membership, native material order, gas order, and the existing mass-transfer accounting.

Prepare the internal electrolyte species layout once, including implicit species needed for dissociation. Keep the conserved analytical input masses in the native material lanes. Check electrolyte electroneutrality independently of any net particle charge. Handle empty/inactive slots before normalizing. Reject invalid active compositions, missing interaction parameters, invalid temperatures, and unsupported mappings explicitly.

Return neutral mole fractions, mole-fraction-based log coefficients and activities, ion molalities and molality-based log coefficients, requested mean electrolyte coefficients and ion activity products, and the speciated composition. Report reference conventions, presence and definition masks, convergence diagnostics, and applicability warnings. Preserve the pinned aqueous infinite-dilution convention and its concentration-scale conversion. Document the molality denominator in mixed solvents, kg of electrolyte-free solvent, independently of the ionic reference state. Characterize absent species explicitly because upstream zero/sentinel output cannot serve as a measured limiting coefficient. See the [upstream output guide and worked electrolyte examples](https://aiomfac.lab.mcgill.ca/help.html).

Add a temperature-aware activity and partial-pressure hook that supplies the actual temperature at evaluation and delegates to existing methods for legacy strategies. Expose a versioned provider contract for preparation, CPU/CUDA evaluation, supported outputs, capture, restart identity, and approximation mode. Resolve `aiomfac` lazily through the activity factory and give the exact extra installation command if the backend is missing. The condensation adapter supplies dimensionless activities for supported neutral volatile species in the mapped order. Only these enter the equilibrium vapor pressure calculation with pure vapor pressure and the Kelvin factor. Ionic coefficients and salt activity products are available through their own result fields.

## Direct model and integration

### 1. Freeze the reference

Build an activity-only reference harness around the pinned Fortran calculation. The upstream input/output wrapper enables viscosity, so call the core with viscosity disabled. Keep the Fortran toolchain in reference generation and qualification. Record the source hash, compiler, flags, float precision, switches, and parameter hashes. Export unrounded intermediate SR, MR, LR, scale conversion, equilibrium residuals, final values, and error/status records. Run cases independently and in permuted order to detect cross-call history.

Trace preparation and speciation through [SubModDefSystem](https://github.com/andizuend/AIOMFAC/blob/24041a1b0db090de1285008814a5285b4b9b746b/FortranCode/SubModDefSystem.f90), [ModCalcActCoeff](https://github.com/andizuend/AIOMFAC/blob/24041a1b0db090de1285008814a5285b4b9b746b/FortranCode/ModCalcActCoeff.f90), and [SubModDissociationEquil](https://github.com/andizuend/AIOMFAC/blob/24041a1b0db090de1285008814a5285b4b9b746b/FortranCode/SubModDissociationEquil.f90). Generate a traceable data manifest with subgroup properties, interaction availability, equations, and source locations. Test the active short-range parameter branch selected by `use_latest_param=.false.`. Verify temperature-equation selection on the normal input path before enabling any additional parameter arrays. Label uncertainty outside supported ranges. See the [temperature dependence study](https://acp.copernicus.org/articles/15/447/2015/).

### 2. Port direct float64 CPU activity

1. Validate mixture definitions, prepare compact tables of active groups and ions, and compute immutable molecular properties.
2. Implement composition conversion and neutral short-range combinatorial and residual contributions. Compare the contributions separately with Fortran.
3. Add middle- and long-range terms, reference-state and scale conversions, ionic coefficients, and mean electrolyte outputs. Validate shared-ion mixtures.
4. Implement sulfate and closed carbonate speciation using the corresponding upstream equations, variable transforms, and solver structure. Preserve the sulfate-specific solution path and use the Powell hybrid method for the coupled nonlinear carbonate equations.
5. Batch independent mixtures with bounded memory. Make workspaces explicit. Reuse composition-independent quantities and cache temperature-dependent terms against actual temperature and model identity.

Use a deterministic cold start with a fixed restart sequence. Upstream has saved initial guesses and seeded random retries. Require independent equilibrium-residual and conservation checks for the chosen root. Return failure when the solve exhausts its budget. A future warm start would need distinct accepted/trial state and checkpoint handling.

### 3. Port direct float64 CUDA activity

Use native NVIDIA Warp kernels. Prepare compact parameter arrays, speciation workspaces, per-row output and status buffers, kernel modules, and stream ownership before capture. Validate memory requirements for the prepared mixture and population capacity. Run independent particle mixtures in parallel, with bounded device iterations and masks for converged rows. Keep all composition conversion, speciation, and error handling on device during stepping. A requested CUDA calculation must surface a device failure rather than silently run on CPU.

First qualify intermediate and final values, then profile memory and kernel costs before specialization or fusion. Verify that independent model instances and particle orderings reproduce the same converged results.

### 4. Integrate kinetic condensation, capture, and restart

Extend CPU condensation, GPU fixed-step condensation, GPU adaptive RK23 and ROS3, and the supported coupled resident workflow. Reuse the activity configuration and material mapping where applicable. Perform thermodynamic evaluation inside the rate calculation so it sees the *current trial composition*. Refresh after every relevant fixed-step species update, each RK stage, every ROS3 base or perturbed rate evaluation, changes of temperature, and composition-changing events. A single evaluation at the start of the outer step would give incorrect fluxes.

Distinguish recoverable invalid adaptive trials from configuration errors and persistent thermodynamic failure. Propagate the status before any rate sanitization can convert a failed evaluation into a zero flux. Preserve the solver’s accepted-step and rollback semantics and keep particle-only dissolved material in every activity evaluation.

For resident CUDA, preallocate and retain resources across graph replay. Keep event handoff and activity state on device. Verify the stepping loop incurs no activity-related host transfers, file access, or allocations. Respect the existing graph/device requirements. See the [Warp graph capture documentation](https://nvidia.github.io/warp/v1.18/user_guide/runtime.html).

At existing checkpoint boundaries include the provider/interface versions, upstream revision, parameter hash, ordered species map and reference states, precision, solver settings, and table identity. Restore with the installed compatible provider and verified local table data. Reject mismatches before modifying simulation state. Qualify both uninterrupted and resumed coupled trajectories. Mid-iteration checkpoints are outside scope.

## Optional sparse activity tables

Keep `evaluation="direct"` as the default. Offer `evaluation="hybrid"` for water and neutral-organic condensation activities. Detailed ionic outputs and speciation diagnostics are always calculated directly for requested states.

Start with water content and temperature at fixed dry composition. Then qualify up to two named dry-mixture coordinates, with any remaining dry components retaining specified ratios. Check membership against the original mixture, including those ratios. Do not project an arbitrary mixture onto a nearby table or assign a constant coefficient per species. Published fast electrolyte approximations offer comparisons for accuracy and cost, but they use their own mixing assumptions. See [MTEM](https://doi.org/10.1029/2004JD004681).

Build tables offline from the direct model using a nested dyadic sparse grid and continuous piecewise-linear hierarchical interpolation. Interpolate log neutral activities only for positive abundances. Refine deterministically against direct-model errors and reserve independent points for qualification. Serialize coefficients, indexing, coordinate definitions, bounds, qualified regions, validation protocol and errors, precision, format version, and model/mixture fingerprints. Evaluate the same numerical representation on CPU and CUDA without a runtime interpolation-framework dependency. Keep tables immutable through simulation and restart.

Fall back to direct calculation on the same device at absent-component boundaries, outside the validated domain, on dry-ratio mismatch, in difficult or unqualified speciation regions, and where the estimated error could reverse the flux direction. Count table hits, direct fallbacks, and failures. An optional approximation must remain useful after fallback overhead is included.

Hybrid fixed-step, RK23 and ROS3 use guarded activities. ROS3 samples direct same-device chemistry for its physical Jacobian, including both sides of a table boundary and time-dependent box forcing. Independent interpolation of activities does not ensure Gibbs–Duhem consistency. Check slopes, thermodynamic consistency where the coordinates permit, fluxes, and trajectories. Restrict these tables to the qualified condensation use.

## Validation and performance

| Evidence | Acceptance or comparison |
| --- | --- |
| Translation | Compare each contribution and final result to pinned Fortran. Initial gates for ordinary converged cases are absolute log-coefficient error ≤ 1e-8 and activity agreement at absolute tolerance 1e-10 and relative tolerance 1e-8. |
| Equilibrium | Require equilibrium log-residual ≤ 1e-8 and normalized elemental and charge-balance residuals ≤ 1e-10. Characterize difficult cases before porting. Record upstream nonconvergence explicitly. |
| Device parity | Compare CPU and real CUDA, including failure statuses, reordering, resident graph replay, and restart. |
| Scientific applicability | Compare with published measurements and physical limits, using observable mean electrolyte coefficients where relevant. Identify reference data used in parameter fitting. Model-to-model agreement alone cannot establish physical accuracy. |
| Hybrid activity | At independent points require absolute water-activity error ≤ 1e-4 and absolute effective neutral log-coefficient error ≤ 1e-3 against direct speciated composition; report maximum errors and the qualified domain. These are empirical criteria. |
| Hybrid trajectories | Require relative particle-radius disagreement ≤ 1e-3 and per-phase gas and particle inventory differences ≤ 1e-3 of each species’ conserved initial inventory. Keep existing conservation tolerances. Check slopes, near-equilibrium flux direction, and timestep sensitivity. |

Build the reference suite from pure water, pure limits, binary and multicomponent organics, NaCl, mixed and shared-ion electrolytes, sulfate/bisulfate/sulfuric acid, closed carbonate/bicarbonate, and mixed organic–electrolyte systems. Cover dilute to concentrated and metastable states. Test missing parameters, zeros, absent species, invalid temperature, failed solves, repeated calls, and permutation invariance. Include tests isolating temperature equation switches and ionic reference-state conversions.

Exercise multi-box populations, inactive slots and capacity changes, inert lanes, gas/material order permutations, condensation, evaporation, depletion, near-equilibrium flux, and temperature changes. Compare accepted/rejected adaptive trials, coupled events, captured versus uncaptured results, and uninterrupted versus resumed trajectories. For tables, include interior held-out points, boundary crossings, dry-ratio mismatch, and direct fallback after composition-changing events.

Benchmark direct and hybrid CPU and CUDA across batch size, material count, active groups and ions, and neutral, simple electrolyte, and iterative speciation cases. Measure standalone activity and complete condensation trajectories. Report setup and JIT separately from warmed timing, device and software versions, peak memory, transfers, iteration counts, table hit rates, and end-to-end speed. Promote a table as an acceleration feature only when it achieves ≥ 2× activity-evaluation speedup against direct evaluation on the same device and a measurable improvement for its declared end-to-end workload while meeting the accuracy criteria.

## Delivery and release

| Milestone | Observable completion criterion | Engineer-weeks |
| --- | --- | ---: |
| 1. Reference and provenance | Reproducible activity-only harness, case corpus, parameter manifest, and license inventory. | 2–3 |
| 2. Package and provider | Isolated archives, standalone import, lazy extra, and temperature-aware compatibility tests. | 2–3 |
| 3. CPU activity terms | Neutral and electrolyte contributions and final outputs pass pinned comparisons. | 3–5 |
| [4. CPU speciation](aiomfac-speciation-and-cuda.md#milestone-4--cpu-equilibrium) | Closed-system cases converge reproducibly with checked residuals and explicit failures. | 3–5 |
| [5. Direct CUDA](aiomfac-speciation-and-cuda.md#milestone-5--direct-cuda-activities) | Real-device parity across supported chemistry, including bounded device speciation. | 3–5 |
| [6. Condensation integration](aiomfac-condensation-integration.md) | Two neutral vapors exchange in a finite box and a thermodynamically coupled rising parcel. Resident RK23 and ROS3 pass replay and restart on NVIDIA. | 8–12 |
| [7. Sparse tables and scaling](aiomfac-sparse-activity-tables.md) | [CPU/CUDA build, lookup and trajectory study](../docs/studies/validation/condensation/aiomfac-sparse/index.md). Fixed-dry and bounded variable-composition domains pass fixed-step, RK23 and ROS3 trajectory and fallback tests. | 4–6 |
| [8. Release qualification and maintainer runbook](aiomfac-pypi-release.md) | PyPI-only distribution of the AIOMFAC backend, verified installation matrix, performance report, user guide, and reproducible examples. | 2–3 |
| **Total** | **Complete activity scope, including optional table qualification** | **27–42** |

These are planning estimates for an experienced scientific-software developer with GPU access and scientific review. Allow 20–30% contingency, especially for difficult speciation and resident integration. The direct CPU calculator is independently usable after milestone 4. Label supported chemistry, solver modes, and devices at each stage.

Add explicit tests, lint, and type checks for the nested project. Test clean installations of base Particula alone, the backend alone, and both together via the extra. Check source archives and wheels, license files, dependency metadata, base documentation bundles, and independent versioning. Route backend releases separately from the base package and the Particula Helper plugin. Account for the existing upstream-main version-change release trigger as well as release-tag events. Stage both distributions together. Publish the compatible `particula-aiomfac` wheel and source archive to PyPI first, then publish the `particula` version that advertises the extra. Verify that `pip install "particula[aiomfac]"` from PyPI in a fresh environment reproduces the mixed reference case’s water activity.

Document subgroup inputs, units, standard states, parameter gaps, approximation domains, numerical failures, and restart compatibility. Publish reproducible qualification evidence and benchmark conditions alongside the owning studies. Track the [Stage 3A thermodynamics roadmap](stage-03a-thermodynamics.md) separately from evidence that each milestone has actually passed.
