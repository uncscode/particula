# Full example and topic conversion

Requested follow-up to the approved topic explorer. Scope: every authored page
under `docs/examples/`, plus fuller topic descriptions, useful entry points,
prerequisite connections, and routes through the examples. Preserve established
URLs. Review scientific code and saved evidence, not headings alone.

Base before this pass: `277ccafb8`. PR: https://github.com/uncscode/particula/pull/1654.

## Acceptance

- Each worked example connects a question, assumptions and units, meaningful code
  stages, result interpretation, closure, and a concrete research adaptation.
- Index pages help readers choose and sequence examples; no obsolete notebook
  maintenance instructions or misleading duplicate descriptions.
- Topics supply useful scientific context and starting resources, respecting
  the existing graph rules and complete page inventory.
- Scientific corrections use owning implementation and sources; changed code
  receives complete workload refreshes. Prose-only pages retain valid evidence.
- Build/link/structure, relevant tests, saved-result checks, and browser review
  pass before the existing PR is updated. No library physics changes are planned.

## Findings and corrections

- Gas fundamentals used unconverted Antoine coefficients, giving implausibly low
  pressures. Replaced with sourced NIST fits, explicit bar/mmHg and sign conversion,
  a common 304–333 K domain, and an independent equation-convention assertion.
  The strategy comparison now shares a 310 K reference. Full CPU refresh completed; collection test results are recorded below.
- Distribution storage printed a retired-strategy module description. Removed
  that obsolete executable prose; refresh completed to realign saved sections.
- Native aerosol construction uses illustrative glycerol parameters; clearly
  distinguish these from a calibrated material model and separate its later
  two-box demonstration from glycerol growth.

- Coagulation narratives now distinguish bin totals, radius densities, and
  normalized sample histograms. Similar total-number decay is no longer described
  as validation; charged-kernel intermediate warnings and the imposed charge
  relation are explicit. All 11 coagulation scripts retain executable cells.

- Chamber narratives no longer claim isolated-process simulations or experimental
  validation that the code does not perform. Corrected spherical duration to
  100 minutes and described the geometry comparison as nearly equal volume.
  Positivity and repeated-Euler checks are distinguished from convergence.

- Thermodynamics review corrected an ideal-water/nonideal-organic overlay to
  compare organic activity under both models. Corrected activity/coefficient
  plot labels and water-activity/equilibrium wording; phase weights are no longer
  described as water mass fractions.
- Surface tutorial passed molar mass as surface tension and omitted the shared
  phase needed for mixture averaging. Uses explicit SI arguments, assigns both
  components to one illustrative phase, and compares glycerol with glycerol.
  Added independent Kelvin-formula and averaging checks. Full refresh completed.
  All ten thermodynamics scripts and three collection indexes are revised;
  refreshed figures have been inspected; collection test results are recorded below.

- Single-cycle chamber narrative now distinguishes the 2 m³ loss geometry
  from its 0.25 m³ represented sample volume, humidity resets from closed
  condensation steps, and wet/dry particle mass ratio from conservation error.
  Removed unsupported 1% drift and activation-threshold assertions. The retained
  dry-phase oscillations remain explicit; executable cells are unchanged.

- Multi-cycle chamber used duration as a step count while each advance was
  dt=0.1 s. It now executes duration/dt updates and records matching physical
  times (3600 updates, 3601 states per scenario). Original four 30 s/60 s cycles,
  sample sizes, and conservation tolerances are retained. Full CPU refresh completed; 11 simulation tests pass, including both
  one-second and 0.1-second duration checks.
- Respiratory-droplet property lookup resolves lysozyme/IgA and two bicarbonates
  to different substances in the saved output. Reframed this retained example
  as an input-audit failure; removed unsupported realistic-fluid and exposure
  claims. Research use requires verified identities and a new material model.
- Biomass narratives distinguish initial gas saturation from liquid activity,
  internally mixed particles from external populations, and continued
  condensation from the selected collision kernel. Cooling-plume narratives
  distinguish organic uptake onto supplied seeds from soot chemistry/nucleation.
  Organic binned coagulation is presented as a species-transport boundary with
  observed inventory failure, not a reliable composition forecast.

## Topic enrichment

All 33 topics now include scientific context and a concrete investigation prompt.
Their parent/child hierarchy is retained; additional reciprocal connections and
prerequisites connect source, growth, representation, and verification questions.
Curated starting resources now cover previously empty leaf entry points. Starting choices were checked against the completed example-family review. Topic rendering,
graph validation, and the authoring reference enforce the added fields.

## Page review ledger

Status records editorial review. Collection-wide validation is recorded below;
it does not establish physical validation of each demonstration model.

| Page | Review status |
| --- | --- |
| `docs/examples/fundamentals/data_containers.py` | Revised and reviewed |
| `docs/examples/fundamentals/gas/environment_data.py` | Revised and reviewed |
| `docs/examples/fundamentals/gas/gas_data.py` | Revised and reviewed |
| `docs/examples/fundamentals/gas/vapor_pressure.py` | Revised and reviewed |
| `docs/examples/fundamentals/native_aerosol.py` | Revised and reviewed |
| `docs/examples/fundamentals/native_state_access.py` | Revised and reviewed |
| `docs/examples/fundamentals/particles/aerosol_distributions.py` | Pilot rechecked |
| `docs/examples/fundamentals/particles/distribution_tutorial.py` | Revised and reviewed |
| `docs/examples/fundamentals/particles/particle_data.py` | Revised and reviewed |
| `docs/examples/gpu/direct/gpu_coagulation_direct.py` | Revised and reviewed |
| `docs/examples/gpu/direct/gpu_complete_process_sequence.py` | Revised and reviewed |
| `docs/examples/gpu/direct/gpu_condensation_parity_walkthrough.py` | Revised and reviewed |
| `docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py` | Revised and reviewed |
| `docs/examples/gpu/direct/gpu_direct_nucleation.py` | Revised and reviewed |
| `docs/examples/gpu/prepared/custom_process_backends.py` | Revised and reviewed |
| `docs/examples/gpu/prepared/gpu_charged_models.py` | Revised and reviewed |
| `docs/examples/gpu/prepared/gpu_dry_environment_sweep.py` | Revised and reviewed |
| `docs/examples/gpu/prepared/gpu_environment_sweep.py` | Pilot rechecked |
| `docs/examples/gpu/prepared/gpu_ion_balance.py` | Revised and reviewed |
| `docs/examples/gpu/prepared/gpu_lucky_few.py` | Revised and reviewed |
| `docs/examples/gpu/prepared/gpu_prepared_cyclic_chamber.py` | Revised and reviewed |
| `docs/examples/gpu/prepared/gpu_prepared_forcing.py` | Revised and reviewed |
| `docs/examples/gpu/prepared/gpu_prepared_lifecycle.py` | Revised and reviewed |
| `docs/examples/gpu/prepared/gpu_prepared_multibox.py` | Revised and reviewed |
| `docs/examples/gpu/prepared/gpu_prepared_quick_start.py` | Revised and reviewed |
| `docs/examples/gpu/prepared/gpu_prepared_stochastic.py` | Revised and reviewed |
| `docs/examples/gpu/prepared/gpu_scaling_results.py` | Revised and reviewed |
| `docs/examples/gpu/resident/gpu_resident_graph_capture.py` | Revised and reviewed |
| `docs/examples/gpu/resident/gpu_resident_multi_timestep.py` | Revised and reviewed |
| `docs/examples/gpu/resident/gpu_resident_session.py` | Revised and reviewed |
| `docs/examples/optics/cuda_optics.py` | Revised and reviewed |
| `docs/examples/optics/sphere_optics.py` | Revised and reviewed |
| `docs/examples/processes/coagulation/charge/coagulation_with_charge_functional.py` | Revised and reviewed |
| `docs/examples/processes/coagulation/charge/coagulation_with_charge_objects.py` | Revised and reviewed |
| `docs/examples/processes/coagulation/coagulation_1_pmf_pattern.py` | Revised and reviewed |
| `docs/examples/processes/coagulation/coagulation_2_pdf_pattern.py` | Revised and reviewed |
| `docs/examples/processes/coagulation/coagulation_3_particle_resolved_pattern.py` | Revised and reviewed |
| `docs/examples/processes/coagulation/coagulation_4_compared.py` | Revised and reviewed |
| `docs/examples/processes/coagulation/coagulation_5_time_evolution.py` | Revised and reviewed |
| `docs/examples/processes/coagulation/functional/coagulation_basic_1_pmf.py` | Revised and reviewed |
| `docs/examples/processes/coagulation/functional/coagulation_basic_2_pdf.py` | Revised and reviewed |
| `docs/examples/processes/coagulation/functional/coagulation_basic_3_compared.py` | Revised and reviewed |
| `docs/examples/processes/coagulation/functional/coagulation_basic_4_particleresolved.py` | Revised and reviewed |
| `docs/examples/processes/condensation/condensation_1_bin.py` | Pilot rechecked |
| `docs/examples/processes/condensation/condensation_2_massbin.py` | Revised and reviewed |
| `docs/examples/processes/condensation/condensation_3_massresolved.py` | Revised and reviewed |
| `docs/examples/processes/condensation/condensation_latent_heat.py` | Revised and reviewed |
| `docs/examples/processes/condensation/staggered_condensation_example.py` | Revised and reviewed |
| `docs/examples/processes/customization/adding_particles_during_simulation.py` | Revised and reviewed |
| `docs/examples/processes/customization/particle_tracers.py` | Revised and reviewed |
| `docs/examples/processes/customization/timed_box_chemistry.py` | Revised and reviewed |
| `docs/examples/processes/dilution/cpu_dilution.py` | Revised and reviewed |
| `docs/examples/processes/nucleation/cpu_nucleation.py` | Revised and reviewed |
| `docs/examples/processes/nucleation/custom_nucleation_single_species.py` | Revised and reviewed |
| `docs/examples/processes/wall-loss/chamber_forward_simulation.py` | Revised and reviewed |
| `docs/examples/processes/wall-loss/rectangular_wall_loss_strategy.py` | Revised and reviewed |
| `docs/examples/processes/wall-loss/spherical_wall_loss_strategy.py` | Revised and reviewed |
| `docs/examples/processes/wall-loss/wall_loss_builders_factory.py` | Revised and reviewed |
| `docs/examples/processes/wall-loss/wall_loss_tutorial.py` | Revised and reviewed |
| `docs/examples/simulations/biomass_burning_cloud_interactions.py` | Revised and reviewed |
| `docs/examples/simulations/cloud_chamber_multi_cycle.py` | Revised and reviewed |
| `docs/examples/simulations/cloud_chamber_single_cycle.py` | Revised and reviewed |
| `docs/examples/simulations/cough_droplets_partitioning.py` | Revised and reviewed |
| `docs/examples/simulations/organic_partitioning_and_coagulation.py` | Revised and reviewed |
| `docs/examples/simulations/soot_formation_in_flames.py` | Revised and reviewed |
| `docs/examples/thermodynamics/activity/activity_tutorial.py` | Revised and reviewed |
| `docs/examples/thermodynamics/activity/bat_activity_example.py` | Revised and reviewed |
| `docs/examples/thermodynamics/activity/equilibria_example.py` | Revised and reviewed |
| `docs/examples/thermodynamics/activity/ideal_activity_example.py` | Revised and reviewed |
| `docs/examples/thermodynamics/activity/kappa_activity_example.py` | Revised and reviewed |
| `docs/examples/thermodynamics/equilibria/activity_part1.py` | Revised and reviewed |
| `docs/examples/thermodynamics/equilibria/equilibria_part1.py` | Revised and reviewed |
| `docs/examples/thermodynamics/particle-properties/activity_functions.py` | Revised and reviewed |
| `docs/examples/thermodynamics/particle-properties/activity_tutorial.py` | Revised and reviewed |
| `docs/examples/thermodynamics/particle-properties/particle_surface_tutorial.py` | Revised and reviewed |
| `docs/examples/fundamentals/data-containers.md` | Revised and reviewed |
| `docs/examples/fundamentals/gas/index.md` | Revised and reviewed |
| `docs/examples/fundamentals/index.md` | Revised and reviewed |
| `docs/examples/fundamentals/particles/index.md` | Revised and reviewed |
| `docs/examples/gpu/index.md` | Revised and reviewed |
| `docs/examples/index.md` | Revised and reviewed |
| `docs/examples/processes/index.md` | Revised and reviewed |
| `docs/examples/processes/nucleation/index.md` | Revised and reviewed |
| `docs/examples/processes/wall-loss/index.md` | Revised and reviewed |
| `docs/examples/simulations/index.md` | Revised and reviewed |
| `docs/examples/thermodynamics/activity/index.md` | Revised and reviewed |
| `docs/examples/thermodynamics/equilibria/index.md` | Revised and reviewed |
| `docs/examples/thermodynamics/index.md` | Revised and reviewed |

## Interim verification (historical)

- Documentation tests: 55 Python tests and 14 browser-model/unit tests passed.
- All 84 registered result bundles matched executable fingerprints after the
  fundamentals, condensation, coagulation, source, and customization edits.
- Coagulation, source, customization, and wall-loss editorial changes preserve
  executable section lists. Only the four documented fundamentals corrections
  differ from the base, with complete CPU refreshes saved.
- Structure validation and Ruff passed. Site verification caught three
  relocated section anchors (staggered condensation and rectangular wall loss);
  explicit compatibility anchors were restored. The rebuilt site and rendered
  link/structure verification pass.
- Remaining families and final whole-collection review are still required;
  no final PR update or goal-completion claim has been made.
- Browser review confirmed the added scientific context, investigation prompt,
  starting link, and parent/child navigation on Brownian and collision topics.
  Desktop and narrow layouts were inspected; temporary viewport overrides reset.
  Screenshot: `/tmp/particula-topic-context-desktop.jpg`. Corrected gas saturation
  and water vapor-pressure comparison plots were also inspected.

- Thermodynamics pass: all five examples with executable corrections completed
  full CPU refreshes; seven functional activity/equilibrium tests passed with
  warnings treated as errors. Surface assertions independently check Kelvin
  radii, SI conversion, phase averaging, and the selected component's curve.
- Inspected the corrected organic-activity comparison, mixture Kelvin curves,
  activity-rule panels, phase-property panels, and equilibrium concentration
  plot. Labels distinguish phase weight from water fraction and describe
  interpolation endpoints and the Kelvin exponent cap.
- Optics narrative revisions preserve executable section lists and saved data.
  All 84 result bundles have valid file hashes and current fingerprints.
- Following thermodynamics, optics, and single-cycle chamber edits, the website
  build passed: 662 checked HTML pages, 303 API modules, 130 redirects, no
  verification errors. Source and rendered structure checks passed.
- Editorial ledger now records 57 revised pages, three pilot pages awaiting
  final audit, and 28 pending pages. Remaining work includes five simulation
  scripts, twenty GPU scripts, and three collection indexes, then final
  collection review, validation, and the PR update.

## Completed editorial pass

All 88 authored example pages (75 Python, 13 Markdown) are reviewed and revised
or rechecked as pilots. GPU changes preserve executable cells, workload sizes,
assertions, and saved CUDA provenance. Added explicit questions and research
extensions, and clarified artificial inputs and the differences between
state preservation, implementation agreement, inventories, and physical evidence.
The main example index now routes by scientific question and the GPU index
includes custom-process backends and the prepared scientific workflow.

Earlier counts above describe interim progress only.

## Final validation

- Full example suite: `pytest examples_tests -q -Werror -m "not slow and not performance" --no-cov`: 239 passed, one declared skip.
- Website suite: `npm --prefix website test`: 58 Python tests and 14 JavaScript
  tests passed. The renderer regression covers current, stale, and missing
  outputs; matching invocation results now precede trailing interpretation.
- Plugin distribution: 19 tests passed; package/payload check validates all
  11 skills. The installed cached plugin is unchanged.
- All 84 saved bundles pass file-hash and executable-fingerprint checks. Ten
  changed CPU examples received full refreshes. All 21 GPU example executable
  section lists are unchanged; the small resident graph-capture example was
  refreshed on the RTX 5060 because its declared helper input changed.
- Corrected chamber, vapor-pressure, activity, and Kelvin plots were inspected.
  Browser review confirmed topic context and investigation prompts, parent/child
  navigation, and the forcing example’s figure before its closing interpretation.
- Restored legacy respiratory-droplet section anchors after final link checking
  identified renamed headings. This prose-only fix preserves saved results.

- Production build passed: 532 content pages; verification checked 662 HTML
  pages, 303 API modules, 1,860 legacy API anchors, and 130 redirects with
  zero errors. Build and test commands must run sequentially because they
  share generated content and download directories.

- Source and rendered documentation structure checks, topic graph validation,
  plugin payload checks, Ruff, and whitespace checks passed.

Delivery: the full conversion is included on `codex/docs-topic-explorer` for
review in PR #1654. This change does not merge or publish the website.

## Scientific limits retained

The respiratory-droplet example is an input-identity audit failure, not a
validated respiratory-fluid model. Binned organic coagulation demonstrates a
species-inventory limitation. Chamber dry-phase oscillations remain a numerical
resolution concern. CUDA demonstrations retain their specific workload and
execution provenance; these editorial changes do not establish universal
convergence, experimental validation, or new ensemble statistics. No library
physics implementation changed.

## PR review follow-up

- Fixed standalone structure-test collection: explicitly add the scripts folder
  before importing the checker, so its topic helper resolves independently of
  test collection order. The isolated 17 tests and all 318 infrastructure tests
  passed; all hosted workflows passed at `e10895939`.
- Reproduced the later review's chamber timestamp failure at 8190 seconds with
  dt=0.1 seconds. Validate spacing on local elapsed times, then add the absolute
  offset, retaining the original 1e-12 second tolerance and all mass checks.
  The 13 simulation tests cover zero and late offsets at both timesteps.
  The full CPU workload was refreshed successfully; all 23 captured outputs
  and all eight figure hashes match the previous run.
