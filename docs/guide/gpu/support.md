# GPU support

Use [prepared simulations](prepared.md) for the native CPU aerosol → resident
CUDA computation → CPU analysis workflow. Direct kernels and the concrete
resident interfaces remain available for callers who need explicit resource
control. Warp is a declared package dependency. CUDA availability is a separate
hardware/runtime requirement; requesting CUDA never selects Warp CPU silently.

<span id="support"></span>
<span id="current-shipped-support-boundaries"></span>

| Area | Support | Boundary |
| --- | --- | --- |
| Native CPU storage | Multi-box particle, gas and environment containers | Explicit leading box axis and population metadata. |
| Native CPU dynamics | Five batched runnables through the shared box loop | Direct strategy methods remain single-box. |
| Prepared runnables | Ordered subsets of condensation, Brownian/charged/settling/shear coagulation, dilution, wall loss, nucleation, atmospheric ion collisions, ion-pair production and neutral ion-product cleanup | One occurrence per family; exact supported configuration, fixed capacity and timestep. |
| Native gas chemistry | Registered resident RK23 and ROS3 for closed gas reactions | Explicit layout binding, bounded per-box scratch, prescribed temperature, same-device restart. [Setup and units](../processes/gas-chemistry.md) |
| Resolved particle chemistry | Registered resident RK23 with bulk reactions and finite shared gas uptake | Explicit material bindings, same-device checkpoint and CUDA graph replay. [Setup and units](../processes/resolved-chemistry.md) |
| CPU → CUDA | `runnable.prepare(aerosol, device="cuda:0", time_step=...)` | Validates, uploads once, owns private resources; initial CPU state is unchanged. |
| CUDA → CPU | `simulation.snapshot()` or reduced `diagnostics()` | Explicit synchronized, detached observations; full snapshots preserve native names and measures. |
| Forcing | Environment and full-gas updates between segments | Explicit uploads; condensation owns derived saturation. |
| Seed cohort observations | Original-lineage threshold, background capture and censoring | Unit-count populations; supported process compositions and explicit initial material markers. |
| Continuation | `checkpoint()` and matching runnable `resume()` | Exact device/configuration and canonical bytes; no automatic CPU fallback. |
| Native CUDA capture | `capture()`, then ordinary `step/run` | Non-advancing capture; fixed bindings; fresh capture after resume. |
| Direct kernels | Supported process-specific fixed-shape interfaces | Caller owns transfers, scratch, species bindings and RNG sidecars. |
| Continuous adaptive condensation | Multi-material RK23/ROS3 with ideal, κ, binary BAT and optional direct AIOMFAC activities and surface phases | `GPUCondensationODE` uses host control; `GPUResidentCondensation` uses native CUDA conditional graphs for adaptation. |
| Direct AIOMFAC condensation | Fixed-step and adaptive neutral exchange from the full dissolved composition with `particula[aiomfac]` | Explicit analytical material map, inert lanes, finite vapor reservoirs and same-device restart. [Two-vapor example](../../studies/examples/thermodynamics/activity/aiomfac_closed_box.py) |
| Sparse AIOMFAC neutral activities | Fixed-step box condensation, host or resident RK23 and ROS3 boxes, and resident RK23 and ROS3 multi-vapor parcels | Select `evaluation="hybrid"` and a matching `table`. Stage fluxes use guarded lookup. ROS3 evaluates the physical Jacobian with direct CUDA activities. [Measured accuracy, timings and activation](../../studies/validation/condensation/aiomfac-sparse/index.md) |
| Multi-vapor rising parcel | Explicit wet classes, finite vapor mixing ratios and shared surface heat balance through `run_parcel` | Constant updraft, continuum transport, unit accommodation and resident RK23/ROS3. [Parcel example](../../studies/examples/thermodynamics/parcel-experiments/aiomfac_multivapor_parcel.py) |
| Coupled adaptive simulation | `GPUCoupledSimulation(..., resident=True)` | Fixed event schedule and persistent RNG; device-only stage handoff, explicit observations and boundary restart. |
| Warp CPU | Separate regression target | CPU evidence is labeled separately from actual CUDA execution and performance. |

Prepared coagulation are particle-resolved and require
equal active counts. Prepared nucleation injects neutral unit-count events with integral
demand and explicit capacity policies. Their evolving compatibility matters;
see [failure boundaries](prepared.md#evolving-compatibility-and-failure).
Prepared atmospheric coagulation additionally requires unit counts, reference
ion masses, and the fixed LF2013 air conditions and fitted domains described in
[atmospheric ion collisions](prepared.md#atmospheric-ion-collisions).
Prepared condensation supports a bounded set of thermodynamic configurations and
four equal fixed substeps; it does not imply general CPU-strategy equivalence.
For the separate opt-in adaptive owner, see
[numerical integration](../processes/numerical-integration.md) and its measured
accuracy and transfer costs. It does not change prepared simulation semantics.

## Direct-kernel details

Direct and prepared coagulation support all 15 nonempty subsets of Brownian,
charged Brownian, SP2016 settling and ST1956 shear. There is no fixed physical
population cap. Exact tiled pair reductions have bounded scratch but quadratic
work; explicit trial and collision budgets reject overloaded steps before
writers. Equal active represented counts remain required. Prepared settling
preserves native mean density, while direct settling uses particle effective
density. No CPU bin interpolation is introduced.
See [configuration](direct/coagulation-configuration.md).

Direct collision outputs and persistent RNG are caller-owned, same-device
resources. Omitted optional sidecars may allocate call-local storage. Supplied
RNG advances in place; `initialize_rng=True` explicitly resets it. These are
execution resources, not new native container fields.

`ParticleData.volume` owns per-box physical volume. `EnvironmentData` owns CPU
temperature, pressure and saturation. Device vapor pressure is derived helper
state and has no native CPU `GasData` field. Direct restore therefore needs
externally retained gas names and drops vapor pressure; prepared snapshots retain
the names automatically, while checkpoints preserve continuation helper state.

Direct validation may synchronize status flags or scans. Prepared steps retain
validated resources and defer observation to explicit boundaries. Neither path
adds hidden backend fallback, general adaptive stepping, dynamic storage resizing,
a new PDF algorithm, unequal-multiplicity coagulation or multi-GPU execution.

Explicit ions also support the opt-in `attachment_model="mobility"` transport
approximation at variable temperature and pressure. LF2013 remains the default
and retains its reference-condition restrictions. See
[variable-environment ion attachment](prepared.md#variable-temperaturepressure-ion-attachment)
for mobility scaling, fixed recombination, and omitted physical processes.

Direct and prepared CUDA also support the complete dispersion and dielectric
thermal mechanism, alone or with settling and shear. The tested range and
strong opposite-charge boundary are recorded in the
[dielectric capture validation](../../studies/validation/coagulation/dispersion/index.md).
