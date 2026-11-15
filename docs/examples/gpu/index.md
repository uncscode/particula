# GPU examples

The simulation examples publish saved figures from actual CUDA runs; the
scaling report renders saved measurements on the CPU.
Start with a native CPU aerosol, prepare supported processes on the GPU, and
bring detached results back for analysis. Install Particula and its plotting
extras, then read the [prepared workflow guide](../../guide/gpu/prepared.md).
Warp is a declared package dependency; CUDA also needs an available NVIDIA GPU
and compatible driver.
There is no CPU fallback in publication runs.

## Choose a route through the studies

If you want to build a study, begin with the prepared quick start, then choose
one of three comparisons: controlled environmental response, repeated stochastic
histories, or source/forcing interactions. If you want to understand execution
internals, use the direct and resident examples after the public workflow.
Their deliberately small fixtures are interface demonstrations, not calibrated
physical populations. The [topic explorer](../../guide/topics/index.md) connects
these execution choices back to growth, collisions, charge, and verification.

## Prepared simulations

| Example | Scientific or workflow question |
| --- | --- |
| [CPU → CUDA → CPU quick start](prepared/gpu_prepared_quick_start.py) | How do eight aerosol boxes evolve, change forcing, capture and resume? |
| [Custom process backends](prepared/custom_process_backends.py) | Can an imposed warming law agree with an analytic solution across CPU, CUDA, transfer, and restart? |
| [Charged Brownian models](prepared/gpu_charged_models.py) | How do five charged fits compare on CUDA against their CPU implementations? |
| [Explicit atmospheric ion balance](prepared/gpu_ion_balance.py) | Do ion production, recombination and neutral-product export reproduce finite-population balance on CUDA? |
| [Ground to mesopause and high pressure](prepared/gpu_dry_environment_sweep.py) | How do dry charged-aerosol size, charge and collision histories change from 0–86 km and from 1–10 atm? |
| [Humid aerosols and explicit ions](prepared/gpu_environment_sweep.py) | How do 2.6 million aerosols and ions exchange water, charge, attach and coagulate across sixteen environments? |
| [The lucky few](prepared/gpu_lucky_few.py) | How often does a 3 nm seed reach 50 nm before background capture, under two explicit ion-attachment assumptions? |
| [Heterogeneous boxes](prepared/gpu_prepared_multibox.py) | How do 64 boxes with different thermodynamic conditions, volume and composition respond? |
| [Cooling, heating and vapor forcing](prepared/gpu_prepared_forcing.py) | How does a four-material/five-gas population respond to an external schedule? |
| [Stochastic coagulation ensemble](prepared/gpu_prepared_stochastic.py) | How much does evolution vary across sixteen random seeds? |
| [Million-particle lifecycle](prepared/gpu_prepared_lifecycle.py) | Can 1,572,864 active particles use all five processes with exact capture/restart agreement? |
| [Repeated source pulses](prepared/gpu_prepared_cyclic_chamber.py) | How do new cohorts, Brownian collisions and cyclic condensation interact over 600 seconds? |

The heterogeneous example varies several conditions together and does not
identify a temperature-only effect. Ensemble shading is a distribution across
seeds, not a confidence interval. Large counts establish execution and accounting;
physical predictions still require appropriate model choice and convergence.

The [RTX 5060 scaling report](prepared/gpu_scaling_results.py) separates
resident throughput, complete-workflow costs, old/new Brownian GPU execution
and memory. Its small downloadable data contains plotted summaries and displayed provenance,
not model states or image arrays.

## Direct kernels and concrete resident interfaces

| Example | Scope |
| --- | --- |
| [Direct quick start](direct/gpu_direct_kernels_quick_start.py) | Manual transfers and a first condensation step. |
| [Direct Brownian coagulation](direct/gpu_coagulation_direct.py) | Size distribution and conserved mass after a direct call. |
| [Complete direct sequence](direct/gpu_complete_process_sequence.py) | Five explicit boundaries; the nucleation call has zero duration. |
| [Condensation parity](direct/gpu_condensation_parity_walkthrough.py) | Fixed-four-substep numerical oracle, transfer and energy diagnostics. |
| [Direct nucleation](direct/gpu_direct_nucleation.py) | Unit-count events in boxes with different physical volumes. |
| [Resident checkpoint lifecycle](resident/gpu_resident_session.py) | State ownership and exact restart, without advancing physics. |
| [Resident multi-timestep loop](resident/gpu_resident_multi_timestep.py) | An open-system sequence with explicit forcing and sinks. |
| [Native graph capture](resident/gpu_resident_graph_capture.py) | Captured resident-loop execution and physical outputs. |

For the public introduction, run:

```bash
python docs/examples/gpu/prepared/gpu_prepared_quick_start.py
```

For the illustrated CUDA path in an older direct/resident source, pass `--cuda`:

```bash
python docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py --cuda
```

Older sources retain portable unflagged regression helpers. Their Warp CPU or
no-Warp behavior is separate from published CUDA evidence. A clean optional skip
does not establish GPU correctness or speed.

## Turn a result into a next experiment

First identify which comparison the page actually makes. A heterogeneous scan
can reveal diverse response without isolating a cause; a seed ensemble can
quantify sampling spread without measuring parameter uncertainty; a capture
comparison can establish repeatability without measuring speed. Choose your
next control accordingly, and compare the scientific diagnostic before timing it.

For example, hold the initial population fixed while changing only a collision
seed, or retain the same physical forcing times while halving a timestep.
Document external sources and sinks before interpreting inventory differences.
The individual studies close with more specific extensions and their evidence
requirements.

## Deliberate result refresh

Site builds render saved results without executing simulations. Refresh a CUDA
example explicitly, then review and commit its complete result bundle:

```bash
python scripts/docs_examples.py refresh gpu-prepared-quick-start
python scripts/docs_examples.py refresh gpu-prepared-cyclic-chamber
```

Large scientific studies may take longer than ten minutes; their registry
entries allow up to two hours. CI checks source, dependency freshness and saved
file integrity without executing these CUDA studies. Download the helper ZIP
for examples that use shared study code, then run `example.py` from its extracted
folder. See [authoring guidance](../../../website/AUTHORING.md) and
[validation records](../../contribute/validation/index.md).
