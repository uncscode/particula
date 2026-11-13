# Dilution And Wall Loss

### Direct GPU dilution

Import the direct dilution operation from the low-level kernel namespace:

```python
from particula.gpu.kernels import dilution_step_gpu
```

This is distinct from the CPU↔Warp transfer helpers in `particula.gpu`.
Callers own conversion, device placement, synchronization and checkpoint
transfer, and the `WarpParticleData` and `WarpGasData` instances. The kernel
does not provide hidden transfer or CPU fallback.

`dilution_step_gpu` is a fixed-shape, caller-owned in-place operation. It
returns the same particle and gas objects and changes only their
`concentration` fields; masses and every other caller-owned particle or gas
field remain unchanged. It applies `alpha = Q / V` in `s^-1` and
`c_new = c * exp(-alpha * time_step)`. The coefficient is either a finite,
nonnegative scalar or a caller-owned, same-device `wp.float64` Warp array with
shape `(n_boxes,)`.

Entry preflight is ordered and read-only. Rejected calls are atomic before
kernel launch. A zero scalar coefficient or zero time step completes full
preflight and is a write-free, no-update-kernel no-op; preflight validation
scans may still allocate or launch. The contract does not promise rollback
after a launched-kernel failure.

CPU dilution supplies the upstream CPU finite-step oracle; direct-process integration is the planned
integrated direct-call consumer. Independent NumPy comparisons of particle and
gas concentrations run on Warp CPU with float64 `rtol=1e-12, atol=0`; CUDA is
optional and skips cleanly when unavailable. This tolerance-based evidence is
not bitwise parity.

### Direct GPU wall loss

The bounded direct wall-loss path is particle-resolved. It supports neutral and
charged execution. Import the step and its concrete-module-only configuration
separately; the configuration is not
re-exported by `particula.gpu.kernels` or `particula.gpu`:

```python
from particula.gpu.kernels import wall_loss_step_gpu
from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

# Use the native container construction immediately above; no hidden transfer.
particles = to_warp_particle_data(particle_data, device="cpu")
rng_states = wp.zeros(1, dtype=wp.uint32, device="cpu")
config = NeutralWallLossConfig(
    geometry="spherical",
    wall_eddy_diffusivity=1.0e-4,  # m^2/s
    chamber_radius=0.5,  # m
)
wall_loss_step_gpu(
    particles,
    temperature=298.15,  # K
    pressure=101325.0,  # Pa
    time_step=1.0,  # s
    config=config,
    rng_states=rng_states,
    initialize_rng=True,
)
```

The snippet is illustrative: `particles` and `rng_states` are caller-owned
same-device Warp state. Later calls reuse `rng_states` and omit
`initialize_rng=True`. Direct `temperature` and `pressure` each accept a scalar
or same-device `(n_boxes,)` Warp array. Alternatively, pass
`environment=...` and set both `temperature=None` and `pressure=None`; direct
values and `environment=` are mutually exclusive.

Wall eddy diffusivity is in m²/s, dimensions are in m, temperature is in K,
pressure is in Pa, and time is in s. Spherical configurations require a
positive radius and no dimensions; rectangular configurations require no radius
and exactly three positive dimensions. The neutral coefficients use the
Crump--Seinfeld spherical and rectangular chamber relations, including turbulent
deposition and gravitational settling:

- Crump, J. G., & Seinfeld, J. H. (1981), *Journal of Aerosol Science*, 12(5).
  https://doi.org/10.1016/0021-8502(81)90036-7
- Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982), *Aerosol Science and
   Technology*, 2(3), 303--309.
   https://doi.org/10.1080/02786828308958636

`NeutralWallLossConfig` remains the sole concrete configuration type. Its
`mode` is either `"neutral"` or `"charged"`. Charged mode validates a finite,
signed scalar `wall_potential` in V. For spherical geometry,
`wall_electric_field` is a finite, signed scalar in V/m. For rectangular
geometry, it is a caller-owned, same-device `wp.float64` Warp array of shape
`(3,)`, with finite signed components in V/m. The boundary does not replace or
mutate that rectangular field array.

For charged mode, slots with finite nonzero charge use the private
image-charge and electric-field-drift coefficient composition. Image-charge
enhancement is active even when `wall_potential` is zero. Zero-charge slots
retain the exact neutral coefficient and RNG path, including when the charged
configuration supplies nonzero potential or electric field.

After read-only preflight, eligible finite positive-rate fixed slots survive with
`exp(-k * time_step)`. A selected slot has every mass lane, concentration, and
charge cleared. Density, volume, dtype, device, capacity, and unselected
storage are preserved; inactive or unusable slots are neither sampled nor
reactivated. The asynchronous call mutates caller-owned state in place. Callers
own CPU↔Warp transfer, device placement, synchronization, and any checkpoint.
There is no hidden CPU checkpoint transfer or fallback. Rejected pre-launch
calls do not mutate caller-owned state, while rollback is not promised after a
mutation kernel launches. Zero time completes preflight but is write-free.

Omitted RNG state is private to each successful nonzero call. Supplied
same-device `(n_boxes,)` `wp.uint32` state mutates in place;
`initialize_rng=True` is the only reset, so repeating `rng_seed` alone does not
reset persistent state. Consumption is sequential per box over eligible finite
positive-rate slots; positive infinite-rate removal is deterministic and
consumes no draw. Exact CPU/Warp or per-seed RNG replay is not promised.

| Scope | Status |
| --- | --- |
| Neutral and charged particle-resolved spherical/rectangular direct execution | Supported |
| Charged nonzero-charge image/electric-field composition; zero-charge neutral fallback | Supported |
| Charged configuration and rectangular field ownership semantics | Supported |
| Geometry-specific deterministic coefficient tolerances and validation eight-stratum exact-binomial survival evidence (4,096 observations per stratum; inclusive bounds with per-stratum alpha 1.25e-7) | Supported bounded evidence |
| CUDA validation | Optional additive evidence; guarded skips are expected when unavailable |
| CPU fallback or hidden transfer; high-level runnable, scheduler, or backend integration | Deferred (direct-process integration covers integration/closeout) |
| Dynamic slots, compaction/activation, graph capture, differentiability, performance guarantees, or exact RNG replay | Deferred |

Warp CPU is the baseline when Warp is installed; CUDA is optional and skips
cleanly when unavailable. The evidence is not CPU-strategy parity or per-seed
trajectory replay.

These direct wall-loss checks do not by themselves establish orchestration,
capture, autodiff or performance evidence. Native resident integration has its
own [walkthrough](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/resident/gpu_resident_multi_timestep.py), and direct GPU
nucleation has its own [source contract](../../processes/nucleation.md#direct-warp-low-level-step).
The CPU-only, one-box `Nucleation` runnable remains a separate high-level path.

