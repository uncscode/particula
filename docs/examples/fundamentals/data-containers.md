# Data Containers

What must remain meaningful when an aerosol state moves between storage
backends? Shape alone is insufficient: represented counts, physical volume,
material ordering, and species identity determine the scientific interpretation.
This page supports the [runnable container example](data_containers.py), helping
you choose between CPU construction and explicit Warp transfers.

Run the published entrypoint from the repository root:

```bash
python docs/examples/fundamentals/data_containers.py
```

The top-level script above is the canonical runnable example. This page is
supporting context only.

The CPU container portion always runs. Warp-backed particle and gas round
trips run only when Warp is available, and they use `device="cpu"` so CUDA is
not required. Enabled transfer failures propagate; they are not reported as
successful CPU-only runs. Set `PARTICULA_EXAMPLE_FORCE_NO_WARP=1` to deliberately
inspect CPU construction without requesting transfers.

The particle example declares `particle_resolved`: storage contains represented
counts `[1, 0.25]` at physical volume `1e-6 m³`. Dividing once by that volume
produces number densities `[1e6, 2.5e5] m⁻³`. Particle masses are per-particle
material masses; multiply by counts and divide by volume for mass density.
Explicit round trips preserve the population declaration and raw counts.

For the low-level direct condensation path, use the separate canonical
quick-start:

```bash
python docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py
```

That runnable script is the direct GPU condensation example. It
demonstrates explicit `to_warp_*` / `from_warp_*` transfers, lazy imports from
`particula.gpu.kernels`, two fixed-four-substep condensation calls, and
caller-owned fp64 scratch, physical-property, latent-heat, and energy sidecars
reused on Warp `device="cpu"` by default. It does not invoke coagulation or
configure RNG state.

For the bounded, low-level particle-resolved Brownian coagulation path, run:

```bash
python docs/examples/gpu/direct/gpu_coagulation_direct.py
```

This standalone example explicitly transfers `ParticleData` to Warp, defaults
to Warp `device="cpu"`, and makes two supported Brownian direct calls. Its
collision-pair, collision-count, and persistent RNG-state sidecars are
caller-owned and reused across those calls before an explicit CPU checkpoint
restore. Warp imports and all conversion/kernel work are skipped when Warp is
unavailable or `PARTICULA_EXAMPLE_FORCE_NO_WARP=1`; this disabled route has no
CPU coagulation fallback. The example is a direct-kernel path, not a Runnable
API.

- [Feature guide: Data Containers and GPU Foundations](../../guide/data-model/index.md)
- [Container construction and transfer example](data_containers.py)
- [Direct kernel quick start](../gpu/direct/gpu_direct_kernels_quick_start.py)
- [Direct coagulation example](../gpu/direct/gpu_coagulation_direct.py)

## Check the meaning of a round trip

The published container example prints shapes and preserves explicit metadata
in its transfer construction. A successful transfer is not a simulation result
or a numerical parity study. For your own population, compare each transferred
field and its count-weighted material inventory, and retain the gas-species names
used by the caller. Shapes can agree even when species order is wrong.

Once storage semantics are clear, the [state-access example](native_state_access.py)
shows how live arrays and detached copies affect a control calculation. That
connection matters when comparing two physical scenarios: a shared mutable
container can change the nominal reference along with the experiment.
