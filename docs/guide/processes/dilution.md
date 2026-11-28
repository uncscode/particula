# Dilution Strategy System

Use dilution for the loss of suspended particles and gas when clean air
flushes a well-mixed chamber at fixed volume. The CPU runnable advances each
box of a native `Aerosol`. The direct Warp example shows where to include
dilution in a sequence, and the prepared guide lists supported CUDA
configurations. [Explicit-transfer sequence](../../studies/tutorials/working-with-gpu/direct-kernels/gpu_complete_process_sequence.py) · [Prepared GPU simulations](../gpu/prepared.md)

## Public API and equation

Construct the public strategy and runnable through `particula.dynamics`:

```python
import numpy as np
import particula as par

particles = par.particles.create_resolved_particle_data(
    masses=[1e-18, 2e-18], density=1000.0,
    population=[4.0, 8.0], volume=2.0,
)
gas = par.gas.GasData(
    name=["vapor", "inert"], molar_mass=np.array([0.1, 0.04]),
    concentration=np.array([[3e-6, 5e-6]]),
    partitioning=np.array([True, False]),
)
environment = par.gas.EnvironmentData(
    temperature=[298.15], pressure=[101325.0], saturation_ratio=[[1.0, 0.0]],
)
aerosol = par.Aerosol(particles=particles, gas=gas, environment=environment)

strategy = par.dynamics.DilutionStrategy(coefficient=0.1)
dilution = par.dynamics.Dilution(strategy)
result = dilution.execute(aerosol, time_step=10.0, sub_steps=2)
assert result is aerosol
```

Execution mutates the supplied `Aerosol` and returns that same `Aerosol`
object. For physical chamber volume `V` [m³] and dilution flow `Q` [m³/s],
the scalar coefficient is `alpha = Q / V` [s⁻¹]. The represented particle
sample can occupy a different air volume. Every supported concentration follows
the exact exponential update

`c_new = c * exp(-alpha * time_step)`.

Here `time_step` is in s. Dilution multiplies the stored population in each
slot by the decay factor at fixed represented air volume. This scales counts
for resolved/discrete slots and represented dN/dr in m⁻¹ for radius PDFs. All lanes of
the unified `GasData.concentration` (kg/m³) decay, regardless of partitioning.
The native runnable applies the coefficient independently to every box.
There is no cross-box transport.

## Compare a flush with physical expansion

A clean-air flush replaces sampled air while the sample volume stays fixed.
Start with 100 represented particles and 1 µg of gas in 0.25 m³. The gas
input is therefore 4 × 10⁻⁹ kg/m³. Over a one-second half-life, the CPU
dilution process removes 50 particles and 0.5 µg of gas.

```python
import numpy as np
import particula as par

sample_particles = par.particles.create_resolved_particle_data(
    masses=[1e-18], density=1000.0, population=[100.0], volume=0.25,
)
sample_gas = par.gas.GasData(
    name=["tracer"], molar_mass=np.array([0.1]),
    concentration=np.array([[4e-9]]), partitioning=np.array([False]),
)
sample_environment = par.gas.EnvironmentData(
    temperature=[298.15], pressure=[101325.0], saturation_ratio=[[0.0]],
)
flushed = par.Aerosol(
    particles=sample_particles.copy(), gas=sample_gas.copy(),
    environment=sample_environment,
)
loss = par.dynamics.Dilution(par.dynamics.DilutionStrategy(coefficient=np.log(2)))
loss.execute(flushed, time_step=1.0)
np.testing.assert_allclose(flushed.particles.population, [[50.0]])
np.testing.assert_allclose(flushed.particles.number_density, [200.0])
np.testing.assert_allclose(flushed.gas.concentration, [[2e-9]], atol=0)
np.testing.assert_allclose(flushed.gas.concentration * flushed.particles.volume[:, None],
                           [[0.5e-9]], atol=0)
```

Expanding the original air sample to 0.5 m³ gives the same final number and
gas densities. The resolved Warp prescribed-volume step retains all 100
particles and 1 µg of gas. With Warp installed, continue from the original
`sample_particles` and `sample_gas` above. Choose `"cpu"` to execute the Warp CPU path, or
`"cuda:0"` to execute on CUDA hardware.

```python
import warp as wp
from particula.gpu import from_warp_particle_data, to_warp_gas_data, to_warp_particle_data
from particula.gpu.kernels.communication import volume_evolution_step_gpu

device = "cpu"
expanded = to_warp_particle_data(sample_particles, device=device)
expanded_gas = to_warp_gas_data(sample_gas, device=device)
volume_evolution_step_gpu(expanded, expanded_gas,
                          wp.array([0.5], dtype=wp.float64, device=device))
native = from_warp_particle_data(expanded)
np.testing.assert_allclose(native.population, [[100.0]])
np.testing.assert_allclose(native.number_density, [200.0])
np.testing.assert_allclose(expanded_gas.concentration.numpy(), [[2e-9]], atol=0)
np.testing.assert_allclose(expanded_gas.concentration.numpy() * native.volume[:, None],
                           [[1e-9]], atol=0)
```

A manual particle-volume edit leaves gas concentration at its previous value.
It therefore gives a different gas inventory when the new volume is applied.
[Replacing a measured particle sample](../migration/v0.4.0/index.md#import-and-replace-a-measured-sample)

## Execution, validation, and recovery

`sub_steps` must be a positive, non-boolean Python or NumPy integer. For the
concrete exact-exponential strategy, the shared runnable driver executes equal
`time_step / sub_steps` slices. Each slice commits only after every box succeeds;
earlier successful slices remain committed if a later slice fails. Custom
compatible strategies retain their own validation and execution guarantees.
The coefficient and duration must be finite, nonnegative real scalars;
complex values are rejected, and booleans and non-scalars are rejected.

A zero coefficient or zero duration is an exact no-op, but supported state is
still preflighted even for a no-op. The direct concrete strategy operates on one
box; `Dilution` stages all boxes through the shared runnable driver. If validation fails, state is unchanged; if an unexpected
CPU write failure occurs, it restores attempted concentration writes from
snapshots. Held containers and destination-array identities are retained.
A failed substep can be retried after correcting its inputs; do not replay
already completed substeps. Extreme finite decay may underflow to zero, but outputs remain
finite and nonnegative.

Dilution changes only particle number concentration and gas mass
concentrations. It protects particle distribution/mass, charge, density,
representation volume, gas names, molar masses, partitioning metadata, and
environment temperature, pressure and saturation ratios.

## Boundaries and example

The CPU runnable applies the same prescribed coefficient independently to
each box. Inlet composition, sources, and transport between boxes require
separate processes.

`dilute_aerosol` and `get_dilution_step` are concrete-module-only helpers;
they are not public `particula.dynamics` imports. Use the public strategy and
runnable above for supported execution.

The deterministic, hardware-free example uses only this public API:

- Source: <https://github.com/Gorkowski/particula/blob/main/docs/studies/examples/particle-processes/sources-and-dilution/cpu_dilution.py>
- Run: `python docs/studies/examples/particle-processes/sources-and-dilution/cpu_dilution.py`

Focused regression commands are:

```bash
pytest examples_tests/dilution_example_test.py -q -Werror
pytest particula/dynamics/tests/dilution_test.py -q -Werror
pytest particula/dynamics/tests/dilution_runnable_test.py -q -Werror
pytest particula/dynamics/tests/dilution_exports_test.py -q -Werror
```
