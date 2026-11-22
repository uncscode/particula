# Run a parcel experiment

Use a closed parcel to ask how aerosol growth changes the air during ascent.
Use replay to ask how particles respond to a prescribed air history. Connect
several histories when the question concerns transport between parcels.
The [parcel theory](../../theory/thermodynamics/parcel-dynamics.md) gives the
water budgets and frame conventions for all three experiments.

## Record a closed ascent and replay it

Start with a subsaturated parcel. Give each dry-radius class its represented
particle count and specify the fixed mass of dry air. The following CPU
calculation records ten seconds of ascent. It writes a CSV and a sibling JSON
in a temporary directory, reloads them, and compares the final radius of each
class in micrometres. A fresh run needs NumPy, SciPy and Warp for this workflow.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from particula.dynamics.parcel import ParcelInitialState, run_parcel
from particula.dynamics.parcel_replay import ParcelTrace, replay

initial = ParcelInitialState(
    dry_radius=np.array([45e-9, 70e-9, 110e-9]),  # m
    number=np.array([2.5e7, 5e7, 2.5e7]),       # particles per parcel
    kappa=0.5, pressure=90_000.0, temperature=288.15,
    vapor=0.0118, dry_air_mass=1.0,              # kg/kg dry air, kg
)
direct = run_parcel(initial, updraft=1.0, timestep=0.05, steps=200)
with TemporaryDirectory() as directory:
    path = Path(directory) / "ascent.csv"
    ParcelTrace.record(direct, parcel_id="lower").save(path)
    trace = ParcelTrace.load(path)

result = replay(trace, max_step=0.0125)
print("Direct wet radii (µm)", direct.wet_radius[-1] * 1e6)
print("Replay wet radii (µm)", result.wet_radius[-1] * 1e6)
```

In the direct run, `timestep` sets the observation spacing. The default CPU
solver uses internal Radau steps. Warp also supports fixed steps and adaptive
RK23 or ROS3. In replay, `max_step` limits the growth step between forcing
records. To measure forcing error, thin the records with `trace.sample_every(2)`
before assigning interval sources. To measure integration error, keep the
records and reduce `max_step`. [Adaptive ascent comparison](../../studies/examples/thermodynamics/adaptive_moist_parcel.py) · [100-second replay and cadence study](../../studies/examples/thermodynamics/parcel-experiments/parcel_replay.py)

### Read or supply a trace

Each CSV contains one parcel ID and strictly increasing times. Its header is
`parcel_id,time_s,height_m,temperature_K,pressure_Pa,volume_m3,water_kg_per_kg_dry_air`.
The JSON carries the initial aerosol spectrum, initial wet radii, species
ownership, units, sampling rule and provenance. The supplied trace was generated
by Particula. A researcher preparing an external history can use the CSV/JSON
pair in the replay study download as a template. Independent CFD replay remains
an open validation step.

Temperature, pressure, vapor and volume are prescribed. Supersaturation at a
record checks consistency with those inputs. Wet-radius and liquid histories
in the JSON are optional comparisons. Direct `liquid` is kg/kg dry air, while
replay `liquid` is kg per parcel. Multiply the direct result by `dry_air_mass`
before comparing them. In the ten-second example above, the largest class reaches
about 0.3151 µm in both runs. Positive `water_exchange` means water entered the
droplets. Positive `water_restored` means the prescribed vapor reservoir
supplied water to the parcel.

## Move a named vapor between cells

This two-cell control uses the trace above in both cells so transport and uptake
are easy to isolate. Start with 10 µg of organic vapor in the lower cell,
prescribe a flow of 0.01 m³/s into the upper cell, and let a first-order sink
there transfer vapor to particles. The receiver's gas plus particle inventory
must equal the cumulative mass across the face.

```python
from dataclasses import replace
from particula.dynamics.parcel_cluster import Connection, ParcelCluster, run_cluster
from particula.dynamics.parcel_replay import OrganicForcing

lower = replace(trace, organic=OrganicForcing(name="SOA", initial_gas=1e-8))
upper = replace(trace, parcel_id="upper", organic=OrganicForcing(name="SOA", uptake_rate=0.25))
edge = Connection("lower", "upper", flow=0.01)
cluster = ParcelCluster.regular(
    {"lower": lower, "upper": upper},
    shape=(1, 1, 2), widths=(1.0, 1.0, 1.0), connections=[edge],
)
run = run_cluster(cluster, max_step=0.0125)
receiver = run.parcels["upper"]
received = receiver.organic_gas[-1] + receiver.organic_particle[-1]
np.testing.assert_allclose(received, run.transfers["lower", "upper"][-1], rtol=1e-10, atol=0)
```

For distinct histories, supply a trace for each ID that covers the same start
and end times. Regular layouts assign traces in insertion order, with the last
axis changing fastest. Cell widths in metres define the geometry for particle
crossings. Each trace's physical volume sets the gas concentration used for
exchange. Custom layouts can specify centers, directed connections and optional
axis-aligned bounds. Binned particle exchange requires matching dry-radius
classes, hygroscopicity and an explicit common dry-material density. Inspect
`run.inventories_transferred` for the component inventories by directed connection.
[Unequal-volume exchange and CUDA gradients](../../studies/examples/thermodynamics/parcel-experiments/parcel_cluster.py)

For a scheduled organic addition, supply one rate in kg/s or one total in kg
per trace interval using `OrganicForcing.rates` or `.totals`. A total enters
once at its interval's right endpoint. To resume, call `replay` with
`stop_index`, save the returned `checkpoint`, and pass it back with the same
trace, species order, `max_step` and humidity perturbation. Look up gas lanes
by name in `gas_species` when reading `gas_masses`.

## Follow individual particles

Use `advance_motion` with a `MotionState` when the observable is a crossing
time or an individual's destination. Supply cell bounds, a stable ID and
world position for each particle, an air velocity and per-cell capacity.
Each returned event identifies the particle, time, source, destination and
boundary action. At a retaining wall a particle can slide along the face.
On entering a neighboring cell it moves inward if the local velocity points
away from the wall. Exterior faces retain particles by default. Periodic
faces wrap, open faces register departures, and depositing faces record
deposition. In a translating mesh, compare height both above ground and
relative to the moving cells. Carry the returned state into the next interval
to preserve contacts and the geometry phase.

The lofting study runs this workflow on CPU with 1,024 individually identified
particles per layout. Call its `simulate(replicas_per_stratum=16)` to release
2,048 per layout, or supply `device="cuda:0"` for native CUDA. The release
catalog and final destinations share a row for each ID. Smooth humidity and
flow sensitivities use the replay and exchange paths. Discrete crossing runs
reject gradient requests. [Stack, periodic row and translating block](../../studies/examples/thermodynamics/parcel-experiments/cluster_lofting.py)
