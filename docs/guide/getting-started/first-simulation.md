# First CPU simulation

Follow a finite vapor reservoir as it grows 100 nm organic particles over ten
minutes. The [complete executable example](../../studies/tutorials/first_cpu_workflow.py)
creates native state, records a time series, saves and reloads results, and
checks an exact restart at five minutes. Its published page includes the plots.

## Run and save the experiment

Start with Python 3.12 or newer and an active virtual environment. To run
the release example, install Particula with plotting support, save the
[version 0.3.5 script](https://raw.githubusercontent.com/uncscode/particula/v0.3.5/docs/studies/tutorials/first_cpu_workflow.py)
as `first_cpu_workflow.py`, and run it from the directory where you saved it:

```bash
python -m pip install "particula[plot]==0.3.5"
python first_cpu_workflow.py
```

If you are working from a repository checkout, run its matching package and
example together from the repository root:

```bash
python -m pip install -e ".[plot]"
python docs/studies/tutorials/first_cpu_workflow.py
```

The script creates `first-cpu-results/` containing a growth figure, an analysis
file and a separate restart file for each of two initial vapor concentrations.
The publication pipeline uses temporary storage and retains only compact plot
data, never the full restart files.

The first case has 1000 particles cm⁻³, 10 µg m⁻³ initial organic vapor,
298.15 K and 101325 Pa. One representative carries 1000 particles in a volume
of 1 cm³: native resolved `concentration` is a count, while `number_density`
is m⁻³. The second case doubles only initial vapor to 20 µg m⁻³. Both are
illustrative single-material organic calculations with explicit constant
properties, ideal activity and Kelvin curvature. They exclude dilution,
coagulation, wall loss, heat feedback and external forcing.

The adaptive RK3(2) method advances particle and gas mass together. The script
sets `rtol=1e-5`, per-particle `particle_atol=1e-27` kg and
`gas_atol=1e-15` kg m⁻³. It records every ten seconds by landing on those times.
This can shorten a proposed step. These are local numerical controls, not
experimental uncertainty or a guaranteed global error bound.

## Reload and inspect results

Run this from the directory containing `first-cpu-results/`:

```python
import numpy as np
from particula.execution.analysis_io import load_analysis

result = load_analysis("first-cpu-results/organic-10.analysis")
time_s = result.arrays["time"]
gas = result.arrays["gas_concentration"][:, 0, 0]
particle = result.arrays["particle_mass_density"][:, 0, 0]
np.testing.assert_allclose(gas + particle, gas[0] + particle[0], rtol=2e-13)
print(result.metadata["configuration"])
print("Final gas concentration:", gas[-1] * 1e9, "µg m^-3")
```

Analysis output includes units, named axes, ordered gas species, population
measure, volume, physical configuration, numerical settings and source/package
identity. Histories contain selected diagnostics; full particle arrays are
opt-in and storage is bounded. Analysis files cannot be used as exact restarts.
The files use ZIP, JSON and numeric arrays and never require pickle to load.

The saved calculation reaches radii of about 136.69 nm and 162.13 nm for the
10 and 20 µg m⁻³ cases. Remaining vapor is about 0.89 and 0.87 µg m⁻³.
More initial vapor supplies more particle material; growth slows as the vapor
approaches its curvature-dependent equilibrium. Number remains constant and
gas plus particle mass closes to floating-point precision. Conservation and
successful restart verify bookkeeping; they do not establish agreement with a
real organic aerosol or an independent numerical reference.

## Resume the five-minute checkpoint

Start Python in the directory containing both the downloaded script and
`first-cpu-results/`. For the checkout route, start Python from the repository
root with `PYTHONPATH=docs/studies/tutorials python` (Bash). The example
supplies the same physical configuration used to write the checkpoint.
Load the saved analysis to compare the gas history from 300 to 600 seconds:

```python
import numpy as np
from particula.execution.analysis_io import load_analysis
from first_cpu_workflow import build_case
from particula.execution.cpu_restart import load_cpu_checkpoint, resume_condensation
from particula.execution.recording import Recorder, run_until

result = load_analysis("first-cpu-results/organic-10.analysis")
_, strategy = build_case(10.0)
saved = load_cpu_checkpoint("first-cpu-results/organic-10.restart")
resumed = resume_condensation(saved, strategy)
history = Recorder()
run_until(
    resumed, 600.0,
    output_times=np.linspace(300.0, 600.0, 31),
    recorder=history,
)
np.testing.assert_array_equal(
    history.arrays()["gas_concentration"],
    result.arrays["gas_concentration"][30:],
)
```

The checkpoint retains accepted time, scaled native state, tolerances, the
proposed next step and solver work counters. A different physics configuration
or incompatible schema/runtime rejects. For this deterministic workflow, the
same subsequent boundaries give exact continuation. Arbitrary callable forcing
has no supported disk identity. Passive recording does not alter physics, but
choosing different output boundaries can alter the numerical trajectory.

## Adapt one scientific input

The example already runs the controlled 10 versus 20 µg m⁻³ comparison. Change
`CONFIGURATION["saturation_vapor_pressure_Pa"]` to investigate volatility next,
then rerun and compare both remaining vapor and radius. Select a new output
directory to preserve earlier files. Before interpreting a named compound,
justify its material properties and compare with independent measurements.

Next: [recording, forcing and restart](recording-restart.md),
[native construction](../data-model/construction.md),
[ownership](../data-model/ownership.md), and
[condensation](../processes/condensation.md). Process combinations and broader
size distributions need their own numerical and scientific checks.

Browse [all runnable Python scripts](../../studies/scripts.md) to find another workflow by topic or required device.
