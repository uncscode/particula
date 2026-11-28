# Calibrate instrument signals and prescribe a measurement product

A particle counter reports a detector count rate. To calculate particles per
cubic centimetre of sampled air, pair that rate with the air flow through the
counter during each reading. A Particula measurement process binds its inputs
by physical quantity and unit. The same process can use another lab's flow
meter when its product satisfies that contract.

Install the CPU measurement dependencies with `pip install "particula[data]"`.
The Python snippets below run in order from the Particula checkout and use the
synthetic counter and flow-meter files in `docs/data/measurement-processing/`.
The [worked two-instrument example](../../studies/examples/calibrated_particle_count.py)
includes those files in its download. See the [import guide](measurements.md)
to adapt a profile to a real logger.

## Register the voltage-to-flow calculation

Suppose a calibration certificate relates voltage $U$ to sample flow $Q$
through $Q = aU + b$. Express $a$ in L/min/V and $b$ in L/min. At 2 V,
$a = 0.30$ L/min/V and $b = 0$ L/min give 0.60 L/min. Register a callable
that returns a Polars expression. At execution its arguments are expressions
for measured values and the coefficients in effect at each timestamp.
The caller supplies the registry and chooses the Python code it contains.

```python
from pathlib import Path

from particula.data import (
    Calibration, Deployment, FieldContract, MeasurementProcess, OperationRegistry,
)
from particula.instruments import Instrument, ParticleCounter

def linear_flow(signal, *, slope, intercept):
    return slope * signal + intercept

registry = OperationRegistry()
registry.register(MeasurementProcess(
    "lab.flow.linear", "1",
    inputs={"signal": FieldContract("voltage", "V")},
    output=FieldContract("volumetric_flow", "L/min"),
    output_field="sample_flow",
    implementation=linear_flow,
    coefficient_units={"slope": "L/min/V", "intercept": "L/min"},
))
```

Give the operation a qualified name and version. The registry rejects a
duplicate pair. If the implementation or its interpretation changes, give
the new implementation a new version for replay. `registry.describe()`
returns the role names, quantities, units, execution mode and parameter
schemas as ordinary data. For a custom process with additional settings,
supply its JSON Schema through `parameter_schema`. Use
`FieldContract("voltage", "V", field="sensor_voltage")` when an export has
several voltage fields and only one can drive the calibration.

## Bind a deployment and select the certificate

Keep the certificate's source, units, effective dates and reference with
the flow meter that it calibrates. This configuration uses one certificate
throughout the synthetic experiment. The half-open interval includes its
UTC start and ends just before its UTC stop.

```python
source = Path("docs/data/measurement-processing")
certificate = Calibration(
    reference="flow-cal-17", operation="lab.flow.linear",
    start="2024-01-01T00:00:00Z", stop="2024-01-01T00:00:05Z",
    coefficients={"slope": 0.30, "intercept": 0.0},
    units={"slope": "L/min/V", "intercept": "L/min"},
    source="Synthetic bench certificate 17",
)
meter = Instrument(
    source / "flow-profile.yaml",
    deployment=Deployment("flow-meter", inlet="inlet", volume_basis="ambient",
                          calibrations=(certificate,)),
    registry=registry,
)
counter = ParticleCounter(
    deployment=Deployment("counter", inlet="inlet", volume_basis="ambient"),
    registry=registry,
)
voltage = meter.read(source / "flow.csv", as_product="voltage")
rate = counter.read(source / "counter.csv", as_product="rate")
calibrate = meter.process("lab.flow.linear", version="1").configure(
    calibration="flow-cal-17"
)
flow = calibrate.execute(signal=voltage)
```

If several revisions belong to a certificate series, give them distinct
`reference` values and a common `group`. Pass that group to `configure`.
Each reading receives the coefficients for its timestamp and retains the
selected reference. An observation at a shared boundary uses the later
certificate. A gap covering an observation or an overlap in the selected
records raises an error. Register different coefficients for a second
deployment even when it uses the same processing code.

For a new delimited instrument, pass its YAML or JSON profile to `Instrument`.
Declare the physical quantity, source unit and output unit for each channel.
A logger that writes millivolts can declare `mV` at the source and `V` at the
output. For a different file format, register the lab's decoder once with
`registry.register_reader("lab.binary", read_binary)` and pass
`reader="lab.binary"` to `Instrument`. The callable accepts
`source, *, profile, **limits` and returns a validated `Measurement` carrying
the scan data, diagnostics and source manifest. The instrument's profile
still describes the product. To select a previously named product explicitly,
use `meter.product("voltage")`.

## Pair the count rate with the calibrated sample flow

Inspect `registry.available_operations(rate)` to find which roles the
counter product can fill. For `particula.particle_counter.number_concentration`,
the count rate fills `rate` and the operation also requires `sample_flow`.
The first role supplies the output timestamps. The built-in operation
joins on exact UTC timestamps, requires unique matches, and checks that both
deployments declare the same inlet and volume basis. For signals integrated
over different acquisition intervals, verify that their sample exposures
describe the same air. Coincident timestamps alone establish only clock
agreement.

```python
derive = counter.process(
    "particula.particle_counter.number_concentration", version="1"
)
concentration = derive.execute(rate=rate, sample_flow=flow)
print(concentration.frame.select(
    "time_utc", "number_concentration", "quality_flags", "calibration_refs"
))
```

The operation converts volumetric flow to cm³/s. At 0.60 L/min the
counter samples 10 cm³/s. A rate of 100 counts/s gives 10 cm⁻³. In the
[worked calibration change](../../studies/examples/calibrated_particle_count.py),
doubling the calibrated flow to 1.20 L/min halves the inferred concentration
to 5 cm⁻³ at the same rate. A zero
rate with a valid positive flow gives zero. A zero flow or missing
matching flow reading gives a null concentration with its reason. Quality
flags from both input readings travel with the result. This calculation
assumes the declared volume basis and one count per detected particle.
Corrections for coincidence and detection efficiency require instrument
evidence. For instruments with different sampling schedules, see
[Compare instruments on a common time grid](measurement-collections.md).

## Apply a causal filter across input batches

The built-in voltage mean orders readings by UTC time within the signal.
`window` is a number of samples. A full window of valid samples is required.
For readings of 1, 2, 3 and 4 V and a three-sample window, the outputs are
null, null, 2 and 3 V. A file or Parquet batch boundary within that sequence
preserves the history. A gap *greater than* `max_gap_seconds` begins a new
sequence and the first two outputs there are null again.

```python
smooth = meter.process("particula.signal.voltage_mean", version="1").configure(
    window=3, max_gap_seconds=2.0
)
mean_voltage = smooth.execute(signal=voltage)
```

Apply the filter to each deployment separately. For a custom filter choose
`mode="ordered"` and declare the window and maximum-gap parameter names in
`history_parameters`. Implement the same window, minimum sample count and
reset rule in the expression. Choose `"elementwise"` for a calibration that uses
only the current sample or `"whole_record"` for a calculation that needs the
complete input. For example, subtracting the mean voltage makes every
result depend on every sample used to compute the mean. Each output then
carries the quality flags, repair status and calibration references from
the complete aligned record.

## Prescribe, save and replay the calculation

A recipe connects operations by input alias. Each step names its operation
and version, binds its roles and names its output. A later step can consume
that output along with another measured input. Here is the YAML for the
single-certificate calculation above. Save it as `inlet-number.yaml`.

```yaml
recipe_version: "1"
steps:
  - operation: lab.flow.linear
    version: "1"
    inputs: {signal: flow_voltage}
    parameters: {calibration: flow-cal-17}
    output: calibrated_flow
  - operation: particula.particle_counter.number_concentration
    version: "1"
    inputs: {rate: inlet_rate, sample_flow: calibrated_flow}
    output: inlet_number
result: inlet_number
```

JSON accepts the same mapping. In Python, `.bind(...)` names the logical
inputs and output of each step and `|` composes the steps in order.
Use the installed recipe schema to validate a candidate mapping offline.
Execution also checks each operation version, input role, field quantity,
unit conversion and parameter schema. Each input must name a supplied
measurement or an earlier output. Reusing an output alias or referring
forward to a later step fails before a dataset is published.

```python
from jsonschema import Draft202012Validator

from particula.data import describe_recipe, open_measurement, run_recipe

inputs = {"flow_voltage": voltage, "inlet_rate": rate}
pipeline = (
    calibrate.bind(signal="flow_voltage", output="calibrated_flow")
    | derive.bind(rate="inlet_rate", sample_flow="calibrated_flow",
                  output="inlet_number")
)
Draft202012Validator(describe_recipe()).validate(pipeline.describe())
from_python = pipeline.execute(inputs=inputs)
run_recipe("inlet-number.yaml", inputs=inputs, registry=registry,
           destination="inlet-number")
result = open_measurement("inlet-number")
assert from_python.frame.equals(result.frame)
print(result.manifest["lineage"]["operation"])
print(result.frame.select("scan_id", "parent_rate_scan_id",
                          "parent_sample_flow_scan_id", "calibration_refs"))
```

The manifest records the field definition, process version, parameters,
parent product identities, alignment and calibration records. Use a parent
scan ID with its parent identity when tracing an observation back to its
source. The last step writes a staged dataset and publishes it with its
recipe after the writes succeed. A failed computation leaves the product
unpublished. Reopening a saved product reads its materialized derived values and
lineage without loading the lab code. To recompute it, supply the original
input products and register the matching implementation and version.

For a custom decoder that streams normalized batches, implement the
`BatchReader` contract. Its `batches()` yields scan and diagnostic frames,
`segment` identifies the active source layout, and `manifest` is complete at
end of file. Register its constructor with
`registry.register_batch_reader("lab.binary", LabReader)`. An instrument
can select `reader="lab.binary"`; for a disk import pass
`reader_factory=registry.batch_reader("lab.binary")` to `import_to_parquet`.
Both paths create a fresh decoder for each source.

An array input contract can specify ordered `axes` and a diameter `basis`.
When an operation combines array fields, their selected axes must have the
same names, order, units and coordinate values. Spectra must also share a
diameter basis and bin geometry.
A missing channel at any depth adds `missing:<role>` to the scan, including
when the operation can compute a partial result from the remaining channels.

The causal voltage mean records the individual source IDs in
`parent_signal_scan_ids` for each trailing window. Quality flags, repair
status and calibration references cover every contributing sample. A window
that spans a calibration change lists both certificates. After a clock gap,
the history starts again with the current observation.
