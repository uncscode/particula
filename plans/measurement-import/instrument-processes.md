# User-defined instruments and prescribed measurement processes

A lab should be able to add its own instrument, supply its calibration or signal-processing code, and combine its outputs with a built-in Particula instrument. Use the simulation API's pattern of a configured process, explicit execution, and ordered composition via `|`. Implement a measurement-specific executor: the existing `RunnableABC.execute` advances an `Aerosol` over a time step and stages per-box writes. Observations already have timestamps and exposure. Executing a measurement process produces a new product with its own identity. [Simulation runnable and sequence](../../particula/runnable.py) · [Measurement collection and product types](../measurement-data-and-instruments.md#multiple-instruments-in-one-collection)

## Public responsibilities

| Object | Responsibility |
| --- | --- |
| `Instrument` | General instrument definition plus deployment. It binds a reader/profile, supported measurement products and named, versioned processing code. A lab can construct it directly with a profile and local Python functions. Bind serial number, inlet, and calibration records per deployment so two instances can use the same code with different coefficients. |
| `ParticleCounter`, `MobilitySizer` and other specific instruments | Supply built-in definitions and typed conveniences. For example, a particle counter exposes count-rate conversion and a mobility sizer exposes operations on its mobility-diameter spectrum. A profile-based instrument that supplies the same validated quantity, units and coordinates can fill the same process input role. |
| `Measurement` and typed product views | Hold the raw or derived values, field definitions, QC and provenance. An instrument can expose several named products. Persisting a measurement saves its manifest and values. Reloading it for further processing resolves named code through an explicitly supplied Python registry. |
| `MeasurementProcess` | Binds an implementation to a versioned operation identifier, named inputs, required quantities/units/axes, parameter schema, output type and execution mode. Instrument definitions expose the processes they own. Common operations such as clock alignment or a ratio may also be registered once and reused across instruments. |
| `MeasurementCollection` | Continues to map campaign names to measurement products. A process can take an instrument with one compatible bound product, an explicitly selected product, or named members of a collection. Ambiguous matches report the compatible choices. A caller chooses whether to publish the returned product into a collection. |

A supported text layout can use the generic reader with a YAML or JSON profile. A proprietary format can supply a Python reader that yields the same validated batches. A local operation registers its callable and input/output descriptions through the same public interface used by built-ins. Use a package-qualified name and version; reject conflicting registrations. Profiles and processing recipes are portable data. A recipe names registered operations, and loading a file never executes a module path or inline Python from its contents. Unknown operation IDs produce an actionable validation error. A user can still open previously materialized values when the original processing package is unavailable.

## Where strategies, builders and factories belong

Particula's `CoagulationFactory` maps names such as `brownian` and `charged` to builders. `CondensationFactory` selects among isothermal, staggered and latent-heat implementations. The shared `StrategyFactoryABC.get_strategy()` finds a builder, applies supplied parameters and returns the built strategy. Instruments have two distinct choices to make: how to decode a file, and how to calculate a product. Give each choice its own contract. [Coagulation factory](../../particula/dynamics/coagulation/coagulation_factories.py) · [Condensation factory](../../particula/dynamics/condensation/condensation_factories.py) · [Shared factory](../../particula/abc_factory.py)

| Pattern | Apply it here | First delivery |
| --- | --- | --- |
| **Strategy** | A reader implementation consumes a source and validated profile and yields bounded batches. A processing implementation accepts named, validated products and parameters. The instrument exposes the processes available for its products. | The particle counter and mobility sizer share the delimited-text reader. Their profiles supply different field mappings, scan layouts and repairs. A binary decoder can later implement the same batch contract. The lab's linear flow calibration can be a function; a stateful filter owns state for one execution. |
| **Factory** | Use the registry as the factory behind configuration-driven calls. Resolve a reader by its format and a prescribed process by its qualified operation ID and version. Keep direct Python construction for callers that already know the instrument or callable. | `ParticleCounter(profile=..., deployment=...)` and `read_measurement(..., profile=...)` share the same reader resolution and validation. The current profile schema specifies delimited text. Introduce a reader selector with format-specific schema branches when another format is supported. A new instrument family can use an existing reader immediately. An explicitly requested but unknown reader or process returns a diagnostic with registered choices. |
| **Builder** | Use one validated profile as the construction input for ordinary readers. Add a concrete builder when there is a real multi-step setup with cross-dependent inputs, such as an optical response assembled from detector geometry, wavelength configuration and calibration. | The particle counter and mobility sizer need only a profile and deployment. Python, YAML and JSON construction must use the same semantic checks. A future builder should normalize into that same configuration and be justified by its first caller. |

Keep the measurement registry small and explicitly extensible by the caller. The existing dynamics factory expects every entry to supply a builder with `set_parameters()` and `build()`. Reader profiles already have nested column mappings, explicit unit pairs and source-located diagnostics; adapting simple readers or functions to that interface would create a second configuration path. Share the scientific unit converters and the convention of explicit execution. Keep process selection, input-role validation and measurement lineage within the measurement API.

**Acceptance.** Both built-in profiles select the same reader implementation and produce their distinct typed products. A lab registers a reader or a calibration function using public APIs without editing a central switch. Duplicate IDs and unknown versions fail before reading or publishing data. Repeated factory calls start with independent parser/filter state. Direct Python and equivalent recipe execution yield matching quantities, units, QC and parent references.

## Worked design example

[Deliveries 5 and 6](../measurement-data-and-instruments.md#delivery-sequence-and-evidence) introduce unary calibration first, then bind its output to a second instrument. The example uses a lab-built flow meter and a CPC at the same sample inlet. The lab supplies a voltage-to-flow calibration in Python. Here the units are V on input, L/min per V for the slope, and L/min for the intercept. The CPC owns the process that converts count rate and calibrated sample flow to number concentration. These are **proposed interfaces**; the draft text-import schema will need voltage and volumetric-flow quantities before the lab profile can validate.

```python
def linear_flow(voltage, *, slope, intercept):
    return slope * voltage + intercept

flow_meter = Instrument.from_profile("lab-flow.yaml", deployment="inlet-flow")
flow_meter.register_process(
    "lab.flow.linear", version="1", implementation=linear_flow,
    contract=flow_calibration_contract,
)
flow_meter.read("flow.csv", as_product="voltage")

counter = ParticleCounter(profile="cpc-3010.yaml", deployment="inlet-counter")
counter.read("counts.csv", as_product="rate")

calibrate = flow_meter.process("lab.flow.linear", version="1").configure(
    calibration="flow-cal-17",
)
derive = counter.process("particula.particle_counter.number_concentration", version="1")
flow = calibrate.execute(signal=flow_meter.product("voltage"))
concentration = derive.execute(rate=counter.product("rate"), sample_flow=flow)
```

Here `flow_calibration_contract` is supplied by the lab. It declares a single scalar voltage field, the coefficient units above, a positive volumetric-flow output, the sample-volume basis, and elementwise batch execution. The deployment record resolves `flow-cal-17` to coefficient values over an effective interval. Particula handles field binding and unit validation before invoking the function. For coincident one-second observations at the same volume basis, 2 V with a slope of 0.30 L/min per V and zero intercept yields 0.60 L/min, or 10 cm³/s. At 100 counts/s the resulting concentration is 10 cm⁻³. A second deployment calibrated to 1.20 L/min at the same voltage yields 5 cm⁻³.

For reproducible processing, the same steps can be prescribed in YAML or equivalent JSON. The following uses logical aliases for the raw inputs bound by the caller.

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

For Python composition, the equivalent proposed expression is `calibrate.bind(signal="flow_voltage", output="calibrated_flow") | derive.bind(rate="inlet_rate", sample_flow="calibrated_flow", output="inlet_number")`. The last step supplies the returned product. Keep explicit role bindings so the second step can consume the CPC and the preceding flow output. Preflight checks resolve the ordered names, reject missing or forward references and output-name collisions, and verify that a step produces the quantity its consumer expects. Use a separate versioned recipe schema; the import profile continues to describe decoding. Operation descriptions drive `available_operations()`, configuration validation and the actual execution checks.

## Execution and qualification

- **Synchronization and QC.** Specify a common clock, alignment policy and tolerance or verified exposure overlap. The worked example uses exact timestamps. An unmatched observation yields a missing result with a reason under the declared policy. A recorded zero rate with valid flow yields zero concentration. A nonpositive, missing or invalid flow cannot yield a concentration. Propagate flags and the calibration reference.
- **Signal processing across batches.** Elementwise calibration needs no history. A causal moving average declares its window, per-deployment carry state, minimum valid samples and gap/reset policy. A three-sample filter over `[1, 2, 3, 4]` with complete windows yields `[null, null, 2, 3]` whether it receives one batch or four. A whole-record algorithm declares that requirement before execution.
- **Product identity and failure.** Keep parents readable and unchanged. Each materialized output records its product type, units, operation and implementation version, canonical parameters and calibration IDs, all parent dataset IDs, alignment/QC rules and contributing record or interval references. Persist recipes beside product manifests. Use the existing staged-Parquet publication design; a failed downstream step leaves the final product unpublished. A retry must reproduce values and lineage independently of batch boundaries.
- **Implementation checks for deliveries 5 and 6.** Register the flow meter and its processing function from a standalone user module using only public APIs. Run unary calibration, the two-input concentration operation and their prescribed sequence. Compare direct Python, YAML and JSON execution against the numerical controls above. Verify missing roles, incompatible units, mismatched volume bases, calibration expiry, duplicate operation registration and ambiguous instrument outputs give specific diagnostics. Reopen a derived product without the lab module, then re-register the matching version and replay it. Verify the same results after swapping in a second compatible source and for multiple input/output batch sizes.
