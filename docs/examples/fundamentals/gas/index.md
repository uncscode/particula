# Gas properties and environmental forcing

A gas inventory and its saturation reference are separate inputs. Work through
these examples to distinguish the amount of vapor present from its tendency to
exchange material with particles.

1. [Vapor pressure](vapor_pressure.py): convert a sourced correlation into the
   implementation's units, then distinguish pure pressure, actual partial
   pressure, and saturation ratio during heating at fixed concentration.
2. [Gas storage](gas_data.py): keep species and their property models in the
   same explicit order, and compare saturation across a mixture.
3. [Prescribed environment](environment_data.py): isolate the effect of cooling
   on water saturation without claiming to simulate an expanding parcel.

The first two examples use sourced coefficients within their stated temperature
intervals. The environment sweep evaluates only water and changes its saturation
reference below freezing; the organic lanes remain stored but unevaluated.

To predict uptake rather than diagnose saturation, continue to
[condensation](../../processes/condensation/condensation_1_bin.py). That adds
particle activity, curvature, transport, and a map between gas and material lanes.
