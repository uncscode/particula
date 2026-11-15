# Build a meaningful initial state

Before choosing a process, decide what the population represents. A value can
be a per-particle mass, a number density, or a represented count; similar arrays
can therefore describe different physical systems.

| Question | Worked example | What to check |
| --- | --- | --- |
| How do I describe a measured size spectrum? | [Size distributions](particles/aerosol_distributions.py) | Coordinate, units, and integration rule |
| How do size and composition become model input? | [Particle construction](particles/particle_data.py) | Per-particle mass versus population abundance |
| How does stored vapor relate to saturation? | [Gas and environment](gas/index.md) | Species order and property-model conventions |
| How do the pieces evolve together? | [Native aerosol](native_aerosol.py) | Held state, box volume, and process configuration |
| How can I replace or inspect state safely? | [State access](native_state_access.py) | Live arrays, detached quantities, and species mapping |
| What survives a device transfer? | [Data containers](data-containers.md) | Representation and caller-owned species identity |

Start with [your first CPU simulation](../../guide/getting-started/first-simulation.md)
if you want a complete small calculation before exploring the individual parts.
After construction, use [process examples](../processes/index.md) to connect an
initial condition with a physical tendency or time history. A constructor accepting
an array does not establish that the input is a realistic aerosol.
