# Particula Design

Particula purposefully sticks to **two complementary coding paradigms** so users can pick the style that fits their workflow.

## Choosing a paradigm

| Preference | Use case | Recommendation |
|------------|----------|----------------|
| Procedural / notebooks | Quick calculations, teaching demos | Stick to `get_` functions. |
| OO / large experiments | Multiple interacting processes, validation, swapping kernels | Use Builders + Strategies. |

## 1. Procedural: Functional core – “verb‑phrased” helpers

Pattern: **Functions**  

These core functions are in most cases stateless and return a value based on inputs. They have no hidden side effects and are not dependent on any object. They are easy to test and can be replaced with JIT‑compiled versions (Numba/C++) in the future.

They are the building blocks of the simulation engine. They can be used in a procedural style, making them suitable for quick calculations or teaching demos.

**`get_`** – prefix that signals “this function returns a value”.

Example:  

```python
import particula as par

kernel = par.dynamics.get_turbulent_shear_kernel_st1956_via_system_state(
    particle_radius=particle_radius,
    turbulent_dissipation=eddy_dissipation,
    temperature=temperature,
    fluid_density=fluid_density,
)
```

## 2. Object‑oriented: – “noun‑phrased” abstractions

Patterns employed

- **Strategy** – e.g. `TurbulentShearCoagulationStrategy` selects which kernel to call, built of `get_` functions. These are the main building blocks of the simulation engine.
- **Builder** – e.g. `CombineCoagulationStrategyBuilder` validates input and assembles strategies.
- **Factory** – e.g. `ActivityFactory` creates an `ActivityStrategy` instance via an `ActivityBuilder`. These allow for more complex meta programming of simulation objects.
- **Decorator** – e.g. `@validate_inputs` checks for valid inputs before running the function. These are used to enforce domain specific invariants (positive radius, non‑negative concentration, finite Coulomb potential, …).
- **Mixin** – thin, single‑responsibility classes adding orthogonal capabilities (density, charge, surface …).  

For a broader overview of design patterns, see [Object‑Oriented Patterns](Object_Oriented_Patterns.md).

Example (combining two kernels):

```python
import particula as par

# Step 1: Build your individual strategies
brownian_strategy = (  # using a builder for the Brownian kernel
    par.dynamics.BrownianCoagulationBuilder()
    .set_distribution_type("discrete")
    .build()
)

turbulent_strategy = (  # directly creating the strategy
    par.dynamics.TurbulentShearCoagulationStrategy(
        distribution_type="discrete",
        turbulent_dissipation=0.01,    # example value [m^2/s^3]
        fluid_density=1.225            # air at sea level [kg/m^3]
    )
)

# Step 2: Combine strategies with the builder
builder = par.dynamics.CombineCoagulationStrategyBuilder()
builder.set_strategies([brownian_strategy, turbulent_strategy])
combined_strategy = builder.build()

# Step 3: Use the combined strategy in a coagulation process
coagulation_process = par.dynamics.Coagulation(
    coagulation_strategy=combined_strategy
)
```

Benefits

- Plug‑and‑play process swapping → perfect for sensitivity studies.  
- Builder validation enforces **Agreeing** (the *A* in WARMED).


## Naming conventions

- **Functions** → `get_<quantity>[_via_system_state]`  
- **Classes** → `<Descriptor><PatternName>` (`TurbulentShearCoagulationStrategy`, `PresetParticleRadiusBuilder`)  

These rules make grep‑based discovery trivial and help LLMs auto‑suggest the correct object.

## Future evolution

Performance work (Numba/C++) will follow the “replace the function, keep the interface” rule—strategies will automatically inherit the speed‑ups without further changes.

---
