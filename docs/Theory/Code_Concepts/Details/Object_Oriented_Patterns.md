
# Object‑Oriented Patterns

Particula mixes a functional “core” with a small, carefully chosen set of object‑oriented design patterns.

The goal is to give **researchers who are new to programming** a mental map for how the library is organized and why certain classes look the way they do.

---

## 1. Why design patterns?

Design patterns are reusable solutions to common software problems.  
They offer three big advantages:

1. A shared vocabulary – “Strategy”, “Builder”, “Factory” tell collaborators (and LLMs!) instantly what to expect.
2. Separation of concerns – each class has one clear job, making the code easier to test and swap.
3. Future proofing – performance upgrades or new physical models can be slotted in without touching user scripts.

If these terms are new to you, **don’t panic**. The next sections introduce each pattern and show how Particula uses it.

---

## 2. Key patterns used in Particula

### 2.1 Strategy – selecting “how”

Problem solved: *“I need to choose between different algorithms at run‑time.”*

General idea  
```text
Context ──> StrategyA
        │
        └──> StrategyB
```

Particula examples  
• `TurbulentShearCoagulationStrategy` – calls the correct turbulence kernel.  
• `VaporPressureStrategy` (and its concrete subclasses) – picks the physical equation for a gas.

Take‑away: **A “Strategy” object is just a plug‑in.** Swap it out, the rest of the simulation keeps running.

---

### 2.2 Builder – piecing together valid objects

Problem solved: *“Creating the object requires many parameters and consistency checks.”*

General idea  
```text
builder = FancyBuilder()
builder.set_x(…)
builder.set_y(…)
obj = builder.build()
```

Particula examples  
• `BrownianCoagulationBuilder` – validates the configured distribution type; execution validates state and environment.
• `CombineCoagulationStrategyBuilder` – glues several strategies into one composite strategy.

Tip for new users: the builder pattern reads almost like an English sentence; you can’t “forget” a required parameter because `build` would refuse to run.

---

### 2.3 Factory – hiding construction complexity

Problem solved: *“I want one function that returns a ready‑to‑use object, but internally different builders/strategies decide what’s best.”*

Executable Particula example (v0.3.0 development checkout):
```python
import particula as par
strategy = par.particles.ActivityFactory().get_strategy("mass_ideal")
```


Particula example  
• `ActivityFactory` selects a supported activity strategy from an explicit name;
it does not infer a physical model from a species name.

---

### 2.4 Decorator – adding orthogonal behavior

Problem solved: *“I need to add validation or logging without touching the original function.”*

Particula example  
```python
from particula.util.validate_inputs import validate_inputs

@validate_inputs({"radius": "positive"})
def sphere_volume(radius):
    return 4.0 / 3.0 * 3.141592653589793 * radius**3
```
`@validate_inputs` throws a helpful error before the computation starts.

---

### 2.5 Mixin – small capability boosters

Problem solved: *“Several unrelated classes need the same micro‑feature.”*

Example (conceptual)  
```text
class ChargeMixin:
    def get_charge_density(self): …

class Particle(ChargeMixin, BaseParticle): …
```

Builder mixins supply reusable setters, such as `BuilderMolarMassMixin`.
The diagram above is conceptual pseudocode, not an importable particle API.

---

### 2.6 Abstract Base Class (ABC) – enforcing an interface

Python code can be “duck‑typed”, but aerosols need guarantees!  
`BuilderABC`, `DistributionStrategy`, and `VaporPressureStrategy` define required methods (`build`, `pure_vapor_pressure`, …).  
Concrete subclasses must implement them or Python raises a clear error.

---

### 2.7 Template Method – fixed skeleton, overridable steps

`RunnableSequence` keeps the loop logic (“for each process: run, pass aerosol to next”), while each individual `Runnable` supplies the physics in its `execute` method.

---

### 2.8 Composition > Inheritance

“Favor composition over inheritance” is the golden rule that keeps Particula’s
class tree shallow.  
Instead of a deep hierarchy like

```text
BaseParticle
└── BrownianParticle
    └── ChargedBrownianParticle
        └── …
```

Particula builds objects out of **smaller collaborating parts**:

```python
from docs.Examples.native_state_access import build_aerosol, build_condensation

aerosol = build_aerosol()  # native particle, full gas and environment containers
process = build_condensation(aerosol.gas)  # explicit physics and species mapping
assert process.condensation_strategy is not None
```

• Native `Aerosol` holds `ParticleData`, `GasData`, and `EnvironmentData` by identity.
• A process owns its physical strategies and mapping; they are not injected into
  a particle carrier. Run this example from the v0.3.0 development checkout.

Benefits  
– no “diamond” problems,  
– compatible container replacement is validated before publication; layout
  changes require new process configuration,
– unit tests target one responsibility at a time.

---

### 2.9 Behavior, abstraction & naming rules

Particula leans on the four classic OO pillars:  

| Pillar          | In practice                                                                      |
|-----------------|----------------------------------------------------------------------------------|
| Encapsulation   | Native data stores arrays; `ParticleData.radii` derives radius from mass and density. Processes own physical behavior. |
| Abstraction     | Public API says *what* (“get mass”), strategies hide *how* (Brownian vs Gopalakrishnan). |
| Inheritance     | Kept minimal—mainly `ABC` bases and tiny mixins. Used for defining interfaces for strategies   |
| Polymorphism    | Choose a compatible process strategy with the same interface and admitted population kind.       |

#### Naming quick‑reference  

* Functions that *return a value* → `get_<quantity>()`  
* Classes that *encapsulate a pattern* → `<Descriptor><PatternName>`  
  – `BrownianCoagulationBuilder`, `BuckVaporPressureStrategy`, `RunnableSequence`

Sticking to these names makes **grep**, IDE auto‑complete and LLM help far
more effective for beginners.

---

## 3. How the patterns cooperate

1. **Builder** validates user input and produces a **Strategy**.  
   1. **Mixin** adds common `set_` to builders.  
2. **Factory** decides which Builder/Strategy combo to use.  
3. **Decorator** checks invariants every time the Strategy’s function is called.  
4. **Template Method (RunnableSequence)** orchestrates the time‑stepping.

This layering means you can:

- swap a Strategy for a faster JIT‑compiled one,  
- add a new vapor‑pressure correlation,  
- or prototype an entirely new process,

without touching more than a single, well‑contained file.

---

## 4. Cheat‑sheet for new users

| Pattern   | You will see it as…                         | What to remember |
|-----------|---------------------------------------------|------------------|
| Strategy  | `*Strategy` classes                         | A plug‑in algorithm |
| Builder   | `*Builder` classes                          | Step‑wise, validated construction |
| Factory   | `*Factory.get_strategy()`                         | One‑line object creation |
| Decorator | `@validate_inputs`              | Adds checks around a function |
| Mixin     | `BuilderMolarMassMixin`               | Supplies a single feature |
| ABC       | Classes inheriting from `*ABC`               | Enforces required methods/interfaces |
| Template  | `RunnableSequence` + `RunnableABC`          | Fixed loop, custom steps |


---

## Example

### Builder Pattern with Inline Comments

The builder below retains the tutorial's three radii, density and bin
concentrations. A builder is an object with setter methods; `build()` returns
a validated native data object. These methods configure values—they do not
create new methods or inject physics into the result.

```python
import numpy as np
import particula as par

radius_nm = np.array([100, 200, 300], dtype=np.float64)
concentration_cm3 = np.array([1e2, 1e3, 1e4], dtype=np.float64)
radius_m = radius_nm * 1e-9
density_kg_m3 = 2.5 * 1000.0
masses = (4.0 / 3.0 * np.pi * radius_m**3 * density_kg_m3)[:, None]

# Class construction creates a builder; chained methods return that builder.
particles = (
    par.particles.ParticleDataBuilder()
    .set_distribution_type("discrete")
    .set_masses(masses, "kg")
    .set_concentration(concentration_cm3, "1/cm^3")
    .set_density(np.array([2.5]), "g/cm^3")
    .set_charge(np.zeros(3))
    .set_volume(1.0, "m^3")
    .build()
)

# Physical strategies belong to a process configuration, not ParticleData.
surface = (
    par.particles.SurfaceStrategyMassBuilder()
    .set_surface_tension(0.072, "N/m")
    .set_density(2.5, "g/cm^3")
    .build()
)

print("Per-particle species mass [kg]:", particles.masses)
print("Radius [m]:", particles.radii)
print("Species mass density [kg/m³]:", particles.species_mass_density)
```

- `ParticleDataBuilder` and `SurfaceStrategyMassBuilder` are classes.
- Calling either class constructs a builder instance; `set_*` methods configure
  its values, and `build()` returns the corresponding data or strategy object.
- `particles.masses` exposes writable stored data. `particles.radii` and
  `particles.species_mass_density` return freshly derived arrays.
- For process construction and validated replacement, see the
  [native migration guide](../../../Features/particle-data-migration/index.md).

---

## Further Information

### YouTube videos

- Firebase's video on [Design Patterns](https://www.youtube.com/watch?v=tv-_1er1mWI) – a great introduction to the most common patterns.
- CodeAesthetic: [Abstraction](https://youtu.be/rQlMtztiAoA?si=6TDwFUwg4eW1fBGf), [Naming](https://youtu.be/-J3wNP6u5YU?si=TJhsetZA-9u6rhjK), [The Flaws of Inheritance](https://youtu.be/hxGOiiR9ZKg?si=1ylStcpG8sfMze2z).
- ThePrimeagen's: [8 Design Patterns](https://www.youtube.com/watch?v=ZfG8BSTX0Lw), [Why Python](https://youtu.be/8D7FZoQ-z20?si=FdjNVtkhJImK9zHJ).

### Reading

- The [WARMED principle](WARMED_principle.md) – Particula's philosophy for writing readable, swap‑friendly scientific code.
- [Object‑Oriented Patterns](https://refactoring.guru/design-patterns) – an expanded overview of design patterns in general.
- [Design Patterns: Elements of Reusable Object‑Oriented Software](https://www.oreilly.com/library/view/design-patterns-elements/0201633612/) – the classic book by Gamma et al. (1994).
