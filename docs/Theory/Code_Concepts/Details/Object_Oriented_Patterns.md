
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
```python
builder = FancyBuilder()
builder.set_x(…)
builder.set_y(…)
obj = builder.build()
```

Particula examples  
• `BrownianCoagulationBuilder` – validates distribution type, temperature, etc.  
• `CombineCoagulationStrategyBuilder` – glues several strategies into one composite strategy.

Tip for new users: the builder pattern reads almost like an English sentence; you can’t “forget” a required parameter because `build` would refuse to run.

---

### 2.3 Factory – hiding construction complexity

Problem solved: *“I want one function that returns a ready‑to‑use object, but internally different builders/strategies decide what’s best.”*

General idea (but not implemented in Particula yet)
```python
strategy = ActivityFactory.create(name="water", temperature=298)
```

Particula example  
• `ActivityFactory` – selects the proper `ActivityStrategy` (Raoult | Pitzer | AI model …) based on user input.

---

### 2.4 Decorator – adding orthogonal behavior

Problem solved: *“I need to add validation or logging without touching the original function.”*

Particula example  
```python
@validate_inputs({"radius": "positive"})
def get_brownian_kernel(…):
    …
```
`@validate_inputs` throws a helpful error before the computation starts.

---

### 2.5 Mixin – small capability boosters

Problem solved: *“Several unrelated classes need the same micro‑feature.”*

Example (conceptual)  
```python
class ChargeMixin:
    def get_charge_density(self): …

class Particle(ChargeMixin, BaseParticle): …
```

Particula’s `DensityMixin`, `ChargeMixin`, etc. inject a single extra property without polluting the main class hierarchy.

---

### 2.6 Abstract Base Class (ABC) – enforcing an interface

Python code can be “duck‑typed”, but aerosols need guarantees!  
`BuilderABC`, `DistributionStrategy`, and `VaporPressureStrategy` define required methods (`build`, `pure_vapor_pressure`, …).  
Concrete subclasses must implement them or Python raises a clear error.

---

### 2.7 Template Method – fixed skeleton, overridable steps

`RunnableSequence` keeps the loop logic (“for each process: run, pass aerosol to next”), while each individual `Runnable` supplies the physics in its `__execute__` method.

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
representation = ParticleRepresentation(
    strategy=MassBasedMovingBin(...),     # ⇦ behaviour plug‑in
    activity=RaoultActivityStrategy(...), # ⇦ second behaviour plug‑in
)
aerosol = Aerosol(atmosphere=atm, particle_representation=representation)
```

• `Aerosol` *has an* `Atmosphere`; it does **not** *inherit from* it.  
• `ParticleRepresentation` *has a* `DistributionStrategy`.  

Benefits  
– no “diamond” problems,  
– you can replace any sub‑component at run‑time,  
– unit tests target one responsibility at a time.

---

### 2.9 Behavior, abstraction & naming rules

Particula leans on the four classic OO pillars:  

| Pillar          | In practice                                                                      |
|-----------------|----------------------------------------------------------------------------------|
| Encapsulation   | State + behavior live together (`ParticleRepresentation.get_radius()` uses its own density). |
| Abstraction     | Public API says *what* (“get mass”), strategies hide *how* (Brownian vs Gopalakrishnan). |
| Inheritance     | Kept minimal—mainly `ABC` bases and tiny mixins. Used for defining interfaces for strategies   |
| Polymorphism    | Swap one `DistributionStrategy` for another without touching calling code.       |

#### Naming quick‑reference  

* Functions that *return a value* → `get_<quantity>()`  
* Classes that *encapsulate a pattern* → `<Descriptor><PatternName>`  
  – `BrownianCoagulationBuilder`, `WaterBuckStrategy`, `RunnableSequence`  

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
| Mixin     | `ChargeMixin`, `DensityMixin`               | Supplies a single feature |
| ABC       | Classes inheriting from `*ABC`               | Enforces required methods/interfaces |
| Template  | `RunnableSequence` + `RunnableABC`          | Fixed loop, custom steps |


---

## Example

### Builder Pattern with Inline Comments

In this example, we point out which parts are classes (blueprints), which parts are objects (instances created from these classes), and which are functions (methods) that you call to change the internal state or retrieve values from the final object. Notice in particular how properties and methods are “injected” into the builder and, ultimately, become part of the final particle representation object (`particle_rep_mass`). This example is based on the Particle Representation Tutorial.

```python
import numpy as np
import particula as par

# Define basic particle parameters.
# Here, 'radius' and 'concentration' are objects from NumPy (created by the np.array() function)
# and 'density' is just a numerical value.
radius = np.array([100, 200, 300], dtype=np.float64)  # Object: NumPy ndarray holding radii values in nanometers.
density = 2.5  # A simple numeric value representing density.
concentration = np.array([1e2, 1e3, 1e4], dtype=np.float64)  # NumPy array holding concentration values.

# ----------------------------------------------------------------------------
# The following section uses builder classes to set up the particle properties.
# ----------------------------------------------------------------------------

# 'SurfaceStrategyMassBuilder' is a class (a blueprint) that, when instantiated,
# returns an object (builder instance) which provides methods to set up a surface strategy.
# Here, we call its methods 'set_surface_tension' and 'set_density'.
# Each of these methods is a function defined on the builder class that sets a property
# into the builder's internal state. These properties are later used to construct the final surface strategy object.
surface_tension_strategy = (
    par.particles.SurfaceStrategyMassBuilder()  # Class constructor: returns a builder object for surface strategy.
    .set_surface_tension(0.072, "N/m")          # Method: injects the surface tension value; function attached to the builder object.
    .set_density(2.5, "g/cm^3")                 # Method: injects the density value; also a function on the builder.
    .build()                                    # Method: finalizes the builder and returns the surface strategy object.
)

# 'ParticleRadiusRepresentationBuilder' is another class used to create a particle representation object (how particles are represented in the simulation).
# This builder gives you a fluent interface where each set_* method is a function that updates the builder instance.
# They are not standalone functions but methods that belong to the builder object.
particle_rep_mass = (
    par.particles.ParticleRadiusRepresentationBuilder()            # Class call: creates a builder object for particle representation.
    .set_distribution_strategy(par.particles.RadiiBasedMovingBin())  # Method call: sets the distribution strategy. Note: 'RadiiBasedMovingBin()' is itself a class constructor; its returned object is injected as a property.
    .set_activity_strategy(par.particles.ActivityIdealMass())      # Method call: sets the activity strategy. 'ActivityIdealMass()' is created by its class constructor and injected here.
    .set_surface_strategy(surface_tension_strategy)                # Method call: injects the previously built surface strategy object.
    .set_concentration(concentration=concentration, concentration_units="1/cm^3")  # Method: sets the concentration property; the function here attaches the numerical array to the builder.
    .set_density(density=density, density_units="g/cm^3")    # Method: sets the density property.
    .set_radius(radius=radius, radius_units="nm")            # Method: sets the radius property.
    .set_charge(charge=0)                                    # Method: sets the charge property (for example, neutral particles).
    .build()   # Method: finalizes the builder and returns the final particle representation object.
               # After 'build()' is called, all were "injected" into a newly created ParticleRepresentation instance.
)

# ----------------------------------------------------------------------------
# At this point, 'particle_rep_mass' is an object (an instance) of the ParticleRepresentation class.
# The methods get_mass(), get_radius(), and get_mass_concentration() are functions defined on that object.
# These functions are "injected" as part of the object's class definition – they allow you to retrieve calculated properties.
# How the properties are calculated is different for each ParticleRepresentation class, but you see the same interface at the object level.
# ----------------------------------------------------------------------------

# Calling methods (functions attached to the object 'particle_rep_mass') to access computed properties:
print("Mass of particles:", particle_rep_mass.get_mass())  
# get_mass() is a method attached to the 'particle_rep_mass' object; it calculates and returns the mass.
print("Radius of particles:", particle_rep_mass.get_radius())  
# get_radius() is also an object method, returning the calculated radii.
print("Total mass of the particle distribution:", particle_rep_mass.get_mass_concentration())
# get_mass_concentration() is another method attached to the final object, computing the overall mass concentration.
```

In this commented code:

- Classes like `SurfaceStrategyMassBuilder` and `ParticleRadiusRepresentationBuilder` use PascalCase as is typical for class names. However, snake_case is used for methods (functions) that are called on the instances of these classes.
- When we call a class (like `par.particles.SurfaceStrategyMassBuilder()`), it creates an object (a builder instance) that we then use to set properties.
- Methods like `set_surface_tension`, `set_density`, `set_concentration`, etc., are functions defined on those builder objects – they modify the internal state (i.e., inject properties) so that when you call `build()`, their values are used to create the final product.
- The final built object, `particle_rep_mass`, is an instance of a `ParticleRepresentation`. It has methods (like `get_mass()`) that are injected via the class definition and allow you to retrieve data computed from the properties previously injected via the builder pattern.

This clear separation of classes (blueprints), objects (instances created via constructors or build methods), and functions (methods attached to the objects) is a key principle in object-oriented programming and is leveraged by Particula for flexible aerosol simulations.

---

## Further Information

### YouTube videos

- Firebase's video on [Design Patterns](https://www.youtube.com/watch?v=tv-_1er1mWI) – a great introduction to the most common patterns.
- CodeAesthetic: [Abstraction](https://youtu.be/rQlMtztiAoA?si=6TDwFUwg4eW1fBGf), [Naming](https://youtu.be/-J3wNP6u5YU?si=TJhsetZA-9u6rhjK), [The Flaws of Inheritance](https://youtu.be/hxGOiiR9ZKg?si=1ylStcpG8sfMze2z).
- ThePrimeagen's: [8 Design Patterns](https://www.youtube.com/watch?v=ZfG8BSTX0Lw), [Why Python](https://youtu.be/8D7FZoQ-z20?si=FdjNVtkhJImK9zHJ).

### Reading

- The [WARMED principle](Warmed_principle.md) – Particula’s philosophy for writing readable, swap‑friendly scientific code.
- [Object‑Oriented Patterns](https://refactoring.guru/design-patterns) – an expanded overview of design patterns in general.
- [Design Patterns: Elements of Reusable Object‑Oriented Software](https://www.oreilly.com/library/view/design-patterns-elements/0201633612/) – the classic book by Gamma et al. (1994).
