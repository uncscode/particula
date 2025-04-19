
# Object‑Oriented Patterns in Particula

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

“Favour composition over inheritance” is the golden rule that keeps Particula’s
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

### 2.9 Behaviour, abstraction & naming rules

Particula leans on the four classic OO pillars:  

| Pillar          | In practice                                                                      |
|-----------------|----------------------------------------------------------------------------------|
| Encapsulation   | State + behaviour live together (`ParticleRepresentation.get_radius()` uses its own density). |
| Abstraction     | Public API says *what* (“get mass”), strategies hide *how* (Brownian vs Gopalakrishnan). |
| Inheritance     | Kept minimal—mainly `ABC` bases and tiny mixins.                                 |
| Polymorphism    | Swap one `DistributionStrategy` for another without touching calling code.       |

Naming quick‑ref (matches the repo‑wide conventions):  

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

• swap a Strategy for a faster JIT‑compiled one,  
• add a new vapor‑pressure correlation,  
• or prototype an entirely new process,

without touching more than a single, well‑contained file.

---

## 4. Cheat‑sheet for new users

| Pattern   | You will see it as…                         | What to remember |
|-----------|---------------------------------------------|------------------|
| Strategy  | `*Strategy` classes                         | A plug‑in algorithm |
| Builder   | `*Builder` classes                          | Step‑wise, validated construction |
| Factory   | `*Factory.create()`                         | One‑line object creation |
| Decorator | `@validate_inputs`, `@time_it`              | Adds checks/logging around a function |
| Mixin     | `ChargeMixin`, `DensityMixin`               | Supplies a single feature |
| ABC       | Classes inheriting from `ABC`               | Enforces required methods/interfaces |
| Template  | `RunnableSequence` + `RunnableABC`          | Fixed loop, custom step |

---

## 5. Further reading/watching

* The [WARMED principle](Warmed_principle.md) – Particula’s philosophy for writing readable, swap‑friendly scientific code.
* Firebase's video on [Design Patterns](https://www.youtube.com/watch?v=tv-_1er1mWI) – a great introduction to the most common patterns.
* ThePrimeagen's [Design Patterns](https://www.youtube.com/watch?v=ZfG8BSTX0Lw) – 8 Design Patterns | Prime Reacts.
* [Object‑Oriented Patterns](https://refactoring.guru/design-patterns) – an expanded overview of design patterns in general.
* [Design Patterns: Elements of Reusable Object‑Oriented Software](https://www.oreilly.com/library/view/design-patterns-elements/0201633612/) – the classic book by Gamma et al. (1994).
