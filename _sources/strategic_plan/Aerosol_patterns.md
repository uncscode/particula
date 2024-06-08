# Program Outline Revision Draft


### Aerosol Data objects

**Gas Module**

This module represents the gaseous environment in which aerosols are suspended. It should encapsulate properties such as vapor pressure, humidity (or water activity), temperature, total pressure, and the masses of each vapor present.

- **Design Suggestion**: Use the **Builder Pattern** to incrementally construct a complex `Gas` object . This pattern can handle various combinations of properties efficiently, allowing for clear and flexible gas object creation.

**Particle Module**

Represents individual particles within an aerosol, containing attributes like the mass of species per particle, particle radius, density, and count per particle size.

- **Design Suggestion**: A **Composite Pattern** might be useful here to treat individual particles and compositions of particles uniformly. This could simplify operations that apply to both single particles and collections of particles, especially when calculating properties that depend on the entire particle ensemble.

**Aerosol Module**

Combines the `Gas` and `Particle` modules to represent an aerosol system. It should include a mechanism for evolving the system over time (time step).

- **Design Suggestion**: The **Facade Pattern** could offer a simplified interface to complex subsystems (Gas and Particle), making the Aerosol module easier to use without sacrificing the flexibility of direct interaction with the subsystems.

## Example

To structure the three data classes (Gas, Particle, Aerosol) in code using the design pattern suggestions provided, you would typically follow an object-oriented programming approach. Here's a high-level structure with examples in Python to illustrate how these patterns can be implemented:

### Gas Module with Builder Pattern

The Builder Pattern is great for constructing complex objects step by step. It allows you to produce different types and representations of an object using the same construction process.

```python
class GasBuilder:
    def __init__(self):
        self.reset()

    def reset(self):
        self._product = Gas()

    def set_vapor_pressure(self, pressure):
        self._product.vapor_pressure = pressure
        return self

    def set_humidity(self, humidity):
        self._product.humidity = humidity
        return self

    def set_temperature(self, temperature):
        self._product.temperature = temperature
        return self

    def set_total_pressure(self, total_pressure):
        self._product.total_pressure = total_pressure
        return self

    def set_masses(self, masses):
        self._product.masses = masses
        return self

    def build(self):
        product = self._product
        self.reset()
        return product

class Gas:
    def __init__(self):
        self.vapor_pressure = None
        self.humidity = None
        self.temperature = None
        self.total_pressure = None
        self.masses = None
```

#### Revision of Gas with multiple species

Given the requirement that the `Gas` class can contain multiple species, each with its own properties (such as mass), you're correct in considering a structure that allows for this complexity. While the Composite Pattern is traditionally used for part-whole hierarchies, for managing multiple species within a `Gas` object, a more fitting approach could be using a container within the `Gas` class to manage these species. However, this container would manage different species and their attributes rather than creating a strict Composite structure. 

Here's how you can structure the `Gas` class to accommodate multiple species, focusing on a flexible and extensible design:

```python
class GasSpecies:
    def __init__(self, name, mass, vapor_pressure=None):
        self.name = name
        self.mass = mass  # Mass of the species in the gas mixture
        self.vapor_pressure = vapor_pressure  # Optional, specific to species

class Gas:
    def __init__(self, temperature=298.15, total_pressure=101325):
        self.temperature = temperature
        self.total_pressure = total_pressure
        self.species = {}  # Dictionary to hold GasSpecies objects

    def add_species(self, name, mass, vapor_pressure=None):
        species = GasSpecies(name, mass, vapor_pressure)
        self.species[name] = species

    def remove_species(self, name):
        if name in self.species:
            del self.species[name]

# Example Usage:
gas_mixture = Gas(temperature=300, total_pressure=101325)
gas_mixture.add_species('O2', mass=32, vapor_pressure=21300)
gas_mixture.add_species('N2', mass=28)

# Accessing a species:
print(gas_mixture.species['O2'].mass)  # 32
```

In this structure, `GasSpecies` represents an individual gas species, encapsulating properties relevant to a single species, such as its name, mass, and optional properties like vapor pressure. The `Gas` class then contains a dictionary of these `GasSpecies` objects, allowing for the addition, modification, and removal of species within the gas mixture.

This design offers flexibility and scalability for handling a mixture of gases with varying properties. It also keeps the interface for interacting with individual species straightforward, enabling easy adjustments to the composition of the gas mixture as needed. 

This approach does not strictly follow the Composite Pattern, as it doesn't aim to treat individual and composite objects uniformly. Instead, it provides a clear way to manage a collection of distinct objects (`GasSpecies`) within another object (`Gas`), which is more aligned with your requirements.

### Particle Module with Composite Pattern

The Composite Pattern allows you to compose objects into tree structures to represent part-whole hierarchies. It lets clients treat individual objects and compositions uniformly.

```python
class Particle:
    def __init__(self, mass, radius, density, count):
        self.mass = mass
        self.radius = radius
        self.density = density
        self.count = count

class ParticleComposite:
    def __init__(self):
        self._children = []

    def add(self, particle):
        self._children.append(particle)

    def remove(self, particle):
        self._children.remove(particle)

    def get_mass(self):
        return sum(child.mass for child in self._children)
```

### Aerosol Module with Facade Pattern

The Facade Pattern provides a simplified interface to a complex subsystem. Here, the Aerosol class acts as a facade to the Gas and Particle modules, making it easier to interact with these subsystems.

```python
class Aerosol:
    def __init__(self, gas_builder, particles):
        self.gas = gas_builder.build()
        self.particles = particles


# Example usage
gas_builder = GasBuilder().set_temperature(300).set_humidity(0.5).set_total_pressure(101325)
particles = ParticleComposite()
particles.add(Particle(mass=1.2, radius=0.5, density=1.0, count=100))

aerosol = Aerosol(gas_builder, particles)
```

In this setup:
- **GasBuilder** allows for the flexible construction of a `Gas` object with various properties.
- **ParticleComposite** manages a collection of `Particle` objects, enabling operations on groups of particles as if they were a single object.
- **Aerosol** acts as a facade for the complex interactions between `Gas` and `Particle` objects, simplifying the usage for clients.



## Processing Objects

Defines common interfaces for operations (evaluate, rate, update) that process the aerosol system, taking an aerosol as input and returning an updated aerosol.

- **Design Suggestion**: The **Strategy Pattern** is ideal here, allowing the algorithm to be selected at runtime. It supports the interchangeable use of different processing methods (e.g., for coagulation or chemical reactions) while maintaining a consistent interface.


## Processes (Coagulation, Vapor Equilibrium, Chemical Reaction, Nucleation)

- **Abstract Factory (Process Factory)**: This serves as the abstract base for creating families of related or dependent process objects without specifying their concrete classes. It defines the interface for creating different types of process factories (CoagulationFactory, VaporEquilibriumFactory, ChemicalReactionFactory, etc.).

### Implementation

**Process Factory Interface**

This is the abstract class that declares a set of methods for creating different process objects. Each process (Coagulation, Vapor Equilibrium, etc.) will have its own concrete factory that implements this interface.

**Concrete Factories**

- **CoagulationFactory**: Creates and returns instances of Coagulation processes, with methods tailored to different coagulation strategies (number-based, moving section-based, super droplet-based).
  
- **VaporEquilibriumFactory**: Specializes in creating Vapor Equilibrium processes, providing different methods for handling vapor equilibrium (number-based condensation, moving section-based condensation, etc.).
  
- **ChemicalReactionFactory**: Generates objects for Chemical Reaction processes, offering various methods for different chemical reactions (gas phase, surface, bulk, particle phase, heterogeneous).

- **NucleationFactory**: Dedicated to creating Nucleation process objects, with options for homogeneous, heterogeneous, ion-induced, and binary nucleation.

- **WallLossFactory**: Focuses on creating Wall Loss processes, suitable for calculating particle loss to walls or boundaries.

### Benefits of This Approach

- **Flexibility in Process Selection**: This structure allows for the dynamic instantiation of processes based on simulation needs or environmental conditions, offering great flexibility and adaptability.

- **Isolation of Construction Logic**: By encapsulating the creation logic within each factory, the complexity of object creation is isolated from the rest of the application.

- **Enhanced Modularity**: The Abstract Factory pattern promotes a more modular system design, where changes to the process creation mechanism or the introduction of new processes require minimal changes to the codebase.

- **Consistency**: Using a standardized interface for creating processes ensures consistency across the application, making it easier to manage and extend.

### Implementation Consideration

While the Abstract Factory and Factory Pattern combination provides a robust structure for your aerosol modeling application, it's important to ensure that the design remains flexible and doesn't introduce unnecessary complexity. Keep the interface of your abstract factory and concrete factories clear and intuitive, and only define separate factories for processes that genuinely require different instantiation logic or have distinctly different configurations.




from langchain_core.runnables import RunnableLambda

def mass_condensation(input = Aerosol, other_settings) -> Aerosol:
    # Perform mass condensation calculations
    return Aerosol

def mass_coagulation(input = Aerosol, other_setting2) -> Aerosol:
    # Perform mass coagulation calculations
    return Aerosol

runnable_1 = mass_condensation(other_settings)
runnable_2 = mass_coagulation(other_settings2)
sequence = runnable_1 | runnable_2
# Or equivalently:
# sequence = RunnableSequence(first=runnable_1, last=runnable_2)

sequence.invoke(input=Aerosol)
