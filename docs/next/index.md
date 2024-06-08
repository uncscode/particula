# Next Outline

Next is the next iteration of the particula simulation model. It is a complete rewrite of the model, with a focus on improving the modularity and extensibility of the model. The goal is to make it easier to add new features and to make the model more flexible and easier to use.

## Tutorials

### Gas Phase

- [Vapor Pressure Tutorial](Tutorials/Vapor_Pressure.ipynb)
- [Gas Species Tutorial](Tutorials/Gas_Species.ipynb)
- [Atmosphere Tutorial](Tutorials/AtmosphereTutorial.ipynb)

### Particle Phase

- [Aerosol Surface Tutorial](Tutorials/Particle_Surface_Tutorial.ipynb)
- [Aerosol Distribution Tutorial](Tutorials/Aerosol_Distributions.ipynb)

## Guides for Developers

![Four Quadrant representation of Tutorials, How to guides, References, and Discussions Areas](DocsImageDevGuide.png)


## Diagrams


### Activity Component

```mermaid
graph TD
    AS[Activity Strategies] --> AFn[Functions]
    AB[Activity Builders] --> AS
    AF[Activity Factories] --> AB
    AFn --> link3["[Activity Strategies](./activity_strategies.md)"]
    AB --> link1["[Activity Builders](./activity_builders.md)"]
    AF --> link2["[Activity Factories](./activity_factories.md)"]
```

*Description of Activity Component here...*

### Distribution Component

```mermaid
graph TD
    DS[Distribution Strategies] --> DFn[Functions]
    DB[Distribution Builders] --> DS
    DF[Distribution Factories] --> DB
    DFn --> link6["[Distribution Strategies](./distribution_strategies.md)"]
    DB --> link4["[Distribution Builders](./distribution_builders.md)"]
    DF --> link5["[Distribution Factories](./distribution_factories.md)"]
```

*Description of Distribution Component here...*

### Properties Component

```mermaid
graph TD
    P[Properties] --> PLink["[Properties](properties/index.md)"]
```

*Description of Properties Component here...*

### Representation Component

```mermaid
graph TD
    RS[Representation Strategies] --> RFn[Functions]
    RB[Representation Builders] --> RS
    R[Representation] --> RB
    RFn --> link13["[Representation Strategies](./representation.md)"]
    RB --> link9["[Representation Builders](./representation_builders.md)"]
    R --> link8["[Representation](./representation.md)"]
```

*Description of Representation Component here...*

### Surface Component

```mermaid
graph TD
    SS[Surface Strategies] --> SFn[Functions]
    SB[Surface Builders] --> SS
    SF[Surface Factories] --> SB
    SFn --> link12["[Surface Strategies](./surface_strategies.md)"]
    SB --> link10["[Surface Builders](./surface_builders.md)"]
    SF --> link11["[Surface Factories](./surface_factories.md)"]
```

*Description of Surface Component here...*
