# Next Outline

Next is the next iteration of the particula simulation model. It is a complete rewrite of the model, with a focus on improving the modularity and extensibility of the model. The goal is to make it easier to add new features and to make the model more flexible and easier to use.

## Tutorials

- [Aerosol Distribution](Tutorials/Aerosol_Distributions.ipynb)
- [Vapor Pressure](Tutorials/Vapor_Pressure.ipynb)
- [Gas Species](Tutorials/Gas_Species.ipynb)

## Guides for Developers

![Four Quadrant representation of Tutorials, How to guides, References, and Discussions Areas](DocsImageDevGuide.png)


## Diagrams

To make your documentation clearer and more digestible, you can split the Mermaid diagram into multiple parts, each focusing on a specific component of your aerosol code structure. Below, each diagram is followed by a placeholder for your text descriptions:

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

This approach not only visually segments the different components of your code but also provides spaces between diagrams for detailed descriptions, making it easier for readers to understand each part's functionality and role within the larger system.