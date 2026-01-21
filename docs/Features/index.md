# Features

Welcome to the Particula Features documentation! This section provides comprehensive guides for the major capabilities and systems within Particula. Each feature guide explains the design patterns, available APIs, and best practices for working with that component.

## Feature Guides

<div class="grid cards" markdown>

-   __[Activity System](activity_system.md)__

    ---

    Strategy-based activity calculations for aerosol thermodynamics,
    including ideal, kappa-Kohler, and BAT (non-ideal) models.

    [:octicons-arrow-right-24: Learn more](activity_system.md)

-   __[Coagulation Strategy System](coagulation_strategy_system.md)__

    ---

    Strategy-based particle coagulation for chamber simulations with
    Brownian, charged, and turbulent coagulation models.

    [:octicons-arrow-right-24: Learn more](coagulation_strategy_system.md)

-   __[Condensation Strategy System](condensation_strategy_system.md)__

    ---

    Strategy-based condensation for particle growth simulations with
    isothermal and coupled dynamics models.

    [:octicons-arrow-right-24: Learn more](condensation_strategy_system.md)

-   __[Wall Loss Strategy System](wall_loss_strategy_system.md)__

    ---

    Strategy-based wall loss for chamber simulations with spherical,
    rectangular, and charged particle deposition models.

    [:octicons-arrow-right-24: Learn more](wall_loss_strategy_system.md)

</div>

## Design Patterns

All feature systems in Particula follow the **Strategy-Builder-Factory** pattern:

- **Strategy**: Abstract base classes defining the interface for each physics domain
- **Builder**: Unit-aware configuration with validation for constructing strategies
- **Factory**: Dynamic strategy selection by name for runtime flexibility

This pattern provides:

- Consistent APIs across different physics modules
- Easy extensibility for new models
- Validated configuration with automatic unit conversion
- Interchangeable implementations for comparison studies

## Getting Started

1. Choose the feature guide for your use case
2. Review the available strategies and their parameters
3. Use builders for validated configuration or factories for dynamic selection
4. Integrate with Particula's dynamics workflow using Runnable objects

## Related Documentation

- **Examples**: See [Examples Gallery](../Examples/index.md) for working code
- **Theory**: See [Theory Documentation](../Theory/index.md) for scientific background
- **Tutorials**: See [Tutorials](../Examples/index.md#tutorials) for step-by-step learning
