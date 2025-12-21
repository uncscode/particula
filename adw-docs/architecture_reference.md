# Architecture Reference

**Version:** 0.2.6
**Last Updated:** 2025-12-02

## Overview

This guide serves as the entry point to the **particula** architecture documentation. It provides references to detailed architectural resources and guidance on when to consult each.

## Architecture Documentation Structure

The architecture documentation is organized as follows:

```
adw-docs/architecture/
├── architecture_guide.md           # Detailed architectural documentation
├── architecture_outline.md         # High-level system overview
└── decisions/                      # Architecture Decision Records (ADRs)
    ├── README.md                   # ADR index and guidelines
    ├── template.md                 # Template for new ADRs
    └── template.md                 # Template for creating new ADRs
```

## Dynamics wall loss strategies

- Implementations live in
  `particula/dynamics/wall_loss/wall_loss_strategies.py`.
- Builders live in `particula/dynamics/wall_loss/wall_loss_builders.py` and
  use mixins in `particula/builder_mixin.py` for geometry/distribution
  validation, unit conversion, and charged parameters (`wall_potential`,
  `wall_electric_field`).
- `WallLossFactory` resides in
  `particula/dynamics/wall_loss/wall_loss_factories.py` and is exported
  alongside strategies via `particula.dynamics.wall_loss` and
  `particula.dynamics`.
- Available strategies: `SphericalWallLossStrategy` (radius),
  `RectangularWallLossStrategy` (validated `(x, y, z)` dimensions in meters),
  and `ChargedWallLossStrategy` (image-charge + optional electric-field drift
  for spherical or rectangular geometry, reduces to neutral when charge/field
  are zero) with builder (`ChargedWallLossBuilder`) and factory entry
  `strategy_type="charged"`.
- Tests are mirrored in `particula/dynamics/wall_loss/tests/` and
  `particula/dynamics/tests/` to cover strategies, builders, and factory
  export paths.

## Quick Navigation

### For New Contributors

Start here to understand the system:

1. **[Architecture Outline](architecture/architecture_outline.md)**: Quick overview of components and structure
2. **[Architecture Guide](architecture/architecture_guide.md)**: Detailed patterns and principles
3. **[Decision Records](architecture/decisions/)**: Historical context for key decisions

### For Implementing Features

When implementing new features, consult:

1. **[Architecture Guide](architecture/architecture_guide.md)**: Ensure your design follows established patterns
2. **[Architecture Outline](architecture/architecture_outline.md)**: Understand module boundaries
3. **[Code Style Guide](code_style.md)**: Follow coding conventions

### For Making Architectural Decisions

When making significant architectural decisions:

1. Review **[Architecture Guide](architecture/architecture_guide.md)** for alignment with principles
2. Review **[Decision Records](architecture/decisions/)** for related past decisions
3. Create a new **[ADR](architecture/decisions/template.md)** to document your decision
4. Request architectural review using `/architecture_review`

### For Code Reviews

When reviewing code for architectural concerns:

1. Check alignment with **[Architectural Principles](architecture/architecture_guide.md#architectural-principles)**
2. Verify adherence to **[Design Patterns](architecture/architecture_guide.md#design-patterns)**
3. Ensure avoidance of **[Anti-Patterns](architecture/architecture_guide.md#anti-patterns)**
4. Reference **[Review Guide](review_guide.md)** for review criteria

## Primary Documentation

### [Architecture Guide](architecture/architecture_guide.md)

The comprehensive architectural documentation covering:

- **Architectural Principles**: Core design principles guiding the system
- **System Architecture**: High-level structure and component organization
- **Design Patterns**: Standard patterns used throughout the codebase
- **Anti-Patterns**: Approaches to avoid
- **Data Flow**: How data moves through the system
- **Error Handling**: Exception hierarchy and error strategies
- **Testing Architecture**: Test organization and strategies
- **Performance & Security**: Key considerations

**When to Read:**
- Designing new modules or major features
- Understanding system-wide patterns
- Making architectural decisions
- Conducting architecture reviews

### [Architecture Outline](architecture/architecture_outline.md)

A high-level overview providing:

- **System Overview**: What the system does
- **Core Components**: Main building blocks and their responsibilities
- **Module Structure**: Directory organization
- **Technology Stack**: Languages, frameworks, and key dependencies
- **Quick Reference**: Design principles and common patterns
- **Extension Points**: Areas designed for customization

**When to Read:**
- First exploring the codebase
- Understanding component responsibilities
- Finding where to add new features
- Getting oriented quickly

### [Architecture Decision Records (ADRs)](architecture/decisions/)

Historical record of significant architectural decisions:

- **Context**: Why the decision was needed
- **Decision**: What was chosen
- **Alternatives**: What else was considered
- **Consequences**: Trade-offs and outcomes

**When to Read:**
- Understanding why things work the way they do
- Reconsidering past decisions in new contexts
- Learning from past trade-offs
- Creating similar decisions

**When to Create:**
- Making significant architectural changes
- Choosing technologies or frameworks
- Establishing new patterns
- Changing system boundaries

See [ADR README](architecture/decisions/README.md) for guidelines on creating ADRs.

## Integration with ADW

ADW commands reference these architecture documents to:

- **Understand Structure**: Know where code belongs
- **Follow Patterns**: Use established approaches
- **Respect Boundaries**: Maintain module separation
- **Make Decisions**: Create ADRs for significant changes

### Relevant ADW Commands

- `/architecture_review`: Review code for architectural consistency
- `/feature`: Plan features using architectural patterns
- `/implement`: Implement following architectural guidelines
- `/review`: Check adherence to architecture

## Related Documentation

- **[Code Style Guide](code_style.md)**: Coding conventions and standards
- **[Testing Guide](testing_guide.md)**: Test organization and patterns
- **[Review Guide](review_guide.md)**: Code review criteria including architecture
- **[Conditional Docs](conditional_docs.md)**: How to document architectural changes

## Maintaining Architecture Documentation

### When to Update

Update architecture documentation when:

- **Adding Major Features**: Update guide and outline with new patterns
- **Changing Module Structure**: Update outline with new organization
- **Making Architectural Decisions**: Create ADR, update guide
- **Deprecating Components**: Update guide, create deprecation ADR
- **Introducing New Patterns**: Add to design patterns section

### How to Update

1. **Make Changes**: Update relevant documentation files
2. **Create ADR**: For significant decisions, create an ADR in `decisions/`
3. **Update Index**: Add new ADRs to [decisions/README.md](architecture/decisions/README.md)
4. **Cross-Reference**: Link related documents
5. **Review**: Get architecture review before finalizing

### Review Process

Architecture documentation changes should be reviewed by:
- Technical leads
- Senior engineers familiar with the system
- Anyone who will be affected by the changes

Use `/architecture_review` to request review.

## Questions?

If you're unsure about:
- **Where something belongs**: Check [Architecture Outline](architecture/architecture_outline.md)
- **What pattern to use**: Check [Architecture Guide](architecture/architecture_guide.md)
- **Why something was done**: Check [Decision Records](architecture/decisions/)
- **Whether to create an ADR**: Check [ADR Guidelines](architecture/decisions/README.md)

When in doubt, ask for an architecture review or consult with technical leads.
