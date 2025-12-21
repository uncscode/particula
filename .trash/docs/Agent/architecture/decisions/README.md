# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the **particula** project.

## What is an ADR?

An Architecture Decision Record (ADR) documents an important architectural decision made along with its context and consequences. ADRs help:

- Preserve institutional knowledge
- Explain why decisions were made
- Track architectural evolution over time
- Onboard new team members
- Revisit decisions when context changes

## When to Create an ADR

Create an ADR when making decisions about:

- **System Architecture**: Component boundaries, module organization
- **Technology Choices**: Languages, frameworks, libraries, tools
- **Design Patterns**: Standardized approaches to common problems
- **Data Models**: Database schemas, data structures, storage strategies
- **APIs**: Interface designs, protocols, contracts
- **Infrastructure**: Deployment, scaling, monitoring strategies
- **Security**: Authentication, authorization, encryption approaches
- **Performance**: Caching strategies, optimization techniques

**Don't** create an ADR for:
- Implementation details within a module
- Bug fixes (unless they reveal architectural issues)
- Routine refactoring
- Code style preferences (use code style guide instead)

## How to Create an ADR

### 1. Copy the Template

```bash
cp template.md 001-your-decision-title.md
```

Use sequential numbering (001, 002, 003, etc.).

### 2. Fill in the Template

Replace all `{{PLACEHOLDERS}}` with actual content:

- **Context**: What problem are you solving? Why does it matter?
- **Decision**: What did you decide to do?
- **Alternatives**: What other options did you consider?
- **Consequences**: What are the positive and negative outcomes?

### 3. Review and Discuss

- Share the draft ADR with team members
- Discuss alternatives and trade-offs
- Update based on feedback
- Set status to "Proposed" during review

### 4. Finalize

- Set status to "Accepted" when consensus is reached
- Add the ADR reference to [architecture_guide.md](../architecture_guide.md)
- Update this README with a link to the new ADR
- Implement the decision

## ADR Index

<!-- Add links to ADRs as they are created -->

### Active Decisions

| ADR | Title | Status | Date | Description |
| --- | ----- | ------ | ---- | ----------- |
| ADR-001 | Strategy-based wall loss subsystem and `wall_loss` package refactor | Accepted | 2025-12-02 | Introduce wall loss strategies and refactor `wall_loss` into a package |


Examples of decisions that should be documented:
- Choosing between different particle distribution representations
- Adopting new coagulation kernel algorithms
- Major refactoring of module boundaries
- Adding new external dependencies (beyond NumPy/SciPy)
- Changing the builder pattern implementation
- Deprecating old API designs

### Superseded Decisions

None yet.

### Deprecated Decisions

None yet.

## ADR Status

Each ADR has a status:

- **Proposed**: Under review and discussion
- **Accepted**: Approved and being implemented
- **Superseded**: Replaced by a newer decision (link to new ADR)
- **Deprecated**: No longer recommended but still in use
- **Rejected**: Considered but not adopted

## Best Practices

### Writing Good ADRs

1. **Be Concise**: Focus on the decision, not implementation details
2. **Provide Context**: Explain why the decision matters
3. **List Alternatives**: Show you considered other options
4. **Be Honest About Trade-offs**: Every decision has pros and cons
5. **Link Related ADRs**: Show how decisions connect

### Maintaining ADRs

- **Update Status**: When a decision is superseded or deprecated
- **Don't Delete**: Keep historical record even for superseded decisions
- **Cross-Reference**: Link bidirectionally between related ADRs
- **Keep Organized**: Use consistent numbering and naming

### Reviewing ADRs

When reviewing someone's ADR:

- Check that alternatives were considered
- Question assumptions
- Suggest additional consequences to consider
- Ensure the rationale is clear
- Verify alignment with architectural principles

## Examples

### Good ADR Titles

- "001-use-postgresql-for-primary-database.md"
- "002-adopt-event-driven-architecture.md"
- "003-implement-plugin-system-for-extensions.md"

### Poor ADR Titles

- "001-fix-bug-in-auth.md" (too specific, not architectural)
- "002-use-react.md" (too vague, lacks context)
- "003-refactor-code.md" (not a decision, just activity)

## Resources

- [Architecture Guide](../architecture_guide.md): Overall architectural documentation
- [Architecture Outline](../architecture_outline.md): High-level system overview
- [Template](template.md): ADR template to copy

## Questions?

If you're unsure whether to create an ADR, ask:
1. Will this decision affect multiple parts of the system?
2. Will future developers need to understand why we made this choice?
3. Are there trade-offs that need to be documented?

If you answered "yes" to any of these, create an ADR!
