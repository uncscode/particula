# Code Culture & Development Philosophy

This document describes the cultural principles and development philosophy for **particula**.

## Core Philosophy: Smooth is Safe, Safe is Fast

Our development philosophy is built on the principle that **smooth is safe, and safe is fast**. This means:

- **Smooth reviews** happen when changes are small, focused, and easy to understand
- **Safe changes** happen when reviews are thorough and careful
- **Fast development** happens when we minimize rework, bugs, and rollbacks

By optimizing for smooth reviews, we naturally achieve safe and fast development.

## The 100-Line Rule

### Guideline
Each GitHub issue and its corresponding PR should add or modify approximately **100 lines of code or less** (excluding tests and documentation).

### Rationale
- **Cognitive load:** Reviewers can thoroughly understand ~100 lines in one sitting
- **Context retention:** Small changes maintain clear mental models
- **Review quality:** Deep, careful review is only sustainable for small changes
- **Risk mitigation:** Smaller changes = smaller blast radius if something goes wrong
- **Debugging:** Easier to identify the source of issues
- **Merge conflicts:** Fewer conflicts with smaller, focused changes

### What Counts
**Counts toward the 100-line limit:**
- Production code in `particula/`
- Configuration files that affect runtime behavior (pyproject.toml changes)
- Scientific algorithm implementations

**Does NOT count:**
- Test files (`*_test.py`)
- Documentation (`.md` files, Jupyter notebooks)
- Code comments and docstrings
- Generated code
- Dependency updates (pyproject.toml dependencies section only)

### Examples

**Good - Within Limit:**
```
✓ Add rate limiting middleware (85 lines)
✓ Implement user authentication (95 lines)
✓ Refactor config parser (70 lines)
```

**Too Large - Should Be Split:**
```
✗ Complete user management system (450 lines)
  → Split into:
    ✓ Phase 1: Add user model (60 lines)
    ✓ Phase 2: Implement authentication (80 lines)
    ✓ Phase 3: Add authorization (75 lines)
    ✓ Phase 4: Create user API endpoints (90 lines)
    ✓ Phase 5: Add user management UI (95 lines)
```

## Breaking Down Large Features

### When to Split
If your implementation exceeds ~100 lines, split it into phases.

### How to Split

1. **Vertical Slicing:** Each phase delivers a complete, testable piece of functionality
   ```
   Bad:  Phase 1: Write all models, Phase 2: Write all controllers, Phase 3: Write all views
   Good: Phase 1: User CRUD, Phase 2: Admin CRUD, Phase 3: Reporting
   ```

2. **Dependency Ordering:** Later phases depend on earlier phases
   ```
   Phase 1: Database schema → Phase 2: API layer → Phase 3: UI
   ```

3. **Independent Work:** If tasks can be done in parallel, create separate issues
   ```
   Instead of: "Integrate with payment providers (3 providers, 300 lines)"
   Create:     Issue #1: Stripe integration (95 lines)
               Issue #2: PayPal integration (88 lines)
               Issue #3: Square integration (92 lines)
   ```

### Phase Documentation

Every phase should be documented as a separate GitHub issue with:
- Clear, focused scope
- Estimated lines of code
- Dependencies on other phases/issues
- Acceptance criteria
- Testing requirements (minimum 90% coverage)
- Scientific references (if applicable)

See `docs/Agent/feature/template.md` and `docs/Agent/maintenance/template.md` for templates.

## Code Review Culture

### For Authors

**Before Creating PR:**
- [ ] Changes are focused on a single concern
- [ ] Code is ≤100 lines (excluding tests/docs)
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Self-reviewed code

**PR Description:**
- Explain the "why" not just the "what"
- Link to relevant issues and ADRs
- Highlight areas that need special attention
- Include testing instructions

### For Reviewers

**Review Thoroughly:**
- Take time to understand the context
- Check logic, not just syntax
- Consider edge cases and failure modes
- Verify tests adequately cover changes
- Ensure documentation is updated

**Review Kindly:**
- Assume good intent
- Ask questions before making statements
- Praise good work
- Suggest improvements, don't demand perfection
- Remember: we're all learning

**Review Promptly:**
- Small PRs enable fast reviews
- Target: review within 4 hours for PRs ≤100 lines
- Unblock teammates quickly

## Issue Management

### Issue Scope

Each GitHub issue should represent:
- **One focused change:** ~100 lines of code
- **Clear acceptance criteria:** Measurable completion conditions
- **Well-defined boundaries:** Know exactly what's in/out of scope
- **Testable outcome:** Can verify completion

### Issue Workflow

1. **Create tracking doc:** Use feature/maintenance templates
2. **Break into phases:** Each phase = one issue (~100 lines)
3. **Create GitHub issues:** One per phase
4. **Link dependencies:** Mark which issues depend on others
5. **Implement sequentially:** Complete phases in order
6. **Review and merge:** Small PRs enable fast reviews

### Labels

Use labels to communicate intent:
- `agent` - Can be completed by AI coding agent
- `enhancement` - New feature or request
- `bug` - Something isn't working
- `documentation` - Docs improvements
- `chore` - Maintenance task
- `workflow:pending` - Not yet started
- `workflow:in-progress` - Currently being worked on
- `workflow:blocked` - Waiting on dependency

## Testing Culture

### Test Coverage

- Aim for **90%+** test coverage
- Minimum **500 tests** required (current: 711 tests)
- Every feature should have:
  - Unit tests for core logic (scientific functions)
  - Integration tests for component interaction
  - Tests for both scalar and array inputs (scientific code)
  - Edge case tests (zero, negative, extreme values)

### Test Quality

**Good tests:**
- Are fast (unit tests < 100ms each)
- Are independent (can run in any order)
- Test behavior, not implementation
- Have clear failure messages
- Are maintainable

**Write tests first when:**
- Fixing bugs (reproduce first, then fix)
- Implementing well-defined requirements
- Refactoring (tests ensure no behavior change)

## Documentation Culture

### Types of Documentation

1. **Code Documentation**
   - Google-style docstrings for public APIs
   - Type hints in all function signatures
   - Comments for complex scientific algorithms
   - Scientific citations in module docstrings
   - See [Docstring Guide](docstring_guide.md)

2. **Architecture Documentation**
   - ADRs for significant decisions
   - Architecture guides for system structure
   - See [Architecture Reference](architecture_reference.md)
   - See `docs/Agent/architecture/decisions/`

3. **User Documentation**
   - README for getting started
   - Tutorial notebooks in `docs/Examples/`
   - API documentation (auto-generated from docstrings)
   - Theory documentation in `docs/Theory/`
   - MkDocs site: https://uncscode.github.io/particula

### When to Update Documentation

Update documentation when:
- Adding new features
- Changing public APIs
- Making architecture decisions
- Fixing non-obvious bugs
- Discovering tribal knowledge

**Rule:** If you have to explain something twice, document it once.

## Communication

### Async by Default

- Use GitHub issues and PRs for work discussions
- Document decisions in ADRs
- Keep team updated via written status
- Respect focus time - minimize interruptions

### Synchronous When Needed

Schedule meetings for:
- Complex design discussions
- Conflict resolution
- Team building and retros
- Unblocking urgent issues

## Continuous Improvement

This culture document evolves. If you find a practice that works better:

1. Try it out
2. Measure results
3. Share with team
4. Update this document
5. Create ADR if significant change

## Key Principles Summary

1. **Small Changes:** ~100 lines of code per PR
2. **Focused Scope:** One issue = one concern
3. **Smooth Reviews:** Easy to understand = thorough review
4. **Safe Deployment:** Small changes = small risk
5. **Fast Iteration:** Quick reviews = rapid progress
6. **Clear Documentation:** Write it down
7. **Test Thoroughly:** Trust but verify
8. **Continuous Learning:** Always improving

Remember: **Smooth is safe, and safe is fast.**
