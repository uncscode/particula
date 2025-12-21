<!--
Summary: Code culture and development philosophy template for {{PROJECT_NAME}}.
Usage:
- Copy into {{DOCS_DIR}}/code_culture.md when scaffolding documentation.
- Replace examples, commands, and links with {{PROJECT_NAME}} specifics.
- Keep sections concise and under the {{LINE_LENGTH}}-character guideline.
Placeholders:
- {{PROJECT_NAME}}
- {{LAST_UPDATED}}
- {{DOCS_DIR}}
- {{SOURCE_DIR}}
- {{TEST_DIR}}
- {{LINE_LENGTH}}
-->

# Code Culture & Development Philosophy

**Project:** {{PROJECT_NAME}}
**Last Updated:** {{LAST_UPDATED}}

This document describes the cultural principles and development philosophy for {{PROJECT_NAME}}.
It should live in `{{DOCS_DIR}}/code_culture.md` and be kept current as practices evolve.

## Core Philosophy: Smooth is Safe, Safe is Fast

Our development philosophy is built on the principle that **smooth is safe, and safe is fast**. This means:

- **Smooth reviews** happen when changes are small, focused, and easy to understand.
- **Safe changes** happen when reviews are thorough and careful.
- **Fast development** happens when we minimize rework, bugs, and rollbacks.

By optimizing for smooth reviews, we naturally achieve safe and fast development.

## The 100-Line Rule

### Guideline
Each GitHub issue and its corresponding PR should add or modify approximately **{{LINE_LENGTH}} lines of
code or less** (excluding tests and documentation).

### Rationale
- **Cognitive load:** Reviewers can thoroughly understand ~{{LINE_LENGTH}} lines in one sitting.
- **Context retention:** Small changes maintain clear mental models.
- **Review quality:** Deep, careful review is only sustainable for small changes.
- **Risk mitigation:** Smaller changes = smaller blast radius if something goes wrong.
- **Debugging:** Easier to identify the source of issues.
- **Merge conflicts:** Fewer conflicts with smaller, focused changes.

### What Counts
**Counts toward the {{LINE_LENGTH}}-line limit:**
- Production code in `{{SOURCE_DIR}}/`.
- Configuration files that affect runtime behavior.
- Database migrations and schema changes.

**Does NOT count:**
- Test files (e.g., `{{TEST_DIR}}/**/*_test.py`).
- Documentation (`.md` files).
- Code comments and docstrings.
- Generated code.
- Dependency updates (e.g., `package.json`, `requirements.txt`).

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
If your implementation exceeds ~{{LINE_LENGTH}} lines, split it into phases.

### How to Split

1. **Vertical Slicing:** Each phase delivers a complete, testable piece of functionality.
   ```
   Bad:  Phase 1: Write all models, Phase 2: Write all controllers, Phase 3: Write all views
   Good: Phase 1: User CRUD, Phase 2: Admin CRUD, Phase 3: Reporting
   ```

2. **Dependency Ordering:** Later phases depend on earlier phases.
   ```
   Phase 1: Database schema → Phase 2: API layer → Phase 3: UI
   ```

3. **Independent Work:** If tasks can be done in parallel, create separate issues.
   ```
   Instead of: "Integrate with payment providers (3 providers, 300 lines)"
   Create:     Issue #1: Stripe integration (95 lines)
               Issue #2: PayPal integration (88 lines)
               Issue #3: Square integration (92 lines)
   ```

### Phase Documentation

Every phase should be documented as a separate GitHub issue with:
- Clear, focused scope.
- Estimated lines of code.
- Dependencies on other phases/issues.
- Acceptance criteria.
- Testing requirements.

See `dev-plans/template-feature.md` and `dev-plans/template-maintenance.md` for templates.

## Code Review Culture

### For Authors

**Before Creating PR:**
- [ ] Changes are focused on a single concern.
- [ ] Code is ≤{{LINE_LENGTH}} lines (excluding tests/docs).
- [ ] All tests pass.
- [ ] Documentation updated.
- [ ] Self-reviewed code.

**PR Description:**
- Explain the "why" not just the "what".
- Link to relevant issues and ADRs.
- Highlight areas that need special attention.
- Include testing instructions.

### For Reviewers

**Review Thoroughly:**
- Take time to understand the context.
- Check logic, not just syntax.
- Consider edge cases and failure modes.
- Verify tests adequately cover changes.
- Ensure documentation is updated.

**Review Kindly:**
- Assume good intent.
- Ask questions before making statements.
- Praise good work.
- Suggest improvements, do not demand perfection.
- Remember: we are all learning.

**Review Promptly:**
- Small PRs enable fast reviews.
- Target: review within 4 hours for PRs ≤{{LINE_LENGTH}} lines.
- Unblock teammates quickly.

## Issue Management

### Issue Scope

Each GitHub issue should represent:
- **One focused change:** ~{{LINE_LENGTH}} lines of code.
- **Clear acceptance criteria:** Measurable completion conditions.
- **Well-defined boundaries:** Know exactly what is in/out of scope.
- **Testable outcome:** Can verify completion.

### Issue Workflow

1. **Create tracking doc:** Use feature/maintenance templates.
2. **Break into phases:** Each phase = one issue (~{{LINE_LENGTH}} lines).
3. **Create GitHub issues:** One per phase.
4. **Link dependencies:** Mark which issues depend on others.
5. **Implement sequentially:** Complete phases in order.
6. **Review and merge:** Small PRs enable fast reviews.

### Labels

Use labels to communicate intent:
- `agent` - Can be completed by AI coding agent.
- `enhancement` - New feature or request.
- `bug` - Something is not working.
- `documentation` - Docs improvements.
- `chore` - Maintenance task.
- `workflow:pending` - Not yet started.
- `workflow:in-progress` - Currently being worked on.
- `workflow:blocked` - Waiting on dependency.

## Testing Culture

### Test Coverage

- Aim for 80%+ test coverage (see `pyproject.toml`).
- Every feature should have:
  - Unit tests for core logic.
  - Integration tests for component interaction.
  - End-to-end tests for critical paths.

### Test Quality

**Good tests:**
- Are fast (unit tests < 100ms each).
- Are independent (can run in any order).
- Test behavior, not implementation.
- Have clear failure messages.
- Are maintainable.

**Write tests first when:**
- Fixing bugs (reproduce first, then fix).
- Implementing well-defined requirements.
- Refactoring (tests ensure no behavior change).

## Documentation Culture

### Types of Documentation

1. **Code Documentation**
   - Docstrings for public APIs.
   - Comments for complex logic.
   - See `architecture/architecture_guide.md` and `docstring_guide.md`.

2. **Architecture Decisions**
   - See `architecture/decisions/`.

3. **User Documentation**
   - README for getting started.
   - API documentation for integrations.
   - Troubleshooting guides.

### When to Update Documentation

Update documentation when:
- Adding new features.
- Changing public APIs.
- Making architecture decisions.
- Fixing non-obvious bugs.
- Discovering tribal knowledge.

**Rule:** If you have to explain something twice, document it once.

## Communication

### Async by Default

- Use GitHub issues and PRs for work discussions.
- Document decisions in ADRs.
- Keep the team updated via written status.
- Respect focus time; minimize interruptions.

### Synchronous When Needed

Schedule meetings for:
- Complex design discussions.
- Conflict resolution.
- Team building and retros.
- Unblocking urgent issues.

## Continuous Improvement

This culture document evolves. If you find a practice that works better:

1. Try it out.
2. Measure results.
3. Share with the team.
4. Update this document.
5. Create an ADR if the change is significant.

## Key Principles Summary

1. **Small Changes:** ~{{LINE_LENGTH}} lines of code per PR.
2. **Focused Scope:** One issue = one concern.
3. **Smooth Reviews:** Easy to understand = thorough review.
4. **Safe Deployment:** Small changes = small risk.
5. **Fast Iteration:** Quick reviews = rapid progress.
6. **Clear Documentation:** Write it down.
7. **Test Thoroughly:** Trust but verify.
8. **Continuous Learning:** Always improving.

Remember: **Smooth is safe, and safe is fast.**
