# Feature Tracking

This directory contains tracking documents for features being developed or planned for particula.

## Current Features

### Completed
- [P2-charge-conservation-coagulation.md](P2-charge-conservation-coagulation.md) - Charge conservation in particle-resolved coagulation

### In Progress
(None currently)

### Backlog
(None currently)

## Organization

Each feature gets its own tracking file in this directory. The file naming convention is:

```
<priority>-<feature-name>.md
```

**Priority levels:**
- `P0` - Critical/Urgent features
- `P1` - High priority features
- `P2` - Medium priority features
- `P3` - Low priority features
- `Backlog` - Future features not yet prioritized

**Examples:**
- `P0-user-authentication.md` - Critical authentication system
- `P1-api-rate-limiting.md` - High priority rate limiting
- `P2-dark-mode.md` - Medium priority UI enhancement
- `Backlog-export-to-pdf.md` - Future export functionality

## Single-Phase vs Multi-Phase Features

### Single-Phase Features
Features that can be completed in a single implementation phase (~100 lines of code or less).

**Example:** `P2-dark-mode.md`
- Single file contains everything
- Lists Phase 1 only
- Includes all implementation details in the file

### Multi-Phase Features
Large features with distinct phases that must be completed in order, where each phase is ~100 lines of code.

**Example:** Multi-tenant architecture: `P0-multi-tenant-architecture.md`
- **One file** contains overview and all phases
- Lists Phase 1, Phase 2, Phase 3, etc. in the Phases section
- Each phase gets its own GitHub issue
- Implementation details for each phase in separate sections of the same file

**File structure:**
```markdown
# Feature: Multi-Tenant Architecture

## Phases
- [ ] Phase 1: Data isolation (~75 lines) - Issue #301
- [ ] Phase 2: Tenant management (~90 lines) - Issue #302
- [ ] Phase 3: Billing integration (~85 lines) - Issue #303

## Phase 1: Data Isolation
[Detailed implementation tasks for Phase 1]

## Phase 2: Tenant Management
[Detailed implementation tasks for Phase 2]

## Phase 3: Billing Integration
[Detailed implementation tasks for Phase 3]
```

### Independent Sub-features
If a large feature has multiple sub-features that can be developed independently, create separate files for each:

**Example:** Payment system with multiple providers
- `P1-payment-stripe.md` - Stripe integration (can be done independently)
- `P1-payment-paypal.md` - PayPal integration (can be done independently)
- `P1-payment-square.md` - Square integration (can be done independently)

## Feature Lifecycle

1. **Proposed** - Feature is documented but not yet approved
2. **Approved** - Feature is approved and prioritized
3. **In Progress** - Active development
4. **Review** - Implementation complete, under review
5. **Testing** - In testing phase
6. **Completed** - Feature is complete and deployed
7. **Cancelled** - Feature was cancelled

## Using the Template

Copy `template.md` to create a new feature tracking file:

```bash
cp docs/Agent/feature/template.md docs/Agent/feature/P1-my-feature.md
```

Then fill in:
- Feature overview and description
- User stories or requirements
- Technical approach
- Implementation tasks
- Dependencies and blockers
- Testing strategy
- Success criteria

## Cross-References

Features often reference:
- **Architecture decisions:** `docs/Agent/architecture/decisions/`
- **Maintenance items:** `docs/Agent/maintenance/`
- **GitHub issues:** https://github.com/Gorkowski/particula/issues
- **Pull requests:** https://github.com/Gorkowski/particula/pulls

## Examples

See these example feature tracking files:
- `example-single-phase.md` - Simple feature with one implementation phase
- `example-multi-phase.md` - Complex feature with multiple sequential phases
- `example-independent-subfeatures.md` - Feature with parallel sub-features
