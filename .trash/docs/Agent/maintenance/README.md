# Maintenance Priority Guidelines

This directory contains priority-based maintenance guideline files for particula. These files define ongoing maintenance areas that the automated workflow system monitors and creates issues for.

## Purpose

Rather than tracking individual maintenance tasks, this directory contains **strategic maintenance priorities** that guide the automated issue creation system. The cron trigger periodically scans these files and generates GitHub issues when maintenance work is needed in these priority areas.

## File Naming Convention

```
P<priority>_<maintenance-area>.md
```

**Priority levels:**
- `P0` - Critical/Urgent (security vulnerabilities, critical bugs, system-breaking issues)
- `P1` - High priority (performance issues, major tech debt, test coverage)
- `P2` - Medium priority (code quality improvements, refactoring)
- `P3` - Low priority (nice-to-have improvements, documentation)

**Examples:**
- `P0_security_dependency_updates.md` - Critical security patches
- `P1_increase_test_coverage.md` - Improve test coverage across modules
- `P2_refactor_legacy_code.md` - Refactoring old code patterns
- `P3_improve_documentation.md` - Documentation improvements

## How It Works

1. **Define Priority Areas**: Create a guideline file for each strategic maintenance area
2. **Automated Monitoring**: The cron trigger scans these files periodically (weekly)
3. **Issue Generation**: When no active issue exists for a priority area, the system creates one
4. **Workflow Execution**: The ADW workflow processes the issue and implements the maintenance work

## Guideline File Structure

Each guideline file should contain:

1. **Title**: Clear description of the maintenance area
2. **Priority Justification**: Why this area needs attention
3. **Scope**: Which modules/components are affected
4. **Guidelines**: Specific requirements or standards to meet
5. **Success Criteria**: How to know when work in this area is complete
6. **Examples**: Sample tasks or improvements that fall under this priority

## Creating a New Priority Guideline

1. Copy the template:
   ```bash
   cp template.md P1_your_maintenance_area.md
   ```

2. Fill in the guideline details:
   - Clear title and priority justification
   - Specific scope (files, modules, components)
   - Measurable success criteria
   - Example tasks

3. Commit the file - the cron system will automatically detect it

## Active vs. Completed Guidelines

- **Active**: Guidelines that still need work remain in this directory
- **Completed**: When a maintenance area reaches its success criteria, move the file to a `completed/` subdirectory or delete it
- The cron trigger only processes files in the main maintenance directory (not subdirectories)

## Integration with ADW

The maintenance system integrates with ADW's automated workflow:

1. **Cron Trigger**: `check_maintenance_review()` scans priority guideline files
2. **Issue Creation**: Creates GitHub issues with `maintenance` label and appropriate priority labels
3. **Workflow Execution**: ADW processes the issue using the guideline file as context
4. **Progress Tracking**: Issues track progress, guideline files define long-term priorities

## Example Workflow

```
P1_increase_test_coverage.md exists
    ↓
Cron trigger detects no open maintenance issue for test coverage
    ↓
Creates GitHub issue #123: "maintenance: increase test coverage in triggers module"
    ↓
ADW processes issue #123, implements tests
    ↓
PR merged, issue closed
    ↓
If P1_increase_test_coverage.md still indicates work needed:
    Next week, cron creates new issue for remaining work
```

## Best Practices

1. **Keep Guidelines Strategic**: Focus on areas, not individual tasks
2. **Be Specific**: Clear scope helps the AI understand what to work on
3. **Measurable Criteria**: Define concrete success metrics
4. **Update Regularly**: Keep guidelines current as priorities shift
5. **Archive Completed Work**: Move completed guidelines to show progress

## Priority Decision Guide

**P0 - Critical**:
- Security vulnerabilities requiring immediate patches
- System-breaking bugs affecting production
- Data loss or corruption risks

**P1 - High**:
- Missing test coverage in critical modules
- Performance bottlenecks affecting users
- Major technical debt blocking new features

**P2 - Medium**:
- Code quality improvements
- Non-critical refactoring
- Developer experience enhancements

**P3 - Low**:
- Documentation updates
- Code style consistency
- Nice-to-have optimizations
