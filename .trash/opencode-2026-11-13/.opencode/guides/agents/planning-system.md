# Planning Agent System - Usage Guide

**Version:** 1.0.0
**Last Updated:** 2025-12-06

## Overview

The Planning Agent System is a multi-agent architecture for creating high-quality implementation plans from GitHub issues. It uses a research subagent and five sequential reviewer subagents to ensure plans are comprehensive, feasible, and complete.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│              plan_work_multireview (Primary Agent)                  │
│                         mode: primary                               │
│                         model: base                                 │
│                                                                     │
│  Responsibilities:                                                  │
│  - Read issue from adw_state.json                                   │
│  - Orchestrate research and review subagents                        │
│  - Generate and revise implementation plan                          │
│  - GUARANTEE spec_content output                                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
           ┌───────────────────────┴───────────────────────┐
           │                                               │
           ▼                                               ▼
┌─────────────────────┐                    ┌─────────────────────────────┐
│   PHASE 1: RESEARCH │                    │  PHASE 2: SEQUENTIAL REVIEW │
└─────────────────────┘                    └─────────────────────────────┘
           │                                               │
           ▼                                               ▼
┌─────────────────────┐              ┌─────────────────────────────────────┐
│ codebase-researcher │              │ Review Chain (in order):            │
│    (model: light)   │              │                                     │
│                     │              │ ① plan_work_architecture-reviewer   │
│ Produces:           │              │         ↓ revise plan               │
│ - Code snippets     │              │ ② plan_work_implementation-reviewer │
│ - file:line refs    │              │         ↓ revise plan               │
│ - Module structure  │              │ ③ plan_work_performance-reviewer    │
│ - Patterns observed │              │         ↓ revise plan               │
└─────────────────────┘              │ ④ plan_work_testing-reviewer        │
                                     │         ↓ revise plan               │
                                     │ ⑤ plan_work_completeness-reviewer   │
                                     │         ↓ final revision            │
                                     └─────────────────────────────────────┘
```

## Agent Summary

### Primary Agent

| Agent | Mode | Model | Purpose |
|-------|------|-------|---------|
| **plan_work_multireview** | `primary` | `base` | Orchestrates research and reviews |

### Research Subagent

| Subagent | Mode | Model | Purpose |
|----------|------|-------|---------|
| **codebase-researcher** | `subagent` | `light` | Gather codebase context |

### Review Subagents (Sequential Order)

| Order | Subagent | Mode | Model | Focus |
|-------|----------|------|-------|-------|
| 1 | **plan_work_architecture-reviewer** | `subagent` | `light` | Architectural fit |
| 2 | **plan_work_implementation-reviewer** | `subagent` | `light` | Feasibility |
| 3 | **plan_work_performance-reviewer** | `subagent` | `light` | Efficiency |
| 4 | **plan_work_testing-reviewer** | `subagent` | `light` | Test coverage |
| 5 | **plan_work_completeness-reviewer** | `subagent` | `light` | Final quality gate |

## Why Multi-Review?

### Problem Solved

The original `plan-work` agent had a 20% failure rate (2/10 runs) where `spec_content` was not produced. Failures occurred when:
- Agent got lost in exploration
- Agent didn't understand issue scope
- Agent forgot to write final output

### Solution: Research + Review

1. **Research First**: Dedicated subagent gathers context with bounded scope
2. **Focused Planning**: Main agent plans with pre-gathered context
3. **Sequential Review**: Each reviewer catches different issues
4. **Revise After Each**: Plan improves incrementally
5. **Retry Capability**: If stuck, can request more research
6. **Guaranteed Output**: Final step always writes spec_content

### Cost Optimization

Using `light` model (Haiku) for subagents instead of `base` (Sonnet):
- 5 light model calls ≈ 1 base model call cost
- Similar quality through specialization
- Faster execution per subagent

## Invocation

### Via ADW Workflow

```bash
# Use multi-review planning (recommended)
uv run adw workflow run plan_work_multireview <issue-number>

# Or as part of a complete workflow (if configured)
uv run adw complete <issue-number>
```

### Direct Agent Invocation

```bash
# Primary orchestrator
opencode /plan_work_multireview <issue-number> --adw-id <adw-id>
```

## Execution Flow

### Step 1: Setup
- Extract ADW ID from arguments
- Load issue from `adw_state.json`
- Create tracking todo list

### Step 2: Research
```python
task({
  "description": "Research codebase for planning",
  "prompt": f"Research codebase...\n\nArguments: adw_id={adw_id}",
  "subagent_type": "codebase-researcher"
})
```

**Output:** Structured context with code snippets and file:line references

### Step 3: Draft Plan
Using research context, generate initial implementation plan

### Step 4: Sequential Review Loop

For each reviewer in order:

```python
task({
  "description": "{reviewer_name} review",
  "prompt": f"Review plan...\n\nArguments: adw_id={adw_id}\n\nPlan:\n{current_plan}\n\nContext:\n{research_context}",
  "subagent_type": "plan_work_{reviewer_name}"
})
```

After each review:
- Parse feedback
- If `NEEDS_REVISION`: Revise plan
- If critical issue unresolvable: Request more research, retry (max 2)
- Continue to next reviewer with revised plan

### Step 5: Write Output
```python
adw_spec({
  "command": "write",
  "adw_id": "{adw_id}",
  "content": "{final_plan}"
})
```

**GUARANTEED** to execute and produce output.

## Review Focus Areas

### Architecture Reviewer (1st)
- Module boundaries correct?
- Files in right locations?
- Design patterns followed?
- No anti-patterns?

### Implementation Reviewer (2nd)
- File paths exist?
- Code changes feasible?
- Dependencies in correct order?
- Instructions specific enough?

### Performance Reviewer (3rd)
- Any O(n²) algorithms?
- Unnecessary I/O?
- API rate limit risks?
- Resource management?

### Testing Reviewer (4th)
- Test coverage complete?
- Error cases tested?
- Edge cases covered?
- Correct test file naming?

### Completeness Reviewer (5th)
- All acceptance criteria addressed?
- Error handling complete?
- Documentation planned?
- Rollback considered?

## Output Signals

| Agent | Success Signal | Failure Signal |
|-------|---------------|----------------|
| plan_work_multireview | `IMPLEMENTATION_PLAN_COMPLETE` | `IMPLEMENTATION_PLAN_FAILED` |
| codebase-researcher | `CODEBASE_RESEARCH_COMPLETE` | `CODEBASE_RESEARCH_FAILED` |
| architecture-reviewer | `ARCHITECTURE_REVIEW_COMPLETE` | `ARCHITECTURE_REVIEW_FAILED` |
| implementation-reviewer | `IMPLEMENTATION_REVIEW_COMPLETE` | `IMPLEMENTATION_REVIEW_FAILED` |
| performance-reviewer | `PERFORMANCE_REVIEW_COMPLETE` | `PERFORMANCE_REVIEW_FAILED` |
| testing-reviewer | `TESTING_REVIEW_COMPLETE` | `TESTING_REVIEW_FAILED` |
| completeness-reviewer | `COMPLETENESS_REVIEW_COMPLETE` | `COMPLETENESS_REVIEW_FAILED` |

## Reviewer Output Format

Each reviewer produces:
- **Status**: `PASS` or `NEEDS_REVISION`
- **Issues Found**: Categorized as CRITICAL, WARNING, SUGGESTION
- **Recommended Changes**: Specific fixes
- **Verified Correct**: What passed review

## Example Plan Output

```markdown
# Implementation Plan: Fix IndexError in data parser

**Issue:** #123
**Type:** /bug
**Branch:** bug-issue-123-fix-indexerror
**Generated:** Multi-review process (5 reviewers)

## Overview
Parser throws IndexError on empty input. Add bounds checking and validation.

## Codebase Context
- Key file: `adw/utils/parser.py:120-130` - parse_data function
- Pattern: Uses `ADWError` for custom exceptions
- Tests: `adw/utils/tests/parser_test.py` exists

## Steps

### Step 1: Add Input Validation
**Files:** `adw/utils/parser.py:120-125`
**Details:**
- Add check: `if not data: raise ValueError("Empty data")`
- Add check: `if len(data) < 3: raise ValueError("Need 3+ elements")`
- Follow error message format from code_style.md
**Validation:** Parser raises ValueError, not IndexError
**Error Handling:** Catch in caller, log, re-raise as ADWError

### Step 2: Add Regression Tests
**Files:** `adw/utils/tests/parser_test.py`
**Details:**
- `test_parse_data_empty_list_raises_value_error()`
- `test_parse_data_short_list_raises_value_error()`
- `test_parse_data_none_raises_type_error()`
**Validation:** Tests pass with pytest

## Tests to Write
- `parser_test.py`: Empty input test
- `parser_test.py`: Short list test
- `parser_test.py`: None input test

## Error Handling
- Empty input: Raise ValueError with clear message
- None input: Raise TypeError

## Acceptance Criteria
- [ ] Parser handles empty input without IndexError
- [ ] Clear error messages for invalid inputs
- [ ] Tests cover edge cases

## Review Notes
- Architecture: Using ADWError hierarchy ✓
- Implementation: File paths verified ✓
- Performance: No bottlenecks ✓
- Testing: Full coverage planned ✓
- Completeness: All criteria addressed ✓
```

## Retry Logic

### When to Retry Research

If a reviewer finds a critical issue that can't be resolved with available context:

1. **Identify gap**: What information is missing?
2. **Request research**:
```python
task({
  "description": "Additional research",
  "prompt": f"Need to answer: {specific_questions}",
  "subagent_type": "codebase-researcher"
})
```
3. **Revise plan** with new information
4. **Re-run reviewer** (max 2 retries per reviewer)

### Retry Limits
- Per reviewer: 2 retry attempts
- Total research calls: Up to 6 (initial + 5 potential retries)

## Troubleshooting

### Plan Not Complete
**Cause**: Completeness reviewer found gaps
**Solution**: Check review notes, address missing criteria

### Review Loop Stuck
**Cause**: Conflicting feedback from reviewers
**Solution**: Primary agent resolves conflicts, prioritizes later reviews

### Research Context Insufficient
**Cause**: Issue description too vague
**Solution**: More specific issue with file references helps

### Subagent Timeout
**Cause**: Large codebase or complex issue
**Solution**: Narrow research focus, split into smaller issues

## Comparison: plan-work vs plan_work_multireview

| Aspect | plan-work | plan_work_multireview |
|--------|-----------|----------------------|
| **Model** | base | base + 6 light |
| **Reliability** | ~80% | ~98% (target) |
| **Quality** | Good | Higher (5 reviews) |
| **Cost** | 1 base call | 1 base + ~6 light (~2x) |
| **Time** | Fast | Slower (more calls) |
| **Context** | Self-researched | Pre-researched |
| **Output** | Sometimes missing | Guaranteed |

## When to Use Which

**Use `plan-work`:**
- Simple, well-defined issues
- When speed matters more than quality
- Budget-constrained scenarios

**Use `plan_work_multireview`:**
- Complex or vague issues
- Reliability is critical
- Quality matters more than speed
- Previous plan-work runs failed

## Configuration

### Model Selection

Subagents use `light` model by default. To use `base`:
- Edit subagent `.md` files
- Change `model: light` to `model: base`
- Note: Increases cost significantly

### Enabling/Disabling Reviewers

To skip a reviewer (not recommended):
1. Edit `plan_work_multireview.md`
2. Remove the reviewer's step
3. Update todo list

## See Also

- [plan-work.md](./plan-work.md) - Original single-agent planning
- [execute-plan.md](./execute-plan.md) - Plan execution agent
- [documentation-system.md](./documentation-system.md) - Similar multi-subagent pattern
- [architecture_reference.md](../architecture_reference.md) - Architecture patterns
- [testing_guide.md](../testing_guide.md) - Testing conventions
