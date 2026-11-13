# ADW Review System - Usage Guide

## Overview

The ADW Review System is a multi-agent code review architecture that provides comprehensive, high-quality code reviews for pull requests and auto-workflow slices without PR artifacts. It uses specialized reviewer agents working in parallel, followed by a consolidation phase that deduplicates and ranks findings, then persists the canonical review result into workflow state and optionally posts structured feedback to the PR.

**Key Design Principles:**
- **Multi-agent ensemble**: Multiple specialized reviewers catch different types of issues
- **Parallel execution**: Reviewers run simultaneously for faster reviews
- **Consolidation**: Duplicate removal and severity ranking reduces noise
- **Actionable feedback**: Every finding includes specific code fixes
- **Platform-agnostic**: Works with both GitHub and GitLab when a PR/MR exists
- **State-first handoff**: Review output is always written to `adw_state.json` for downstream fix planning
- **Todo tracking**: Progress tracked via todo list for visibility

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        adw-review-orchestrator                               │
│                           (primary agent)                                    │
│  - Builds todo list for tracking                                             │
│  - Dispatches 7 required + 1 optional reviewer in parallel                   │
└─────────────────────────────────────────────────────────────────────────────┘
        │         │         │         │         │         │         │         │
        ▼         ▼         ▼         ▼         ▼         ▼         ▼         ▼
   ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
   │ code-  ││correct-││  cpp-  ││ python-││security││  test- ││  doc-  ││ arch-  │
   │quality ││  ness  ││  perf  ││  perf  ││reviewer││coverage││ument- ││itecture│
   │reviewer││reviewer││reviewer││reviewer││        ││reviewer││  ation ││(opt.)  │
   └────────┘└────────┘└────────┘└────────┘└────────┘└────────┘└────────┘└────────┘
     Style,    Bugs,   OpenMP/   NumPy/    Memory,   Missing   Docstring  Module
    Idioms,  Edge cases, MPI,    Numba,   Input val,  tests,    quality,  bounds,
    Linting  Numerical Kokkos,  Multiproc GPU safety  quality  type hints  APIs
        │         │         │         │         │         │         │         │
        └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                                          │
                                          ▼
                                ┌──────────────────┐
                                │  consolidation-  │
                                │     reviewer     │
                                │  Merge, Dedupe,  │
                                │  Rank, Filter    │
                                └──────────────────┘
                                          │
                                          ▼
                                ┌──────────────────┐
                                │ review-state-    │
                                │    writer        │
                                │ request_fix +    │
                                │ findings payload │
                                └──────────────────┘
                                          │
                                          ▼
                                ┌──────────────────┐
                                │  feedback-       │
                                │     poster       │
                                │ Optional PR/MR   │
                                │ comments         │
                                └──────────────────┘
```

## Agents

### 1. adw-review-orchestrator (Primary)

**Purpose:** Coordinate the entire review process.

**Mode:** `primary`

**Responsibilities:**
- Parse review context (diff, description, changed files, PR metadata when available)
- Build todo list to track review progress
- Dispatch specialized reviewers in parallel (7 required + 1 optional)
- Route C++ files to cpp-performance, Python files to python-performance
- Optionally invoke architecture review (large PRs or `--include-architecture` flag)
- Invoke consolidation after all reviewers complete
- Invoke review-state-writer to persist `request_fix`, `review_feedback`, and `review_findings`
- Invoke feedback-poster only when a PR/MR number exists

**Key Tools:** `task`, `platform_operations`, `run_linters`, `git_diff`,
`git_stage`, `git_commit`, `git_branch`, `git_merge`, `git_worktree`, `todowrite`

---

### 2. adw-review-code-quality (Subagent)

**Purpose:** Review code style, readability, and maintainability.

**Mode:** `subagent`

**Focus Areas:**
- PEP8 compliance (Python)
- Modern C++ style guidelines
- Naming conventions
- Code smells (long functions, deep nesting, duplication)
- Documentation completeness
- Linter findings integration

**Languages:** Python and C++

---

### 3. adw-review-correctness (Subagent)

**Purpose:** Find bugs, logic errors, and edge case issues.

**Mode:** `subagent`

**Focus Areas:**
- Logic bugs and incorrect implementations
- Edge case handling (empty inputs, boundaries)
- Numerical stability (overflow, NaN, precision)
- Concurrency issues (race conditions, deadlocks)
- Error handling completeness

**Languages:** Python and C++

---

### 4. adw-review-cpp-performance (Subagent)

**Purpose:** HPC-specific C++ performance optimization.

**Mode:** `subagent`

**Focus Areas:**
- OpenMP patterns (false sharing, load balance, barriers)
- MPI communication efficiency
- Kokkos usage and memory spaces
- CUDA/GPU optimization (coalescing, divergence)
- Cache efficiency and memory access patterns
- Vectorization (SIMD, restrict pointers)
- Memory allocation in hot paths

**Languages:** C++ only (.cpp, .hpp, .h, .cu, .cuh)

---

### 5. adw-review-python-performance (Subagent)

**Purpose:** Python performance optimization for scientific computing.

**Mode:** `subagent`

**Focus Areas:**
- NumPy vectorization (avoiding Python loops)
- Numba JIT compilation patterns
- Multiprocessing vs threading (GIL awareness)
- Pandas efficiency (avoiding iterrows)
- Memory efficiency (copies, generators)
- Python anti-patterns (string concatenation in loops)

**Languages:** Python only (.py)

---

### 6. adw-review-security (Subagent)

**Purpose:** Security and robustness review.

**Mode:** `subagent`

**Focus Areas:**
- Memory safety (buffer overflow, use-after-free)
- Input validation
- Error handling completeness
- Resource management (leaks, exhaustion)
- Unsafe deserialization (pickle, eval)
- GPU-specific security (CUDA error checking)
- Path traversal and injection risks

**Languages:** Python and C++

**Key Tools:** `run_linters` (for security-related ruff rules)

---

### 7. adw-review-test-coverage (Subagent)

**Purpose:** Analyze test coverage for changed code (read-only).

**Mode:** `subagent`

**Focus Areas:**
- Missing tests for new/changed functions
- Test file naming conventions (`*_test.py`)
- Test quality (assertions, edge cases)
- Mock usage appropriateness
- Test organization and structure

**Note:** Does NOT run tests - performs static analysis only.

**Languages:** Python and C++

---

### 8. adw-review-documentation (Subagent)

**Purpose:** Review documentation quality.

**Mode:** `subagent`

**Focus Areas:**
- Python docstring completeness (Google-style)
- C++ Doxygen comment quality
- Type hint completeness
- Docstring accuracy (matches code)
- README updates for new features

**Languages:** Python and C++

**Key Tools:** `run_linters` (for docstring rules D100-D419)

---

### 9. adw-review-architecture (Subagent, Optional)

**Purpose:** Review architectural and design concerns.

**Mode:** `subagent`

**Focus Areas:**
- Module boundary violations
- Circular dependency introduction
- Public API breaking changes
- Design pattern consistency
- Separation of concerns
- Import structure and layering

**Invocation:** Optional - triggered by:
- `--include-architecture` flag
- PRs with >10 changed files
- Changes to `adw/core/`
- New modules added

**Languages:** Python and C++

---

### 10. adw-review-consolidation (Subagent)

**Purpose:** Merge, deduplicate, and rank findings from all reviewers.

**Mode:** `subagent`

**Responsibilities:**
- Parse all reviewer outputs (including test, docs, architecture)
- Identify and merge duplicate findings
- Rank by severity × actionability
- Apply reviewer-specific handling rules
- Filter false positives and low-value suggestions
- Perform review quality self-check
- Select findings for inline comments vs overview (cap at ~10-15)
- Produce prioritized final review

**Reviewer-Specific Handling:**
- Test coverage: Prioritize public API tests, relax for private helpers
- Documentation: Prioritize public API docstrings, type hints are WARNING
- Architecture: Higher false-positive rate, verify with import analysis

---

### 11. adw-review-state-writer (Subagent)

**Purpose:** Persist canonical review results to workflow state.

**Mode:** `subagent`

**Responsibilities:**
- Parse actionable review outcome from consolidation output
- Write `request_fix` first as the fail-closed workflow gate
- Best-effort write bounded `review_feedback`
- Best-effort write full `review_findings`
- Keep downstream fix planning independent from PR comments

**Key Tools:** `adw_spec`

---

### 12. adw-review-feedback-poster (Subagent)

**Purpose:** Post review results to PR/MR when review platform context exists.

**Mode:** `subagent`

**Responsibilities:**
- Post overview comment with full review summary
- Post inline comments on specific lines
- Format comments for readability
- Handle rate limits and errors
- Work with both GitHub and GitLab

**Key Tools:** `platform_operations`

---

## Usage

### Invoke via Workflow

```bash
# Review a specific PR
adw workflow review <PR-number>

# With explicit adw_id (for resuming)
adw workflow review <PR-number> --adw-id <id>

# Include optional architecture review
adw workflow review <PR-number> --include-architecture
```

### Invoke Manually

The orchestrator can also be invoked directly:

```python
task({
  "description": "Review workflow slice",
  "prompt": "Review this workflow slice for code quality, correctness, performance, and security. Persist results to state even if no PR exists.",
  "subagent_type": "adw-review-orchestrator"
})
```

## Workflow Definition

**File:** `.opencode/workflow/review.json`

```json
{
  "name": "review",
  "version": "1.0.0",
  "description": "Multi-agent code review for PRs or state-only auto runs",
  "workflow_type": "review",
  "steps": [
    {
      "type": "agent",
      "name": "Review",
      "agent": "adw-review-orchestrator",
      "prompt": "Review PR for code quality, correctness, performance, and security...",
      "model": "base",
      "timeout": 1800
    }
  ]
}
```

## Review Output

### Persisted State (always written)

The review system writes these fields to `adw_state.json` after consolidation:
- `request_fix`: boolean workflow gate for the trailing fix pass
- `review_feedback`: bounded summary for status and quick inspection
- `review_findings`: full consolidated review payload for fix planning

### Overview Comment

Posted when a PR/MR exists, with:
- Summary table (Critical/Warning/Suggestion counts)
- Critical issues with full details and code fixes
- Warnings list
- Suggestions list
- Positive observations
- List of inline comment locations

### Inline Comments

Posted on specific lines for high-priority findings when a PR/MR exists:
- CRITICAL issues (all)
- High-actionability WARNINGs
- Limited to ~10-15 comments to avoid noise

### Comment Format

```markdown
**CRITICAL:** {Brief title}

{One-sentence problem description}

**Suggested fix:**
```python
{code_snippet}
```

{One-sentence reason}
```

## Severity Levels

| Level | Meaning | Action |
|-------|---------|--------|
| **CRITICAL** | Security, crashes, data corruption, >10x slowdown | Must fix before merge |
| **WARNING** | Edge cases, maintainability, 2-10x slowdown | Should fix |
| **SUGGESTION** | Style, minor optimizations | Nice to have |

## Language Routing

| File Extension | Reviewers |
|----------------|-----------|
| `.py` | code-quality, correctness, python-performance, security, test-coverage, documentation |
| `.cpp`, `.hpp`, `.h`, `.cu` | code-quality, correctness, cpp-performance, security, test-coverage, documentation |
| `.json`, `.yaml`, `.md` | code-quality only |

**Note:** Architecture reviewer is optional and runs on all file types when enabled.

## HPC-Specific Features

The review system has specialized support for HPC/scientific computing:

### C++ HPC Patterns
- OpenMP parallel regions and directives
- MPI communication patterns
- Kokkos portable parallelism
- CUDA kernel optimization
- Cache and memory hierarchy

### Python Scientific Computing
- NumPy vectorization
- Numba JIT compilation
- Multiprocessing (avoiding GIL)
- Pandas optimization

### Numerical Stability
- Integer/floating-point overflow detection
- NaN/Inf propagation analysis
- Precision loss identification
- Accumulation error warnings

## Configuration

### Timeout

Default timeout is 1800 seconds (30 minutes). Adjust in workflow JSON:

```json
{
  "timeout": 3600
}
```

### Model Tier

Default is `base`. For complex codebases, use `heavy`:

```bash
adw workflow review 42 --model heavy
```

## Troubleshooting

### Review Times Out

- Large PR with many files
- Complex C++ templates taking analysis time
- **Solution:** Increase timeout or split PR into smaller changes

### Missing Inline Comments

- GitHub/GitLab API rate limits
- Comments batched into overview
- No PR/MR available for posting in state-only auto-review runs
- **Solution:** Check overview comment for "Additional Inline Feedback" section

### False Positives

- Pattern flagged incorrectly for your codebase
- **Solution:** Note in PR description if intentional; consolidation filters based on context

### No C++ Performance Review

- PR only contains Python files
- **Expected behavior:** cpp-performance reviewer is skipped automatically

## Integration with ADW

The review system integrates with other ADW workflows:

### Standalone Review
```bash
adw workflow review <PR>
```

### After Complete Workflow
The `review` workflow can be run after `complete` to get a thorough review before merge.
For `complete-auto` / `patch-auto`, review also works without a per-slice PR because
the canonical output is persisted to workflow state first.

### With Ship Workflow
Run review before ship to ensure quality:
```bash
adw workflow review <PR>
# If review passes:
adw workflow ship <issue> --adw-id <id>
```

## Agent Summary Table

| Agent | Mode | Purpose | Required |
|-------|------|---------|----------|
| `adw-review-orchestrator` | primary | Coordinate review process | Yes |
| `adw-review-code-quality` | subagent | Style, idioms, linting | Yes |
| `adw-review-correctness` | subagent | Bugs, edge cases, numerical | Yes |
| `adw-review-cpp-performance` | subagent | C++ HPC optimization | If C++ files |
| `adw-review-python-performance` | subagent | Python scientific computing | If Python files |
| `adw-review-security` | subagent | Memory, input, resources | Yes |
| `adw-review-test-coverage` | subagent | Test existence and quality | Yes |
| `adw-review-documentation` | subagent | Docstrings, type hints | Yes |
| `adw-review-architecture` | subagent | Module bounds, APIs | Optional |
| `adw-review-consolidation` | subagent | Merge, dedupe, rank | Yes |
| `adw-review-state-writer` | subagent | Persist review output to state | Yes |
| `adw-review-feedback-poster` | subagent | Post to PR/MR | Only if PR/MR exists |

## Best Practices

1. **Review early**: Run on draft PRs to catch issues before full review
2. **Trust critical findings**: Always address CRITICAL severity issues
3. **Batch small changes**: Review multiple related commits together
4. **Note intentional patterns**: Use PR description to explain unusual code
5. **Use architecture review for refactors**: Add `--include-architecture` for structural changes
6. **Check test coverage findings**: Ensure new public APIs have tests

## See Also

- [Review Guide](../review_guide.md) - General code review standards
- [Code Style](../code_style.md) - Coding conventions
- [Testing Guide](../testing_guide.md) - Test quality expectations
- [Docstring Guide](../docstring_guide.md) - Documentation standards
- [ADW Build Family](adw-build-family.md) - Implementation agents
