# Agent Creator - Quick Reference Card

## Creating a New Agent

```
"I need an agent to [specific task] with [permissions and scope]"
```

## Agent Modes

| Mode | Permissions | Use For |
|------|-------------|---------|
| `read` | Read-only | Review, analysis, recommendations |
| `write` | Read + Write | Implementation, docs, refactoring |
| `all` | Full access + tools | Orchestration, complex workflows |

**Default to read-only unless write is essential.**

## File Restrictions (for write mode)

Specify allowed file types:
- Documentation: `.md`, `.txt`, `.rst`
- Configuration: `.yaml`, `.yml`, `.json`, `.toml`
- Code: `.py`, `.ts`, `.js`, `.rs`
- Tests: `*_test.py`, `*.test.ts`

## Context Files by Agent Type

**Architecture/Planning**:
- `adw-docs/architecture_reference.md`
- `adw-docs/architecture/architecture_guide.md`

**Implementation**:
- `adw-docs/code_style.md`
- `adw-docs/testing_guide.md`
- `adw-docs/linting_guide.md`

**Documentation**:
- `adw-docs/documentation_guide.md`
- `adw-docs/docstring_guide.md`

**Review**:
- `adw-docs/review_guide.md`
- `adw-docs/code_style.md`

**Features**:
- `adw-docs/dev-plans/features/`
- `adw-docs/architecture_reference.md`

## Tool Recommendations

- `get_version` - Project version info
- `get_datetime` - Timestamps (UTC or America/Denver via `localtime`)
- `run_pytest` - Run tests (Python)
- `adw` - Workflow operations

## Design Patterns

1. **Focused Specialist** - Single task, minimal permissions
2. **Workflow Orchestrator** - Multi-step, broader access
3. **Review and Analysis** - Read-only, produces reports
4. **Generator and Builder** - Creates files, restricted paths

## Common Agent Types

| Type | Mode | Example |
|------|------|---------|
| Implementation | `write` or `all` | Execute plans, write code |
| Planning | `read` or limited `write` | Design architecture |
| Review | `read` | Analyze quality |
| Documentation | `write` (`.md` only) | Update README |
| Testing | `write` (`*_test.*` only) | Generate tests |
| Refactoring | `write` | Improve structure |
| Maintenance | `write` | Update dependencies |

## Example Requests

### Security Review
```
"Create a read-only agent that reviews code for security vulnerabilities"
```

### Doc Updater
```
"Create an agent that updates markdown documentation but can't touch code"
```

### Feature Developer
```
"Create an agent to implement features from specs in adw-docs/dev-plans/features/"
```

### Test Generator
```
"Create an agent that generates pytest test files following our *_test.py naming"
```

## After Creation

1. ✅ Review generated content
2. ✅ Save to `.opencode/agent/[name].md`
3. ✅ Save docs to `adw-docs/agents/[name].md`
4. ✅ Test with examples
5. ✅ Update `adw-docs/agents/README.md`

## Links

- **Agent Definition**: `adw/templates/opencode_config/agent/agent_creator.md`
- **Usage Guide**: `adw-docs/agents/agent-creator.md`
- **OpenCode Docs**: https://opencode.ai/docs/agents/
