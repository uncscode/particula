# Agent Creator - Quick Reference Card

## Creating a New Agent

```
"I need an agent to [specific task] with [permissions and scope]"
```



**Default to read-only unless write is essential.**

## File Restrictions (for write mode)

Specify allowed file types:
- Documentation: `.md`, `.txt`, `.rst`
- Configuration: `.yaml`, `.yml`, `.json`, `.toml`
- Code: `.py`, `.ts`, `.js`, `.rs`
- Tests: `*_test.py`, `*.test.ts`

## Context Files by Agent Type

**Architecture/Planning**:
- `docs/Agent/architecture_reference.md`
- `docs/Agent/architecture/architecture_guide.md`

**Implementation**:
- `docs/Agent/code_style.md`
- `docs/Agent/testing_guide.md`
- `docs/Agent/linting_guide.md`

**Documentation**:
- `docs/Agent/documentation_guide.md`
- `docs/Agent/docstring_guide.md`

**Review**:
- `docs/Agent/review_guide.md`
- `docs/Agent/code_style.md`

**Features**:
- `docs/Agent/feature/`
- `docs/Agent/architecture_reference.md`

## Tool Recommendations

- `get_version` - Project version info
- `get_date` - Timestamps
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
"Create an agent to implement features from specs in docs/Agent/feature/"
```

### Test Generator
```
"Create an agent that generates pytest test files following our *_test.py naming"
```

## After Creation

1. ✅ Review generated content
2. ✅ Save to `.opencode/agent/[name].md`
3. ✅ Save docs to `docs/Agent/agents/[name].md`
4. ✅ Test with examples
5. ✅ Update `docs/Agent/agents/README.md`

## Links

- **Agent Definition**: `adw/templates/opencode_config/agent/agent_creator.md`
- **Usage Guide**: `docs/Agent/agents/agent-creator.md`
- **OpenCode Docs**: https://opencode.ai/docs/agents/
