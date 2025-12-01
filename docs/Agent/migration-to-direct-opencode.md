# Migration Guide: Backend Abstraction to Direct OpenCode

## Overview

ADW v3.0.0 removes the backend abstraction layer in favor of direct OpenCode execution.

## Breaking Changes

### Removed Modules
- `adw.backends` - Entire module removed
- `BackendFactory` - Use `execute_opencode_agent()` instead
- `AgentBackend` - No longer needed
- `OpenCodeBackend` - Functionality moved to `adw.core.opencode`

### API Changes

**Before:**
```python
from adw.backends import get_backend
backend = get_backend()
response = backend.execute_template(request)
```

**After:**
```python
from adw.core.opencode import execute_opencode_agent
response = execute_opencode_agent(
    prompt=prompt,
    adw_id=adw_id,
    agent_name="implementor",
    model_tier="base",
)
```

### Configuration Changes

**Before (.env):**
```bash
AGENT_CLI_TOOL=opencode
```

**After (.env):**
```bash
OPENCODE_MODEL_LIGHT=opencode/claude-3-5-haiku
OPENCODE_MODEL_BASE=opencode/claude-sonnet-4
OPENCODE_MODEL_HEAVY=opencode/claude-opus-4-1
```

### Model Selection Changes

**Before:** Used model names (`haiku`, `sonnet`, `opus`)
**After:** Uses model tiers (`light`, `base`, `heavy`)

## Migration Steps

1. Update imports from `adw.backends` to `adw.core.opencode`
2. Replace `get_backend()` calls with `execute_opencode_agent()`
3. Update model references from names to tiers
4. Update .env configuration (remove AGENT_CLI_TOOL, add tier variables)
5. Test workflows with new execution path

## Benefits

- Simpler codebase (~400 fewer lines)
- Direct access to all OpenCode CLI flags
- Environment-driven model selection
- Dynamic agent/command discovery
- Better performance (fewer indirection layers)