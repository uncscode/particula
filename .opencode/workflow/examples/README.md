# Workflow Examples

This directory contains example workflow definitions demonstrating common patterns and best practices.

## Available Examples

### conditional-build.json
Demonstrates conditional step execution based on workflow type. Shows how to skip docstring updates for patches while running full validation for complete workflows.

**Use Case:** Build workflows with different validation levels for different workflow types.

### custom-ci.json
Continuous integration validation pipeline with linting, testing, and security checks. Uses light model tier for cost-effective CI operations.

**Use Case:** Pre-commit validation, pull request checks, automated quality gates.

### quick-fix.json
Minimal workflow for rapid patches and hotfixes. Only 2 steps (implement + commit) with short timeouts.

**Use Case:** Urgent hotfixes, typo corrections, simple bug fixes.

## Using Examples

### As Templates

Copy an example to create your own workflow:

```bash
cp .opencode/workflow/examples/conditional-build.json .opencode/workflow/my-workflow.json
```

Then customize:
1. Update `name` to match filename (without .json)
2. Modify `description` and `description_long`
3. Adjust steps, conditions, and parameters
4. Test with a real issue

### Running Examples

Execute examples directly:

```bash
# Run conditional build example
adw workflow conditional-build <issue-number>

# Run CI example
adw workflow custom-ci <issue-number>

# Run quick fix example
adw workflow quick-fix <issue-number>
```

### Learning from Examples

Each example demonstrates specific features:

| Example | Features Demonstrated |
|---------|----------------------|
| **conditional-build** | Conditional execution (`if_condition`, `skip_if`), workflow types |
| **custom-ci** | Light model tier, conservative timeouts, retry configuration |
| **quick-fix** | Minimal workflow, patch workflow type, rapid execution |

## Documentation

For detailed explanations and more examples, see:
- **[Workflow Examples Guide](../../../docs/Agent/workflow-examples.md)** - Complete guide with 6 examples
- **[Workflow Engine Guide](../../../docs/Agent/workflow-engine.md)** - Main workflow engine documentation
- **[Workflow JSON Schema](../../../docs/Agent/workflow-json-schema.md)** - Complete schema reference
- **[Workflow Conditionals](../../../docs/Agent/workflow-conditionals.md)** - Conditional syntax guide

## Best Practices

### Model Tier Selection
- **light:** Linting, commits, simple validation
- **base:** Implementation, planning, standard tasks
- **heavy:** Complex reasoning, architecture decisions

### Timeout Guidelines
- **120s:** Simple tasks (commits)
- **300s:** Linting, security scans
- **600s:** Implementation, planning
- **900s:** Testing, comprehensive operations

### When to Use Each Example
- **conditional-build:** Production workflows with different validation levels
- **custom-ci:** Automated validation pipelines
- **quick-fix:** Urgent patches requiring minimal overhead

## Contributing Examples

To add a new example:

1. Create JSON file in this directory
2. Follow existing naming pattern (kebab-case)
3. Include complete metadata (name, version, description, description_long)
4. Add clear descriptions for each step
5. Test thoroughly
6. Update this README with new example
7. Add entry to workflow-examples.md guide

## Validation

Validate example workflows:

```bash
# Validate JSON syntax and schema
python -c "from adw.workflows.engine.parser import load_workflow; \
load_workflow('.opencode/workflow/examples/conditional-build.json')"
```

## Support

For issues or questions:
- Check [Troubleshooting Guide](../../../docs/Agent/workflow-engine.md#troubleshooting)
- Review [Workflow Examples Guide](../../../docs/Agent/workflow-examples.md)
- See [Migration Guide](../../../docs/Agent/workflow-migration-guide.md) for Python → JSON conversion
