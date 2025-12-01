# OpenCode Advanced Features

## Overview

This guide documents advanced OpenCode CLI features available through ADW's OpenCode integration. These features enable enhanced workflow performance, richer context provision to agents, and advanced session management.

## File Attachments (--file)

### Purpose
Attach context files to agent prompts for direct file reference, enabling agents to access file contents without embedding them in prompts.

### Use Cases
- **Implementation Specs**: Attach implementation specifications for `/implement` command
- **Architecture Docs**: Attach architecture documentation for `/document` command
- **Configuration Files**: Attach configuration files for `/review` command
- **Context Provision**: Provide structured context files for better agent understanding

### Usage

```python
from adw.core.opencode import execute_opencode_agent

# Attach single file
response = execute_opencode_agent(
    prompt="Review this implementation against the spec",
    adw_id="abc123",
    agent_name="reviewer",
    attached_files=["spec.md"],
)

# Attach multiple files
response = execute_opencode_agent(
    prompt="Implement the feature described in the spec with architecture guidance",
    adw_id="abc123", 
    agent_name="implementor",
    attached_files=["feature_spec.md", "architecture_guide.md", "config.json"],
)
```

### Benefits
- **Reduced Prompt Size**: Avoid embedding large files directly in prompts
- **Improved Context Quality**: Provide structured file references
- **Direct File Access**: OpenCode can read file contents directly
- **Better Organization**: Separate concerns between prompt instructions and reference materials

## Custom Commands (--command)

### Purpose
Execute custom OpenCode commands defined in `.opencode/command/` directory instead of using default agent behavior.

### Use Cases
- **Specialized Workflows**: Use repository-specific commands for unique processes
- **Command Chaining**: Execute specific command sequences
- **Custom Analysis**: Run custom analysis tools through OpenCode

### Usage

```python
# Execute custom command
response = execute_opencode_agent(
    prompt="Run the analysis",
    adw_id="abc123",
    agent_name="build",
    custom_command="analyze",  # Must exist in .opencode/command/analyze.md
)

# Custom testing command
response = execute_opencode_agent(
    prompt="Run comprehensive tests",
    adw_id="abc123",
    agent_name="tester",
    custom_command="test_comprehensive",
)
```

### Command Validation
- Commands are validated against available files in `.opencode/command/`
- Helpful error messages list available commands if validation fails
- Commands must be valid `.md` files (excluding `README.md`)

## Server Mode (--attach)

### Purpose
Connect to a running OpenCode server to reduce cold boot time and improve performance for long-running workflows.

### Performance Benefits
- **Cold Boot Time Savings**: 5-10 seconds per execution
- **MCP Server Reuse**: Reuses existing Model Context Protocol server connections
- **Optimal for Long Workflows**: Best for workflows with multiple agent invocations

### Setup

1. **Start OpenCode Server**:
   ```bash
   opencode serve --port 9107
   ```

2. **Connect to Server**:
   ```python
   response = execute_opencode_agent(
       prompt="Implement the feature",
       adw_id="abc123",
       agent_name="implementor",
       attach_to_server="http://localhost:9107",
   )
   ```

### Usage Examples

```python
# Basic server connection
response = execute_opencode_agent(
    prompt="Review code",
    adw_id="abc123",
    agent_name="reviewer",
    attach_to_server="http://localhost:9107",
)

# Server with custom port
response = execute_opencode_agent(
    prompt="Generate documentation",
    adw_id="abc123",
    agent_name="documenter", 
    attach_to_server="http://localhost:8080",
)

# HTTPS server connection
response = execute_opencode_agent(
    prompt="Remote execution",
    adw_id="abc123",
    agent_name="build",
    attach_to_server="https://opencode-server.example.com",
)
```

## Session Continuation (--session)

### Purpose
Continue existing OpenCode sessions for multi-turn conversations and maintain context across multiple agent invocations.

### Use Cases
- **Multi-Turn Workflows**: Continue conversations across multiple steps
- **Context Preservation**: Maintain conversation history
- **Iterative Development**: Build upon previous interactions

### Usage

```python
# Start new session (session ID will be returned)
response = execute_opencode_agent(
    prompt="Start implementing the feature",
    adw_id="abc123",
    agent_name="implementor",
)
session_id = response.session_id

# Continue the session
response = execute_opencode_agent(
    prompt="Now add error handling to the implementation",
    adw_id="abc123",
    agent_name="implementor",
    session_id=session_id,
)
```

## Session Sharing (--share)

### Purpose
Enable session sharing for collaboration between team members or different workflow components.

### Use Cases
- **Team Collaboration**: Share sessions between developers
- **Workflow Handoffs**: Pass sessions between different workflow phases
- **Debugging**: Share problematic sessions for investigation

### Usage

```python
# Create shared session
response = execute_opencode_agent(
    prompt="Analyze this complex issue",
    adw_id="abc123",
    agent_name="analyzer",
    share_session=True,
    title="Complex Bug Analysis Session",
)
```

## Session Titles (--title)

### Purpose
Set descriptive titles for OpenCode sessions to improve organization and identification.

### Use Cases
- **Session Organization**: Provide meaningful names for sessions
- **Workflow Tracking**: Identify sessions by purpose
- **Debugging**: Quickly identify relevant sessions

### Usage

```python
response = execute_opencode_agent(
    prompt="Implement OAuth authentication",
    adw_id="abc123",
    agent_name="implementor",
    title="OAuth Implementation - User Authentication Module",
)
```

## Custom Ports (--port)

### Purpose
Use custom server ports instead of OpenCode defaults when connecting to servers.

### Use Cases
- **Port Conflicts**: Avoid conflicts with other services
- **Custom Deployments**: Connect to servers on non-standard ports
- **Load Balancing**: Connect to specific server instances

### Usage

```python
response = execute_opencode_agent(
    prompt="Execute on custom port",
    adw_id="abc123",
    agent_name="build",
    server_port=8080,  # Must be in range 1024-65535
)
```

### Port Requirements
- **Valid Range**: 1024-65535 (non-privileged ports)
- **Availability**: Port must not be in use by another process
- **Validation**: Invalid ports raise descriptive errors

## Combining Advanced Features

You can combine multiple advanced features for complex workflows:

```python
# Complex workflow with all advanced features
response = execute_opencode_agent(
    prompt="Implement the feature with full context and server optimization",
    adw_id="abc123",
    agent_name="implementor",
    model_tier="heavy",  # Use Opus for complex task
    attached_files=["feature_spec.md", "architecture.md", "api_docs.md"],
    custom_command="implement_with_tests",
    attach_to_server="http://localhost:9107", 
    session_id="continuing_session_123",
    share_session=True,
    title="Feature Implementation with Full Context",
    server_port=9107,
    working_dir="/path/to/worktree",
)
```

## Security Considerations

### File Path Validation
- All attached files are validated for existence before execution
- Files must be accessible from the current working directory
- **Future Enhancement**: Restriction to project directory only

### URL Validation
- Server URLs must start with `http://` or `https://`
- Invalid URLs are rejected with helpful error messages
- Prevents command injection through malformed URLs

### Port Range Restrictions
- Ports restricted to non-privileged range (1024-65535)
- Prevents security issues with system ports
- Invalid ports rejected with valid range information

### Command Validation
- Custom commands validated against available commands in `.opencode/command/`
- Prevents execution of non-existent or malicious commands
- Helpful error messages list available commands

## Error Handling

### Validation Errors
All advanced features include comprehensive validation with helpful error messages:

```python
# File not found
ValueError: Attached file not found: /path/to/missing/file.md

# Invalid server URL
ValueError: Invalid server URL: localhost:9107 (must start with http:// or https://)

# Invalid port
ValueError: Invalid port: 80 (must be 1024-65535)

# Command not found
ValueError: Command 'nonexistent' not found. Available: test, lint, implement, review
```

### Best Practices
- Always validate inputs before execution
- Handle errors gracefully in calling code  
- Use try-catch blocks around agent execution
- Log errors for debugging purposes

## Performance Guidelines

### When to Use File Attachments
- **Large Context Files**: When files are larger than ~1KB
- **Structured Data**: Configuration files, specifications, API documentation
- **Multiple References**: When referencing multiple files simultaneously

### When to Use Server Mode
- **Long-Running Workflows**: Workflows with 3+ agent invocations
- **Performance Critical**: When execution speed is important
- **Resource Optimization**: To reduce server startup overhead

### Optimal Feature Combinations
- **File Attachments + Server Mode**: Best for complex implementations with large context
- **Custom Commands + Session Continuation**: Ideal for multi-step custom workflows  
- **Session Sharing + Titles**: Perfect for collaborative development workflows

## Migration from Basic Usage

### Before (Basic Usage)
```python
response = execute_opencode_agent("Implement feature", "abc123")
```

### After (Advanced Usage)
```python
response = execute_opencode_agent(
    prompt="Implement feature according to specification",
    adw_id="abc123", 
    agent_name="implementor",
    model_tier="heavy",
    attached_files=["feature_spec.md"],
    attach_to_server="http://localhost:9107",
    title="Feature Implementation Session",
)
```

All advanced features are **backward compatible** - existing code continues to work without modification.