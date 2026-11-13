# Container Isolation for OpenCode Agents

**Version:** 2.1.0
**Last Updated:** 2025-12-07

## Overview

Container isolation provides defense-in-depth security for running OpenCode agents. While OpenCode includes built-in permission controls, container isolation adds an additional OS-level security layer that restricts filesystem access, network connectivity, and system capabilities.

Use container isolation when you need:

- **Enterprise security compliance**: Additional isolation for regulated environments
- **CI/CD integration**: Sandboxed agent execution in automated pipelines
- **Air-gapped networks**: Complete network isolation for sensitive operations
- **Untrusted code review**: Extra protection when agents analyze unknown code

> **Note:** `ADW_PROJECT_ROOT` is required and recognized by OpenCode permission allowlists. Set it to the project root (e.g., `/workspace` in containers) before running agents so filesystem rules resolve correctly.

## Quick Reference

| Method | Platform | Network Isolation | Setup Complexity | Best For |
|--------|----------|-------------------|------------------|----------|
| Docker | All | Yes (`--network=none`) | Low | Cross-platform, CI/CD |
| Bubblewrap | Linux | Yes (`--unshare-net`) | Medium | Linux servers, minimal overhead |
| Firejail | Linux | Yes (`net none`) | Medium | Desktop Linux, profiles |

## Docker Isolation

### Prerequisites

- Docker Engine installed
- User added to `docker` group (or use `sudo`)

### Dockerfile

```dockerfile
FROM python:3.12-slim

# Install only required dev tools
RUN pip install uv && \
    uv pip install --system ruff mypy pytest

# Create non-root user for security
RUN useradd -m -s /bin/bash agent
USER agent

# Mount project directory
WORKDIR /workspace
VOLUME /workspace

# Set environment variable for portable paths
ENV ADW_PROJECT_ROOT=/workspace

# No secrets in image - pass at runtime
# Run with: docker run --network=none -v $(pwd):/workspace opencode-sandbox
```

### Running the Container

```bash
# Build the sandbox image
docker build -t opencode-sandbox .

# Run with network isolation
docker run --network=none -v $(pwd):/workspace opencode-sandbox opencode

# Run interactively
docker run -it --network=none -v $(pwd):/workspace opencode-sandbox /bin/bash
```

### Docker Compose Example

```yaml
# docker-compose.yml
version: '3.8'
services:
  opencode:
    build: .
    network_mode: none
    volumes:
      - .:/workspace
    environment:
      - ADW_PROJECT_ROOT=/workspace
```

## Bubblewrap Isolation

### Prerequisites

- Linux operating system
- bubblewrap package installed (`apt install bubblewrap` or `dnf install bubblewrap`)

### Command-Line Usage

```bash
# Set project root for portable paths
export ADW_PROJECT_ROOT="$HOME/Code/Agent"

bwrap \
  --unshare-net \                        # Network isolation
  --ro-bind /usr /usr \                  # Read-only system binaries
  --ro-bind /lib /lib \                  # Read-only libraries
  --ro-bind /lib64 /lib64 \              # Read-only 64-bit libraries
  --ro-bind /bin /bin \                  # Read-only binaries
  --bind "$ADW_PROJECT_ROOT" /workspace \ # Writable project directory
  --bind /tmp /tmp \                     # Writable temp
  --proc /proc \                         # Process filesystem
  --dev /dev \                           # Device filesystem
  --setenv ADW_PROJECT_ROOT /workspace \ # Set env inside sandbox
  --chdir /workspace \                   # Working directory
  -- opencode                            # Command to run
```

### Flag Explanations

| Flag | Purpose | Security Impact |
|------|---------|-----------------|
| `--unshare-net` | Creates isolated network namespace | Prevents network access |
| `--ro-bind` | Read-only filesystem mount | Prevents system modification |
| `--bind` | Writable filesystem mount | Allows project file changes |
| `--proc /proc` | Mount proc filesystem | Required for process info |
| `--dev /dev` | Mount device filesystem | Required for device access |
| `--setenv` | Set environment variable | Configures sandbox environment |
| `--chdir` | Set working directory | Convenience for commands |

### Example Script

```bash
#!/bin/bash
# run-opencode-sandboxed.sh

export ADW_PROJECT_ROOT="${ADW_PROJECT_ROOT:-$(pwd)}"

exec bwrap \
  --unshare-net \
  --ro-bind /usr /usr \
  --ro-bind /lib /lib \
  --ro-bind /lib64 /lib64 \
  --ro-bind /bin /bin \
  --bind "$ADW_PROJECT_ROOT" /workspace \
  --bind /tmp /tmp \
  --proc /proc \
  --dev /dev \
  --setenv ADW_PROJECT_ROOT /workspace \
  --chdir /workspace \
  -- "$@"
```

## Firejail Isolation

### Prerequisites

- Linux operating system
- firejail package installed (`apt install firejail` or `dnf install firejail`)

### Profile File

Create `~/.config/firejail/opencode.profile`:

```ini
# ~/.config/firejail/opencode.profile
include /etc/firejail/default.profile

# Network isolation
net none

# Filesystem restrictions - whitelist approach
whitelist ${HOME}/Code/Agent
read-only ${HOME}/.config/opencode
noexec /tmp

# Disable dangerous capabilities
caps.drop all
nonewprivs
noroot
```

### Usage

```bash
# Run with the custom profile
firejail --profile=opencode opencode

# Override network isolation for specific commands
firejail --profile=opencode --net=none opencode
```

## Network Isolation Options

Each isolation method provides network isolation:

| Method | Flag | Effect |
|--------|------|--------|
| Docker | `--network=none` | No network interfaces available |
| Bubblewrap | `--unshare-net` | Isolated network namespace |
| Firejail | `net none` | Network disabled in profile |

**When to use network isolation:**

- Air-gapped code review
- Preventing data exfiltration
- Blocking unintended API calls
- Security compliance requirements

## Recommendations

### When to Use Container Isolation

1. **Enterprise/Compliance**: Required for SOC 2, HIPAA, or similar compliance
2. **CI/CD Pipelines**: Standard practice for automated workflows
3. **Untrusted Code**: When reviewing code from unknown sources
4. **Multi-tenant**: Isolating different projects or users

### Method Comparison

| Consideration | Docker | Bubblewrap | Firejail |
|--------------|--------|------------|----------|
| Cross-platform | Yes | No (Linux) | No (Linux) |
| Setup complexity | Low | Medium | Medium |
| Overhead | Medium | Very low | Low |
| Container images | Yes | No | No |
| Profile-based | No | No | Yes |

### Platform Considerations

- **macOS/Windows**: Use Docker (only cross-platform option)
- **Linux servers**: Consider bubblewrap for minimal overhead
- **Linux desktops**: Firejail profiles offer convenience

### Performance Implications

- **Docker**: ~5-10% overhead from container runtime
- **Bubblewrap**: Negligible overhead (namespace isolation only)
- **Firejail**: Low overhead (seccomp + namespace isolation)

For performance-critical workloads on Linux, bubblewrap offers the lowest overhead while still providing filesystem and network isolation.

## See Also

- [Backend Configuration](../backend_configuration.md) - Security best practices for tokens and credentials
- [Architecture Reference](../architecture_reference.md) - System architecture overview
