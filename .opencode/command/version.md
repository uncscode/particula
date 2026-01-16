---
description: "Get ADW Version"
---

# Get ADW Version

Display the current version of the ADW (AI Developer Workflow) package.

## Purpose

Quickly check the current version of the ADW system installed in this repository.

## Instructions

1. **Read the VERSION file**
   - Use the get_version tool to read the VERSION file
   - Display the version in a clear format

2. **Show version information**
   - Display the version number
   - Optionally show where the VERSION file is located
   - Optionally show the last modified date of the VERSION file

## Output Format

```
ADW Version: 2.1.0
```

## Implementation

Use the `.claude/tools/get_version.py` tool or directly read the VERSION file at the project root.

## Notes

- The VERSION file is located at the project root
- Version follows semantic versioning (MAJOR.MINOR.PATCH)
- Version changes are documented in CHANGELOG.md
