# Conditional Documentation Guide

This guide helps you determine which repository-specific documentation to read based on your current task. Only read documentation when it's directly relevant to avoid information overload.

## Instructions
- Review the task you've been asked to perform
- Check each documentation path in the Conditional Documentation section
- For each path, evaluate if any of the listed conditions apply to your task
  - IMPORTANT: Only read the documentation if any one of the conditions match your task
- IMPORTANT: Avoid excessive documentation reading. Only read what's relevant to your task.

## Conditional Documentation

### Core Repository Files

- readme.md
  - Conditions:
    - When first understanding the project structure and goals
    - When learning about installation and setup procedures
    - When understanding the core package architecture
    - When working on package configuration

- pyproject.toml
  - Conditions:
    - When adding or modifying package dependencies
    - When configuring development tools
    - When updating package metadata or version


### Repository-Specific Agent Documentation (docs/ai_docs/)

- docs/ai_docs/README.md
  - Conditions:
    - When first exploring the repository's AI documentation structure
    - When understanding which guides are available
    - When looking for an overview of development conventions

- docs/ai_docs/conditional_docs.md
  - Conditions:
    - When and overview of the code repo is needed.
    - Use when adding code to the core repo, or needing to write new code.
    - Use when first exploring the repository.

- docs/ai_docs/architecture_reference.md
  - Conditions:
    - When first exploring the architecture documentation structure
    - When looking for the right architecture document to read
    - When unsure which architectural resource to consult

- docs/ai_docs/architecture/architecture_guide.md
  - Conditions:
    - When designing new modules or major features
    - When understanding architectural principles and patterns
    - When learning about error handling, data flow, or testing architecture
    - When making significant architectural decisions
    - When reviewing code for architectural consistency
    - When the /architecture_review command is invoked

- docs/ai_docs/architecture/architecture_outline.md
  - Conditions:
    - When first exploring the codebase structure
    - When understanding component responsibilities and relationships
    - When finding where to add new features
    - When needing a quick overview of the system

- docs/ai_docs/architecture/decisions/
  - Conditions:
    - When understanding why architectural decisions were made
    - When creating a new Architecture Decision Record (ADR)
    - When making significant technology or design pattern choices
    - When reconsidering past decisions in new contexts
    - When documenting architectural changes

- docs/ai_docs/code_style.md
  - Conditions:
    - When writing new code and unsure about style conventions
    - When reviewing code for style consistency
    - When understanding naming conventions
    - When implementing type hints
    - When choosing design patterns
    - When onboarding new contributors

- docs/ai_docs/testing_guide.md
  - Conditions:
    - When creating new test files
    - When running tests and interpreting results
    - When understanding test organization and naming conventions
    - When configuring test frameworks
    - When troubleshooting test failures
    - When the /test or /resolve_failed_test commands are invoked

- docs/ai_docs/linting_guide.md
  - Conditions:
    - When running linting tools
    - When fixing linting errors or warnings
    - When understanding which linters are used and why
    - When configuring linting tools
    - When the /lint command is invoked

- docs/ai_docs/docstring_guide.md
  - Conditions:
    - When writing or updating function/class docstrings
    - When understanding the required docstring format
    - When adding examples to documentation
    - When reviewing code for documentation quality
    - When the /docstring, /docstring_function, or /docstring_class commands are invoked

- docs/ai_docs/documentation_guide.md
  - Conditions:
    - When creating feature or chore documentation files
    - When understanding documentation format requirements
    - When learning about file naming conventions
    - When handling screenshots or visual assets
    - When updating conditional documentation
    - When the /document command is invoked

- docs/ai_docs/review_guide.md
  - Conditions:
    - When conducting code reviews
    - When understanding review criteria and checklists
    - When classifying issue severity
    - When reviewing different types of code modules
    - When the /review command is invoked

- docs/ai_docs/commit_conventions.md
  - Conditions:
    - When creating git commits
    - When understanding commit message format requirements
    - When learning about issue class conventions
    - When implementing commit message validation
    - When the /commit command is invoked

- docs/ai_docs/pr_conventions.md
  - Conditions:
    - When creating pull requests
    - When understanding PR title and body format requirements
    - When learning about branch naming conventions
    - When linking issues to pull requests
    - When the /pull_request command is invoked

## Usage Tips

1. **Start with README**: Always read `readme.md` first when beginning work in a new repository
2. **Read relevant guides before commands**: When using slash commands (e.g., `/test`, `/lint`), read the corresponding guide first
3. **Consult guides during reviews**: When reviewing code, reference the style, testing, and architecture guides
4. **Update conditional docs**: When adding new repository-specific guides, update this file to include them
