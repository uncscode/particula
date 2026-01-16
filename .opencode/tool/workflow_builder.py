#!/usr/bin/env python3
"""Workflow Builder Tool for ADW.

Provides access to WorkflowBuilderTool for creating and validating ADW workflow
JSON files. Enables interactive workflow creation with incremental validation.

This tool wraps adw/workflows/engine/builder.py and is used by the
workflow-builder agent (.opencode/agent/workflow-builder.md) to provide
interactive workflow creation with the /create-workflow slash command.

Commands:
    create      Create new workflow file with initial structure
    add_step    Add validated step to existing workflow
    remove_step Remove step by index or name
    get         Retrieve workflow details
    list        List all available workflows
    update      Update entire workflow with validated JSON
    validate    Validate workflow JSON without saving

Workflow Types:
    complete    Full validation: plan → build → test → review → docs → ship
    patch       Quick fix: plan → build → ship (skips validation)
    custom      User-defined steps

Usage:
    python3 workflow_builder.py list
    python3 workflow_builder.py get --workflow-name patch
    python3 workflow_builder.py create --workflow-name my-flow --description "..."

Examples:
    # List all available workflows
    python3 .opencode/tool/workflow_builder.py list

    # Get workflow details
    python3 .opencode/tool/workflow_builder.py get --workflow-name patch

    # Create new workflow
    python3 .opencode/tool/workflow_builder.py create \\
        --workflow-name quick-deploy \\
        --description "Quick deployment workflow" \\
        --workflow-type custom

    # Add step to workflow
    python3 .opencode/tool/workflow_builder.py add_step \\
        --workflow-name quick-deploy \\
        --step-json '{"type":"agent","name":"Build","command":"/implement"}'

    # Validate workflow JSON
    python3 .opencode/tool/workflow_builder.py validate \\
        --workflow-json '{"name":"test","version":"1.0.0",...}'

See Also:
    .opencode/tool/workflow_builder.md - Complete documentation
    .opencode/agent/workflow-builder.md - Interactive builder agent
    .opencode/command/create-workflow.md - /create-workflow command
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def find_project_root() -> Path:
    """Find project root by looking for pyproject.toml or .git.

    Traverses up the directory tree from the current working directory
    until a pyproject.toml or .git directory is found.

    Returns:
        Path to the project root directory, or current working directory
        if no project markers are found.
    """
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent
    return Path.cwd()


def workflow_builder_tool(
    command: str,
    workflow_name: str | None = None,
    description: str | None = None,
    version: str = "1.0.0",
    workflow_type: str = "custom",
    step_json: str | None = None,
    step_index: int | None = None,
    step_name: str | None = None,
    position: int | None = None,
    workflow_json: str | None = None,
    output_mode: str = "summary",
) -> tuple[int, str]:
    """Execute workflow builder commands.

    Dispatches to the appropriate WorkflowBuilderTool method based on the
    command parameter. Handles validation and error reporting.

    Args:
        command: Operation to perform. One of:
            - "create": Create new workflow file (needs workflow_name, description)
            - "add_step": Add step to workflow (needs workflow_name, step_json)
            - "remove_step": Remove step (needs workflow_name, step_index or step_name)
            - "get": Retrieve workflow details (needs workflow_name)
            - "list": List all available workflows (no args required)
            - "update": Update entire workflow (needs workflow_name, workflow_json)
            - "validate": Validate JSON without saving (needs workflow_json)
        workflow_name: Name of workflow (used as filename without .json).
            Required for: create, add_step, remove_step, get, update.
        description: Short workflow description. Required for create command.
        version: Semantic version string (default: "1.0.0"). Used for create.
        workflow_type: Workflow type for create. One of:
            - "complete": Full validation (test, review, docs)
            - "patch": Quick fix (build and ship only)
            - "custom": User-defined steps (default)
        step_json: JSON string of step to add. Must be valid workflow step
            schema. Required for add_step command.
        step_index: Zero-based index of step to remove. For remove_step.
        step_name: Name of step to remove. For remove_step. Use this OR
            step_index, not both.
        position: Index position to insert step (for add_step). None appends.
        workflow_json: Complete workflow JSON string. Required for update
            and validate commands.
        output_mode: Output format for results. One of:
            - "summary": Human-readable with key details (default)
            - "full": Complete workflow JSON dump
            - "json": Structured JSON for programmatic use

    Returns:
        Tuple of (exit_code, output_string) where exit_code is 0 on success
        and 1 on failure. Output includes status indicators (✅/❌).

    Raises:
        ImportError: If adw module cannot be imported (not in project root).
        Exception: For unexpected errors during command execution.
    """
    # Add project root to path
    project_root = find_project_root()
    sys.path.insert(0, str(project_root))

    try:
        from adw.workflows.engine.builder import WorkflowBuilderTool

        builder = WorkflowBuilderTool()

        # Execute command
        if command == "create":
            if not workflow_name or not description:
                return 1, "ERROR: 'create' requires workflow_name and description"

            success, message = builder.create_workflow(
                workflow_name, description, version, workflow_type
            )

            if output_mode == "json":
                output = json.dumps(
                    {"success": success, "message": message, "workflow_name": workflow_name},
                    indent=2,
                )
            else:
                output = f"{'✅' if success else '❌'} {message}"

            return 0 if success else 1, output

        elif command == "add_step":
            if not workflow_name or not step_json:
                return 1, "ERROR: 'add_step' requires workflow_name and step_json"

            success, message = builder.add_step(workflow_name, step_json, position)

            if output_mode == "json":
                output = json.dumps(
                    {"success": success, "message": message, "workflow_name": workflow_name},
                    indent=2,
                )
            else:
                output = f"{'✅' if success else '❌'} {message}"

            return 0 if success else 1, output

        elif command == "remove_step":
            if not workflow_name:
                return 1, "ERROR: 'remove_step' requires workflow_name"
            if step_index is None and step_name is None:
                return 1, "ERROR: 'remove_step' requires either step_index or step_name"

            success, message = builder.remove_step(workflow_name, step_index, step_name)

            if output_mode == "json":
                output = json.dumps(
                    {"success": success, "message": message, "workflow_name": workflow_name},
                    indent=2,
                )
            else:
                output = f"{'✅' if success else '❌'} {message}"

            return 0 if success else 1, output

        elif command == "get":
            if not workflow_name:
                return 1, "ERROR: 'get' requires workflow_name"

            success, message, workflow_data = builder.get_workflow(workflow_name)

            if not success or workflow_data is None:
                if output_mode == "json":
                    output = json.dumps(
                        {"success": False, "message": message, "workflow": None}, indent=2
                    )
                else:
                    output = f"❌ {message}"
                return 1, output

            if output_mode == "json":
                output = json.dumps(
                    {"success": True, "message": message, "workflow": workflow_data}, indent=2
                )
            elif output_mode == "full":
                lines = []
                lines.append("=" * 60)
                lines.append(f"WORKFLOW: {workflow_name}")
                lines.append("=" * 60)
                lines.append(json.dumps(workflow_data, indent=2))
                lines.append("=" * 60)
                output = "\n".join(lines)
            else:  # summary
                lines = []
                lines.append(f"✅ Workflow: {workflow_name}")
                lines.append(f"Description: {workflow_data.get('description', 'N/A')}")
                lines.append(f"Type: {workflow_data.get('workflow_type', 'N/A')}")
                lines.append(f"Steps: {len(workflow_data.get('steps', []))}")
                step_names = [s.get("name", "unnamed") for s in workflow_data.get("steps", [])]
                for i, name in enumerate(step_names, 1):
                    lines.append(f"  {i}. {name}")
                output = "\n".join(lines)

            return 0, output

        elif command == "list":
            workflows = builder.list_workflows()

            if output_mode == "json":
                output = json.dumps({"workflows": workflows, "count": len(workflows)}, indent=2)
            else:
                if not workflows:
                    output = "No workflows found"
                else:
                    lines = []
                    lines.append(f"Available Workflows ({len(workflows)}):")
                    lines.append("")
                    for wf_name in sorted(workflows):
                        success, _, wf_data = builder.get_workflow(wf_name)
                        if success and wf_data:
                            desc = wf_data.get("description", "No description")
                            wf_type = wf_data.get("workflow_type", "unknown")
                            step_count = len(wf_data.get("steps", []))
                            lines.append(f"  • {wf_name} ({wf_type}) - {step_count} steps - {desc}")
                        else:
                            lines.append(f"  • {wf_name} (unable to load)")
                    output = "\n".join(lines)

            return 0, output

        elif command == "update":
            if not workflow_name or not workflow_json:
                return 1, "ERROR: 'update' requires workflow_name and workflow_json"

            success, message = builder.update_workflow(workflow_name, workflow_json)

            if output_mode == "json":
                output = json.dumps(
                    {"success": success, "message": message, "workflow_name": workflow_name},
                    indent=2,
                )
            else:
                output = f"{'✅' if success else '❌'} {message}"

            return 0 if success else 1, output

        elif command == "validate":
            if not workflow_json:
                return 1, "ERROR: 'validate' requires workflow_json"

            success, error_msg, parsed_data = builder.validate_workflow_json_str(workflow_json)

            if output_mode == "json":
                output = json.dumps(
                    {"valid": success, "error": error_msg, "data": parsed_data}, indent=2
                )
            else:
                if success:
                    output = f"✅ Workflow JSON is valid\n\nParsed workflow:\n{json.dumps(parsed_data, indent=2)}"
                else:
                    output = f"❌ Validation failed:\n{error_msg}"

            return 0 if success else 1, output

        else:
            return (
                1,
                f"ERROR: Unknown command '{command}'. Valid commands: create, add_step, remove_step, get, list, update, validate",
            )

    except ImportError as e:
        error_msg = (
            f"Failed to import adw module: {e}\nMake sure you're running from the project root."
        )
        if output_mode == "json":
            output = json.dumps({"success": False, "error": error_msg}, indent=2)
        else:
            output = f"ERROR: {error_msg}"
        return 1, output

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        if output_mode == "json":
            output = json.dumps({"success": False, "error": error_msg}, indent=2)
        else:
            output = f"ERROR: {error_msg}"
        return 1, output


def main() -> int:
    """Main entry point for CLI usage.

    Parses command-line arguments and executes workflow builder commands.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Workflow builder tool for creating and validating ADW workflows",
        epilog="""
Commands:
  create      Create new workflow file (requires --workflow-name, --description)
  add_step    Add step to workflow (requires --workflow-name, --step-json)
  remove_step Remove step (requires --workflow-name, --step-index or --step-name)
  get         Get workflow details (requires --workflow-name)
  list        List all available workflows
  update      Update workflow (requires --workflow-name, --workflow-json)
  validate    Validate JSON without saving (requires --workflow-json)

Examples:
  %(prog)s list
  %(prog)s get --workflow-name patch
  %(prog)s create --workflow-name my-flow --description "My workflow"
  %(prog)s add_step --workflow-name my-flow --step-json '{"type":"agent",...}'
  %(prog)s validate --workflow-json '{"name":"test",...}'
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Command argument
    parser.add_argument(
        "command",
        type=str,
        choices=["create", "add_step", "remove_step", "get", "list", "update", "validate"],
        help="Command to execute (see examples below)",
    )

    # Workflow name
    parser.add_argument(
        "--workflow-name",
        type=str,
        help="Workflow name (used as filename without .json extension)",
    )

    # Create command arguments
    parser.add_argument(
        "--description",
        type=str,
        help="Workflow description (required for create)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Semantic version (default: 1.0.0)",
    )
    parser.add_argument(
        "--workflow-type",
        type=str,
        default="custom",
        choices=["complete", "patch", "custom"],
        help="Workflow type: complete (full validation), patch (quick), custom (default)",
    )

    # Step arguments
    parser.add_argument(
        "--step-json",
        type=str,
        help="Step JSON string conforming to workflow step schema (for add_step)",
    )
    parser.add_argument(
        "--step-index",
        type=int,
        help="Zero-based step index (for remove_step)",
    )
    parser.add_argument(
        "--step-name",
        type=str,
        help="Step name to remove (for remove_step, alternative to --step-index)",
    )
    parser.add_argument(
        "--position",
        type=int,
        help="Position to insert step (for add_step, omit to append)",
    )

    # Update/validate arguments
    parser.add_argument(
        "--workflow-json",
        type=str,
        help="Complete workflow JSON string (for update, validate)",
    )

    # Output mode
    parser.add_argument(
        "--output",
        type=str,
        choices=["summary", "full", "json"],
        default="summary",
        help="Output mode: summary (default), full (complete JSON), json (structured)",
    )

    args = parser.parse_args()

    exit_code, output = workflow_builder_tool(
        command=args.command,
        workflow_name=args.workflow_name,
        description=args.description,
        version=args.version,
        workflow_type=args.workflow_type,
        step_json=args.step_json,
        step_index=args.step_index,
        step_name=args.step_name,
        position=args.position,
        workflow_json=args.workflow_json,
        output_mode=args.output,
    )

    print(output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
