#!/usr/bin/env python3
"""Workflow Builder Tool

Provides access to WorkflowBuilderTool for creating and validating ADW workflow JSON files.
Enables interactive workflow creation with incremental validation.
"""

import argparse
import json
import sys
from pathlib import Path


def find_project_root() -> Path:
    """Find project root by looking for pyproject.toml or .git."""
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

    Args:
        command: Operation to perform (create, add_step, remove_step, get, list, update, validate)
        workflow_name: Name of workflow (required for most commands)
        description: Workflow description (for create)
        version: Workflow version (for create, default: "1.0.0")
        workflow_type: Type of workflow (for create: complete, patch, custom)
        step_json: JSON string of step to add (for add_step)
        step_index: Index of step to remove (for remove_step)
        step_name: Name of step to remove (for remove_step)
        position: Position to insert step (for add_step, None = append)
        workflow_json: Complete workflow JSON (for update, validate)
        output_mode: Output format (summary, full, json)

    Returns:
        Tuple of (exit_code, output_string)
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
                return (
                    1,
                    "ERROR: 'create' requires workflow_name and description",
                )

            success, message = builder.create_workflow(
                workflow_name, description, version, workflow_type
            )

            if output_mode == "json":
                output = json.dumps(
                    {
                        "success": success,
                        "message": message,
                        "workflow_name": workflow_name,
                    },
                    indent=2,
                )
            else:
                output = f"{'✅' if success else '❌'} {message}"

            return 0 if success else 1, output

        elif command == "add_step":
            if not workflow_name or not step_json:
                return (
                    1,
                    "ERROR: 'add_step' requires workflow_name and step_json",
                )

            success, message = builder.add_step(
                workflow_name, step_json, position
            )

            if output_mode == "json":
                output = json.dumps(
                    {
                        "success": success,
                        "message": message,
                        "workflow_name": workflow_name,
                    },
                    indent=2,
                )
            else:
                output = f"{'✅' if success else '❌'} {message}"

            return 0 if success else 1, output

        elif command == "remove_step":
            if not workflow_name:
                return 1, "ERROR: 'remove_step' requires workflow_name"
            if step_index is None and step_name is None:
                return (
                    1,
                    "ERROR: 'remove_step' requires either step_index or step_name",
                )

            success, message = builder.remove_step(
                workflow_name, step_index, step_name
            )

            if output_mode == "json":
                output = json.dumps(
                    {
                        "success": success,
                        "message": message,
                        "workflow_name": workflow_name,
                    },
                    indent=2,
                )
            else:
                output = f"{'✅' if success else '❌'} {message}"

            return 0 if success else 1, output

        elif command == "get":
            if not workflow_name:
                return 1, "ERROR: 'get' requires workflow_name"

            success, message, workflow_data = builder.get_workflow(
                workflow_name
            )

            if not success or workflow_data is None:
                if output_mode == "json":
                    output = json.dumps(
                        {
                            "success": False,
                            "message": message,
                            "workflow": None,
                        },
                        indent=2,
                    )
                else:
                    output = f"❌ {message}"
                return 1, output

            if output_mode == "json":
                output = json.dumps(
                    {
                        "success": True,
                        "message": message,
                        "workflow": workflow_data,
                    },
                    indent=2,
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
                lines.append(
                    f"Description: {workflow_data.get('description', 'N/A')}"
                )
                lines.append(
                    f"Type: {workflow_data.get('workflow_type', 'N/A')}"
                )
                lines.append(f"Steps: {len(workflow_data.get('steps', []))}")
                step_names = [
                    s.get("name", "unnamed")
                    for s in workflow_data.get("steps", [])
                ]
                for i, name in enumerate(step_names, 1):
                    lines.append(f"  {i}. {name}")
                output = "\n".join(lines)

            return 0, output

        elif command == "list":
            workflows = builder.list_workflows()

            if output_mode == "json":
                output = json.dumps(
                    {"workflows": workflows, "count": len(workflows)}, indent=2
                )
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
                            lines.append(
                                f"  • {wf_name} ({wf_type}) - {step_count} steps - {desc}"
                            )
                        else:
                            lines.append(f"  • {wf_name} (unable to load)")
                    output = "\n".join(lines)

            return 0, output

        elif command == "update":
            if not workflow_name or not workflow_json:
                return (
                    1,
                    "ERROR: 'update' requires workflow_name and workflow_json",
                )

            success, message = builder.update_workflow(
                workflow_name, workflow_json
            )

            if output_mode == "json":
                output = json.dumps(
                    {
                        "success": success,
                        "message": message,
                        "workflow_name": workflow_name,
                    },
                    indent=2,
                )
            else:
                output = f"{'✅' if success else '❌'} {message}"

            return 0 if success else 1, output

        elif command == "validate":
            if not workflow_json:
                return 1, "ERROR: 'validate' requires workflow_json"

            success, error_msg, parsed_data = (
                builder.validate_workflow_json_str(workflow_json)
            )

            if output_mode == "json":
                output = json.dumps(
                    {"valid": success, "error": error_msg, "data": parsed_data},
                    indent=2,
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
        error_msg = f"Failed to import adw module: {e}\nMake sure you're running from the project root."
        if output_mode == "json":
            output = json.dumps(
                {"success": False, "error": error_msg}, indent=2
            )
        else:
            output = f"ERROR: {error_msg}"
        return 1, output

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        if output_mode == "json":
            output = json.dumps(
                {"success": False, "error": error_msg}, indent=2
            )
        else:
            output = f"ERROR: {error_msg}"
        return 1, output


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Workflow builder tool for creating and validating ADW workflows"
    )

    # Command argument
    parser.add_argument(
        "command",
        type=str,
        choices=[
            "create",
            "add_step",
            "remove_step",
            "get",
            "list",
            "update",
            "validate",
        ],
        help="Command to execute",
    )

    # Workflow name
    parser.add_argument("--workflow-name", type=str, help="Workflow name")

    # Create command arguments
    parser.add_argument(
        "--description", type=str, help="Workflow description (for create)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Workflow version (default: 1.0.0)",
    )
    parser.add_argument(
        "--workflow-type",
        type=str,
        default="custom",
        choices=["complete", "patch", "custom"],
        help="Workflow type (default: custom)",
    )

    # Step arguments
    parser.add_argument(
        "--step-json", type=str, help="Step JSON string (for add_step)"
    )
    parser.add_argument(
        "--step-index", type=int, help="Step index (for remove_step)"
    )
    parser.add_argument(
        "--step-name", type=str, help="Step name (for remove_step)"
    )
    parser.add_argument(
        "--position", type=int, help="Position to insert step (for add_step)"
    )

    # Update/validate arguments
    parser.add_argument(
        "--workflow-json", type=str, help="Complete workflow JSON (for update)"
    )

    # Output mode
    parser.add_argument(
        "--output",
        type=str,
        choices=["summary", "full", "json"],
        default="summary",
        help="Output mode (default: summary)",
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
