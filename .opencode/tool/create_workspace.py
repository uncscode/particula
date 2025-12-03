#!/usr/bin/env python3
"""Workspace Creator Tool

Creates isolated ADW workspace with all pre-LLM setup steps:
- Fetches GitHub issue
- Generates branch name
- Creates worktree
- Initializes ADW state

This tool performs all deterministic setup before any LLM/AI work begins.
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


def create_workspace_cli(
    issue_number: str,
    workflow_type: str,
    adw_id: str | None,
    triggered_by: str,
    output_mode: str,
) -> tuple[int, str]:
    """Create ADW workspace via CLI.

    Args:
        issue_number: GitHub issue number (e.g., "123")
        workflow_type: Workflow type ("complete", "patch", "document", "generate")
        adw_id: Optional existing ADW ID
        triggered_by: Trigger source ("manual", "cron", "webhook")
        output_mode: Output format ("summary", "full", "json")

    Returns:
        Tuple of (exit_code, output_string)
    """
    # Add project root to path so we can import adw
    project_root = find_project_root()
    sys.path.insert(0, str(project_root))

    try:
        from adw.workflows.operations.workspace import create_workspace

        # Call the workspace creation function
        result_adw_id, error = create_workspace(
            issue_number=issue_number,
            workflow_type=workflow_type,  # type: ignore
            adw_id=adw_id,
            triggered_by=triggered_by,
        )

        if error:
            # Workspace creation failed
            if output_mode == "json":
                output = json.dumps(
                    {
                        "success": False,
                        "error": error,
                        "adw_id": None,
                    },
                    indent=2,
                )
            else:
                lines = []
                lines.append("=" * 60)
                lines.append("WORKSPACE CREATION FAILED")
                lines.append("=" * 60)
                lines.append(f"\nIssue: #{issue_number}")
                lines.append(f"Workflow Type: {workflow_type}")
                if adw_id:
                    lines.append(f"ADW ID: {adw_id}")
                lines.append(f"\n❌ Error: {error}")
                lines.append("\n" + "=" * 60)
                output = "\n".join(lines)

            return 1, output

        # Workspace created successfully
        # Load state to get full details
        from adw.state.manager import ADWState
        from adw.utils.logging import get_logger

        logger = get_logger(result_adw_id, "workspace_tool")
        state = ADWState.load(result_adw_id, logger)

        if not state:
            return 1, f"ERROR: Failed to load state for ADW ID {result_adw_id}"

        # Format output based on mode
        if output_mode == "json":
            output = json.dumps(
                {
                    "success": True,
                    "adw_id": result_adw_id,
                    "issue_number": issue_number,
                    "workflow_type": workflow_type,
                    "branch_name": state.get("branch_name"),
                    "worktree_path": state.get("worktree_path"),
                    "parent_issue_number": state.get("parent_issue_number"),
                },
                indent=2,
            )
        elif output_mode == "full":
            # Full output includes complete state
            lines = []
            lines.append("=" * 60)
            lines.append("WORKSPACE CREATED SUCCESSFULLY")
            lines.append("=" * 60)
            lines.append(f"\n✅ ADW ID: {result_adw_id}")
            lines.append("\nWorkspace Details:")
            lines.append(f"  Issue: #{issue_number}")
            lines.append(f"  Workflow Type: {workflow_type}")
            lines.append(f"  Branch: {state.get('branch_name')}")
            lines.append(f"  Worktree: {state.get('worktree_path')}")
            if state.get("parent_issue_number"):
                lines.append(
                    f"  Parent Issue: #{state.get('parent_issue_number')}"
                )
            lines.append(f"  Triggered By: {triggered_by}")
            lines.append("\nNext Steps:")
            lines.append(
                "  1. Generate implementation plan using /plan or build_plan()"
            )
            lines.append(
                f"  2. The workspace is ready at: {state.get('worktree_path')}"
            )
            lines.append(
                f"  3. State file: agents/{result_adw_id}/adw_state.json"
            )
            lines.append("\n" + "=" * 60)
            lines.append("COMPLETE STATE")
            lines.append("=" * 60)
            lines.append(json.dumps(state.data, indent=2))
            lines.append("=" * 60)
            output = "\n".join(lines)
        else:  # summary
            lines = []
            lines.append("=" * 60)
            lines.append("WORKSPACE CREATED SUCCESSFULLY")
            lines.append("=" * 60)
            lines.append(f"\n✅ ADW ID: {result_adw_id}")
            lines.append("\nWorkspace Details:")
            lines.append(f"  Issue: #{issue_number}")
            lines.append(f"  Workflow Type: {workflow_type}")
            lines.append(f"  Branch: {state.get('branch_name')}")
            lines.append(f"  Worktree: {state.get('worktree_path')}")
            if state.get("parent_issue_number"):
                lines.append(
                    f"  Parent Issue: #{state.get('parent_issue_number')}"
                )
            lines.append("\nNext Steps:")
            lines.append(
                "  1. Generate implementation plan using /plan or build_plan()"
            )
            lines.append(
                f"  2. The workspace is ready at: {state.get('worktree_path')}"
            )
            lines.append("\n" + "=" * 60)
            output = "\n".join(lines)

        return 0, output

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
        description="Create isolated ADW workspace with pre-LLM setup"
    )

    # Required arguments
    parser.add_argument(
        "issue_number",
        type=str,
        help='GitHub issue number to create workspace for (e.g., "123")',
    )

    # Optional arguments
    parser.add_argument(
        "--workflow-type",
        type=str,
        choices=["complete", "patch", "document", "generate"],
        default="complete",
        help="Workflow type (default: complete)",
    )
    parser.add_argument(
        "--adw-id",
        type=str,
        help="Optional existing ADW ID (generates new one if not provided)",
    )
    parser.add_argument(
        "--triggered-by",
        type=str,
        default="manual",
        help='Who/what triggered this workflow (default: "manual")',
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["summary", "full", "json"],
        default="summary",
        help="Output mode: summary (default), full details, or JSON",
    )

    args = parser.parse_args()

    exit_code, output = create_workspace_cli(
        issue_number=args.issue_number,
        workflow_type=args.workflow_type,
        adw_id=args.adw_id,
        triggered_by=args.triggered_by,
        output_mode=args.output,
    )

    print(output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
