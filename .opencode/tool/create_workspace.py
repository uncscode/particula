#!/usr/bin/env python3
"""Workspace Creator Tool for ADW.

Creates an isolated ADW workspace with all pre-LLM setup steps:
    - Fetches GitHub issue details and metadata
    - Generates deterministic branch name based on issue content
    - Creates isolated git worktree under trees/{adw_id}/
    - Initializes ADW state with auto-populated fields

This tool performs all deterministic setup before any LLM/AI work begins.
No LLM calls are made. No GitHub status updates are posted.

Usage:
    python3 create_workspace.py 123
    python3 create_workspace.py 456 --workflow-type patch
    python3 create_workspace.py 789 --adw-id abc12345 --output json

Examples:
    # Create workspace for issue #123 with default complete workflow
    python3 .opencode/tool/create_workspace.py 123

    # Create workspace for quick patch (skips test/review/docs)
    python3 .opencode/tool/create_workspace.py 456 --workflow-type patch

    # Resume existing workspace with known ADW ID
    python3 .opencode/tool/create_workspace.py 789 --adw-id abc12345

    # Get structured JSON output for programmatic use
    python3 .opencode/tool/create_workspace.py 101 --output json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


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


def create_workspace_cli(
    issue_number: str,
    workflow_type: str,
    adw_id: str | None,
    triggered_by: str,
    output_mode: str,
) -> tuple[int, str]:
    """Create ADW workspace via CLI interface.

    Orchestrates the complete workspace creation flow including:
    1. Fetching GitHub issue details
    2. Generating deterministic branch name
    3. Creating isolated git worktree
    4. Initializing ADW state file

    Args:
        issue_number: GitHub issue number as string (e.g., "123").
        workflow_type: Workflow type to initialize. One of:
            - "complete": Full validation cycle (test, review, docs)
            - "patch": Quick fixes (skips test/review/docs)
            - "document": Documentation-only workflow
            - "generate": Code generation workflow
        adw_id: Optional existing ADW ID (8-char hex like 'abc12345').
            If not provided, generates a new unique ID.
        triggered_by: Source that triggered this workflow. Examples:
            - "manual": User-initiated via CLI
            - "cron": Scheduled automation
            - "webhook": GitHub webhook event
        output_mode: Output format for results. One of:
            - "summary": Human-readable with key details (default)
            - "full": Complete state dump with all fields
            - "json": Structured JSON for programmatic use

    Returns:
        Tuple of (exit_code, output_string) where exit_code is 0 on success
        and 1 on failure. The output_string contains formatted results or
        error details depending on the output_mode.

    Raises:
        ImportError: If adw module cannot be imported (not in project root).
        Exception: For unexpected errors during workspace creation.
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
            lines.append(f"\nWorkspace Details:")
            lines.append(f"  Issue: #{issue_number}")
            lines.append(f"  Workflow Type: {workflow_type}")
            lines.append(f"  Branch: {state.get('branch_name')}")
            lines.append(f"  Worktree: {state.get('worktree_path')}")
            if state.get("parent_issue_number"):
                lines.append(f"  Parent Issue: #{state.get('parent_issue_number')}")
            lines.append(f"  Triggered By: {triggered_by}")
            lines.append(f"\nNext Steps:")
            lines.append(f"  1. Generate implementation plan using /plan or build_plan()")
            lines.append(f"  2. The workspace is ready at: {state.get('worktree_path')}")
            lines.append(f"  3. State file: agents/{result_adw_id}/adw_state.json")
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
            lines.append(f"\nWorkspace Details:")
            lines.append(f"  Issue: #{issue_number}")
            lines.append(f"  Workflow Type: {workflow_type}")
            lines.append(f"  Branch: {state.get('branch_name')}")
            lines.append(f"  Worktree: {state.get('worktree_path')}")
            if state.get("parent_issue_number"):
                lines.append(f"  Parent Issue: #{state.get('parent_issue_number')}")
            lines.append(f"\nNext Steps:")
            lines.append(f"  1. Generate implementation plan using /plan or build_plan()")
            lines.append(f"  2. The workspace is ready at: {state.get('worktree_path')}")
            lines.append("\n" + "=" * 60)
            output = "\n".join(lines)

        return 0, output

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

    Parses command-line arguments and executes workspace creation.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Create isolated ADW workspace with pre-LLM setup",
        epilog="""
Examples:
  %(prog)s 123                          Create workspace for issue #123
  %(prog)s 456 --workflow-type patch    Create quick patch workspace
  %(prog)s 789 --adw-id abc12345        Resume existing workspace
  %(prog)s 101 --output json            Get JSON output for scripting
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="""Workflow type: complete (default, full validation), patch (quick fixes),
                document (docs only), generate (code generation)""",
    )
    parser.add_argument(
        "--adw-id",
        type=str,
        help="Existing ADW ID to resume (8-char hex like 'abc12345'). Generates new if omitted.",
    )
    parser.add_argument(
        "--triggered-by",
        type=str,
        default="manual",
        help="Trigger source: manual (default), cron, webhook, or custom identifier",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["summary", "full", "json"],
        default="summary",
        help="Output mode: summary (default, key details), full (complete state), json (structured)",
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
