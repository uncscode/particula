"""Select changed notebooks as a JSON matrix without splitting file names."""

import json
import os
import subprocess
from pathlib import Path


def changed_notebooks(event: dict, event_name: str) -> list[str]:
    """Return existing changed notebook paths for a push or pull request."""
    if event_name == "pull_request":
        pull_request = event["pull_request"]
        revision = (
            f"{pull_request['base']['sha']}...{pull_request['head']['sha']}"
        )
        command = [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            revision,
            "--",
            "docs/**/*.ipynb",
        ]
    elif event["before"] == "0" * 40:
        command = ["git", "ls-files", "-z", "--", "docs/**/*.ipynb"]
    else:
        command = [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            event["before"],
            event["after"],
            "--",
            "docs/**/*.ipynb",
        ]
    result = subprocess.run(  # noqa: S603 -- Git argv, never a shell command
        command, check=True, capture_output=True, text=True
    )
    return sorted(path for path in result.stdout.split("\0") if path)


if __name__ == "__main__":
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    notebooks = changed_notebooks(event, os.environ["GITHUB_EVENT_NAME"])
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"notebooks={json.dumps(notebooks)}\n")
        output.write(f"has_changes={str(bool(notebooks)).lower()}\n")
    print(f"Selected {len(notebooks)} notebooks: {json.dumps(notebooks)}")
