"""Compare external feedstock test inputs/commands with our tested recipe.

Read recipe data only: never render Jinja or execute external commands. The
current recipes have literal top-level test mappings. Unsupported templating,
selectors, anchors, or schemas fail rather than silently certifying parity.
Source PRs can report valid drift with ``--allow-drift`` while awaiting the
release archive and external recipe handoff. Default release checks are strict.
Requires PyYAML (installed by conda-build or the CI contract-check job).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import yaml


def _test_mapping_text(recipe: str) -> str:
    """Extract a literal top-level test block without evaluating templates."""
    # An enclosing condition can remove the test block even when its closing
    # directive follows another top-level section. Only standalone set
    # directives are supported; do not render remote recipe control flow.
    directives = re.findall(r"{%[-+]?\s*(\w+)", recipe)
    if "{#" in recipe or any(directive != "set" for directive in directives):
        raise ValueError("Templated test contracts require explicit review")
    lines = recipe.splitlines()
    starts = [
        i for i, line in enumerate(lines) if re.fullmatch(r"test:\s*", line)
    ]
    if len(starts) != 1:
        raise ValueError("Expected one literal top-level test: mapping")
    block = []
    for line in lines[starts[0] + 1 :]:
        if line.startswith(("{%", "{{", "{#")):
            raise ValueError("Templated test contracts require explicit review")
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        block.append(line)
    text = "\n".join(block)
    # Keep the common requirement as a literal value for comparison. It is
    # valid YAML already; no rendering or sentinel substitution is needed.
    remaining = text.replace("python {{ python_min }}", "")
    if any(token in remaining for token in ("{{", "{%", "{#")):
        raise ValueError("Templated test contracts require explicit review")
    if re.search(r"#\s*\[", text):
        raise ValueError("Conditional test selectors require explicit review")
    return text


def test_contract(recipe: str) -> dict[str, list[str]]:
    """Validate and normalize literal input lists and ordered test commands."""
    text = _test_mapping_text(recipe)
    if any(
        isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken))
        for token in yaml.scan(text)
    ):
        raise ValueError("Anchored test contracts require explicit review")
    node = yaml.compose(text)
    if isinstance(node, yaml.nodes.MappingNode):
        keys = [key.value for key, _value in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate test keys require explicit review")
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or set(data) != {
        "requires",
        "source_files",
        "commands",
    }:
        raise ValueError("Expected requires, source_files, and commands only")
    for key, values in data.items():
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str) or not value.strip()
                for value in values
            )
        ):
            raise ValueError(f"test.{key} must be a nonempty string list")
    return {
        key: values if key == "commands" else sorted(values)
        for key, values in data.items()
    }


def check_contract(
    mirror: Path,
    external: Path,
    artifacts: Path,
    *,
    allow_drift: bool = False,
) -> bool:
    """Save parity evidence and return whether the selected policy passes.

    ``passed`` in the report always means contract parity. ``check_passed``
    and the return value permit valid differences only in advisory mode;
    unreadable or unsupported test contracts fail under either policy.
    """
    artifacts.mkdir(parents=True, exist_ok=True)
    report = {}
    contracts = {}
    errors = []
    for name, path in (("mirror", mirror), ("external", external)):
        try:
            payload = path.read_bytes()
            (artifacts / f"{name}-meta.yaml").write_bytes(payload)
            report[name] = {
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            contracts[name] = test_contract(payload.decode("utf-8"))
        except (OSError, ValueError, yaml.YAMLError) as error:
            errors.append(f"{name}: {error}")
    differences = []
    if not errors:
        differences = [
            f"test.{key}"
            for key in contracts["mirror"]
            if contracts["mirror"][key] != contracts["external"][key]
        ]
    report.update(contracts=contracts, differences=differences, errors=errors)
    report["passed"] = not errors and not differences
    report["mode"] = "advisory" if allow_drift else "strict"
    report["check_passed"] = not errors and (report["passed"] or allow_drift)
    output = json.dumps(report, indent=2) + "\n"
    (artifacts / "contract.json").write_text(output, encoding="utf-8")
    print(output, end="")
    if differences and allow_drift and not errors:
        prefix = (
            "::warning::"
            if os.environ.get("GITHUB_ACTIONS") == "true"
            else "WARNING: "
        )
        print(
            prefix
            + "External feedstock test contract differs: "
            + ", ".join(differences)
            + ". Source PR drift is advisory; external release readiness "
            "still requires the handoff, a strict check, and a feedstock build."
        )
    elif not report["passed"]:
        print(
            "Feedstock test contract mismatch/unavailable. Apply the handoff "
            "in conda/README.md and rerun; the local mirror alone is not "
            "external feedstock validation."
        )
    return report["check_passed"]


def main() -> int:
    """Compare contracts, enforcing parity unless valid drift is permitted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-recipe", type=Path, required=True)
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help=(
            "Report valid differences without blocking a source PR. "
            "Missing or unsupported test contracts still fail."
        ),
    )
    parser.add_argument(
        "--mirror", type=Path, default=Path("conda/recipe/meta.yaml")
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(".artifacts/feedstock-contract"),
    )
    args = parser.parse_args()
    return int(
        not check_contract(
            args.mirror,
            args.external_recipe,
            args.artifacts,
            allow_drift=args.allow_drift,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
