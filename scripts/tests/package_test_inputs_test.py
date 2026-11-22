"""Keep checkout-only imports out of package tests, including CUDA-only paths."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "particula"
CHECKOUT_MODULES = {"scripts", "docs", "examples_tests", "website"}


def test_package_tests_import_distributable_helpers():
    """Report every checkout import, even inside an unavailable device branch."""
    violations = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE)
        if not {"tests", "integration_tests"}.intersection(relative.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                modules = [node.module or ""]
            else:
                continue
            for name in modules:
                if name.split(".", 1)[0] in CHECKOUT_MODULES:
                    violations.append(
                        f"particula/{relative}:{node.lineno}: {name}"
                    )
    assert not violations, (
        "Checkout-only imports in package tests:\n" + "\n".join(violations)
    )
