"""Produce a portable execution inventory from verified publication bundles."""

import json

from docs_examples import ROOT, bundle, registry, result_state


def execution_inventory():
    """Validate and report provenance for every retained runnable example."""
    rows = []
    for key, config in registry().items():
        directory, manifest = bundle(key)
        if result_state(config, manifest) != "current":
            raise ValueError(f"{key}: successful current results required")
        required = (
            "figure_dpi",
            "elapsed_seconds",
            "package_source_sha256",
            "input_sha256",
            "revision",
            "executed_at",
            "device",
            "python",
            "versions",
        )
        if any(name not in manifest for name in required):
            raise ValueError(f"{key}: incomplete execution provenance")
        if manifest["figure_dpi"] != 300 or manifest["elapsed_seconds"] <= 0:
            raise ValueError(f"{key}: 300 DPI and positive duration required")
        if config["device"] == "cuda" and not manifest["device"].startswith(
            "cuda"
        ):
            raise ValueError(f"{key}: native CUDA execution required")
        if "source.py" not in manifest["files"]:
            raise ValueError(f"{key}: executed source missing from bundle")
        for outputs in manifest["outputs"]:
            for output in outputs:
                if (
                    "image" in output
                    and output["image"] not in manifest["files"]
                ):
                    raise ValueError(
                        f"{key}: undeclared figure {output['image']}"
                    )
        rows.append(
            {
                "example": key,
                "source": config["source"],
                "policy": config["policy"],
                "bundle": str(directory.relative_to(ROOT)),
                "state": "current",
                **{name: manifest[name] for name in required},
                "figures": sum(
                    "image" in item
                    for cell in manifest["outputs"]
                    for item in cell
                ),
                "output_items": sum(len(cell) for cell in manifest["outputs"]),
            }
        )
    return {"examples": rows, "successful_count": len(rows)}


if __name__ == "__main__":
    destination = ROOT / "website/.cache/execution-inventory.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(execution_inventory(), indent=2) + "\n")
    print(f"Wrote {destination.relative_to(ROOT)}")
