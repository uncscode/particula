import json
from pathlib import Path

for relative in [
    "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb",
    "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.ipynb",
]:
    path = Path(relative)
    try:
        with path.open("r", encoding="utf-8") as fh:
            json.load(fh)
    except Exception as exc:
        print(relative, "load failed", type(exc).__name__, exc)
    else:
        print(relative, "valid")
