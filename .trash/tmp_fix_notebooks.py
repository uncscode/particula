import nbformat
from pathlib import Path


def test_rewrite_notebooks() -> None:
    root = Path(__file__).resolve().parent
    notebook_paths = [
        "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb",
        "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.ipynb",
    ]

    for relative_path in notebook_paths:
        path = root / relative_path
        nb = nbformat.read(path, as_version=nbformat.NO_CONVERT)
        nbformat.write(nb, path)
