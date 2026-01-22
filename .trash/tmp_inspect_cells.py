import nbformat
from pathlib import Path

path = Path(
    ".trash/docs/Examples/Simulations/Notebooks/Cloud_Chamber_Cycles.ipynb"
)
nb = nbformat.read(path, as_version=4)
for idx, cell in enumerate(nb.cells):
    src = cell.source
    if isinstance(src, list):
        text = "".join(src)
    else:
        text = src
    text = text.strip().splitlines()
    first_line = text[0] if text else "<empty>"
    print(idx, cell.cell_type, first_line)
