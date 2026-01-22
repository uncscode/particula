import copy
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / ".trash/docs/Examples/Simulations/Notebooks/Cloud_Chamber_Cycles.ipynb"
)
SINGLE_DEST = (
    ROOT
    / "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
)
MULTI_DEST = (
    ROOT / "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.ipynb"
)

SINGLE_INTRO = nbformat.v4.new_markdown_cell(
    """# Cloud Chamber Single Cycle (split from Cloud_Chamber_Cycles.ipynb)

This notebook covers **sections 1–8** of the original combined tutorial.
It focuses on one activation–deactivation cycle that fits within CI time constraints.
For multi-cycle comparisons, see the companion [Cloud_Chamber_Multi_Cycle.ipynb](Cloud_Chamber_Multi_Cycle.ipynb).

**Learning objectives:**
- Configure chamber geometry and wall-loss settings.
- Define hygroscopic seeds with kappa-theory.
- Build a particle-resolved aerosol with speciated mass.
- Run a single activation and deactivation cycle.
- Visualize droplet growth/shrinkage and verify mass conservation.
"""
)

MULTI_INTRO = nbformat.v4.new_markdown_cell(
    """# Cloud Chamber Multi Cycle (split from Cloud_Chamber_Cycles.ipynb)

This notebook covers **sections 9+** of the original combined tutorial.
It runs four activation–deactivation cycles across multiple seed scenarios.
Start with the single-cycle foundation in [Cloud_Chamber_Single_Cycle.ipynb](Cloud_Chamber_Single_Cycle.ipynb).

**Learning objectives:**
- Reuse the single-cycle setup for multi-cycle runs.
- Compare hygroscopicity across ammonium sulfate, sucrose, and mixed seeds.
- Analyze activation timing, water uptake, and mass retention over cycles.
"""
)


def clear_outputs(cells: list[nbformat.NotebookNode]) -> None:
    for cell in cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        cell.metadata.pop("execution", None)


def test_split_notebooks() -> None:
    nb = nbformat.read(SOURCE, as_version=4)
    metadata = copy.deepcopy(nb.metadata)
    cells = nb.cells

    multi_index = None
    for idx, cell in enumerate(cells):
        if cell.cell_type == "markdown":
            source = (
                cell.source
                if isinstance(cell.source, str)
                else "".join(cell.source)
            )
            if "# Part 2:" in source:
                multi_index = idx
                break

    assert multi_index is not None, "Could not find multi-cycle partition"

    single_cells = [copy.deepcopy(cell) for cell in cells[:multi_index]]
    multi_cells = [copy.deepcopy(cell) for cell in cells[multi_index:]]

    clear_outputs(single_cells)
    clear_outputs(multi_cells)

    single_nb = nbformat.v4.new_notebook()
    single_nb.cells = [SINGLE_INTRO, *single_cells]
    single_nb.metadata = metadata

    multi_nb = nbformat.v4.new_notebook()
    multi_nb.cells = [MULTI_INTRO, *multi_cells]
    multi_nb.metadata = metadata

    nbformat.write(single_nb, SINGLE_DEST)
    nbformat.write(multi_nb, MULTI_DEST)
