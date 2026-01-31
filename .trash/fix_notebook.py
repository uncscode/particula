#!/usr/bin/env python3
"""Fix notebook by removing inconsistent cell IDs."""

import json
import sys

notebook_path = "/home/kyle/Code/particula/trees/b047e282/docs/Examples/Simulations/Notebooks/Soot_Formation_in_Flames.ipynb"

# Read the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Track changes
cells_modified = 0
cells_total = len(notebook.get("cells", []))

# Remove 'id' field from all cells
for cell in notebook.get("cells", []):
    if "id" in cell:
        del cell["id"]
        cells_modified += 1

# Write back
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
    f.write("\n")  # Ensure trailing newline

print(f"Fixed notebook: {notebook_path}")
print(f"Total cells: {cells_total}")
print(f"Cells with 'id' removed: {cells_modified}")
print("SUCCESS")
