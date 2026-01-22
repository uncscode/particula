from pathlib import Path

path = Path(
    "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
)
text = path.read_text()
needle = '    .set_wall_eddy_diffusivity(0.001, "m^2/s")\n'
print("found", needle in text)
