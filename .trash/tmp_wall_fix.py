from pathlib import Path

path = Path(
    "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
)
text = path.read_text()
old = '.set_wall_eddy_diffusivity(0.001, "m^2/s")'
new = '.set_wall_eddy_diffusivity(0.001, "1/s")'
if old not in text:
    raise SystemExit("target string not found")
path.write_text(text.replace(old, new, 1))
