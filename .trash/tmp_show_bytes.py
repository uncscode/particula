from pathlib import Path

path = Path(
    "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
)
data = path.read_bytes()
idx = 6579
print("byte", data[idx])
start = idx - 20
end = idx + 20
print(data[start:end])
print(data[start:end].decode("utf-8", errors="backslashreplace"))
