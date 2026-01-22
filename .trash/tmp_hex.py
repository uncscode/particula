from pathlib import Path

path = Path(
    "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
)
data = path.read_bytes()
idx = 6579
start = max(0, idx - 20)
end = min(len(data), idx + 20)
snippet = data[start:end]
print("range", start, end)
print("repr", snippet)
print("list", list(snippet))
