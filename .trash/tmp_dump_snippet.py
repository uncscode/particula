from pathlib import Path

path = Path(
    "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
)
data = path.read_bytes()
idx = 6579
start = max(0, idx - 20)
end = min(len(data), idx + 20)
snippet = data[start:end]
Path("tmp_bytes.bin").write_bytes(snippet)
with open(
    "tmp_snippet.txt", "w", encoding="utf-8", errors="backslashreplace"
) as fh:
    fh.write(snippet.decode("utf-8", errors="backslashreplace"))
print("wrote snippet to tmp_snippet.txt")
