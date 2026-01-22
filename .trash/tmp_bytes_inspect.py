from pathlib import Path

path = Path(
    "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
)
data = path.read_bytes()
idx = 6579
print(len(data))
print("byte at idx", idx, data[idx])
start = max(0, idx - 40)
end = min(len(data), idx + 40)
snippet = data[start:end]
print(snippet)
print("decoded:", snippet.decode("utf-8", errors="backslashreplace"))
for i, b in enumerate(snippet):
    pos = start + i
    if b < 32 and b not in (9, 10, 13):
        print("control", pos, b)
print("done snippet")
