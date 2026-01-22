from pathlib import Path

path = Path(
    "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
)
data = path.read_bytes()
idx = 6579
start = idx - 40
end = idx + 40
snippet = data[start:end]
print("len", len(data))
print("range", start, end)
for offset, byte in enumerate(snippet):
    pos = start + offset
    ch = chr(byte) if 32 <= byte <= 126 else repr(byte)
    print(pos, byte, ch)
