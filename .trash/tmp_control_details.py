from pathlib import Path

path = Path(
    "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
)
data = path.read_bytes()
print("len", len(data))
for idx, byte in enumerate(data):
    if byte < 32 and byte not in (9, 10, 13):
        print("control", idx, byte)
        break
else:
    print("no control bytes")
