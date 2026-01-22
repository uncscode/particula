from pathlib import Path


def test_inspect_bytes():
    path = Path(
        "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
    )
    data = path.read_bytes()
    idx = 6579
    start = max(0, idx - 40)
    end = min(len(data), idx + 40)
    snippet = data[start:end]
    print("len", len(data))
    print("range", start, end)
    for offset, byte in enumerate(snippet):
        pos = start + offset
        if byte < 32 and byte not in (9, 10, 13):
            print("control", pos, byte)
        print(pos, byte, repr(chr(byte)) if 32 <= byte <= 126 else byte)
    assert True
