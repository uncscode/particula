from pathlib import Path


def test_find_control_byte():
    path = Path(
        "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
    )
    data = path.read_bytes()
    for idx, byte in enumerate(data):
        if byte < 32 and byte not in (9, 10, 13):
            print("control byte at", idx, byte)
            assert byte == 0
            return
    print("no invalid control bytes")
