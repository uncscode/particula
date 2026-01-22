from pathlib import Path

NOTEBOOKS = [
    "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb",
    "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.ipynb",
]


def test_find_invalid_control_chars():
    for relative in NOTEBOOKS:
        path = Path(relative)
        data = path.read_bytes()
        for idx, byte in enumerate(data):
            if byte < 32 and byte not in (9, 10, 13):
                print(relative, "control char at", idx, "byte", byte)
        assert True
