from pathlib import Path


def test_find_control_char_single_cycle():
    path = Path(
        "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
    )
    data = path.read_bytes()
    for idx, byte in enumerate(data):
        if byte < 32 and byte not in (9, 10, 13):
            print("single-cycle: index", idx, "byte", byte)
            break
    else:
        print("single-cycle: no control char")
    assert True
