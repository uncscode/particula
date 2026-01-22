from pathlib import Path


def test_dump_json():
    path = Path(
        "docs/Examples/Simulations/Notebooks/Cloud_Chamber_Single_Cycle.ipynb"
    )
    data = path.read_bytes()
    idx = 6579
    start = idx - 20
    end = idx + 20
    print("byte", data[idx])
    snippet = data[start:end]
    print(snippet)
    print(snippet.decode("utf-8", errors="backslashreplace"))
    assert True
