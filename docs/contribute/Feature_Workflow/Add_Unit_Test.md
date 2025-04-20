# Add Unit Tests

## Philosophy

Every new public function or class must have at least one test for accuracy and one edge case or failure mode.

## Location

particula/<area>/tests/test_<symbol>.py  (mirror the package directory)

## Template for Functions

```python
"""Docstring for the test module."""
import numpy as np  # if needed
import pytest
from particula.<area> import <symbol>

def test_<symbol>_accuracy():
    """Docstring for the test."""
    result = <symbol>(<valid_args>)
    assert np.isclose(result, <expected>)

@pytest.mark.parametrize("bad_input", [...])

def test_<symbol>_bad_inputs(bad_input):
    """Docstring for the test."""
    with pytest.raises(ValueError):
        <symbol>(bad_input)
```

## Template for Classes

```python
"""Docstring for the test module."""
import numpy as np  # if needed
import pytest
from particula.<area> import <ClassName>

def test_<ClassName>_accuracy():
    """Docstring for the test."""
    obj = <ClassName>(<valid_args>)
    result = obj.<method_name>(<valid_args>)
    assert np.isclose(result, <expected>)


@pytest.mark.parametrize("bad_input", [...])
def test_<ClassName>_bad_inputs(bad_input):
    """Docstring for the test."""
    with pytest.raises(ValueError):
        <ClassName>(bad_input)
```

## Run locally

Execute `pytest -q` before opening the pull request; all tests must be green.
