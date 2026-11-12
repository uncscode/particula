"""Regression coverage for removal-safe native execution guards."""

import importlib
import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("loaded", [False, True])
def test_guard_blocks_missing_imports_and_cached_constructors(
    monkeypatch, request, loaded
):
    """Guard both fresh imports and aliases captured before module removal."""
    module_name = "particula._legacy_aerosol"

    class Facade:
        pass

    if loaded:
        module = ModuleType(module_name)
        module.Aerosol = Facade
        monkeypatch.setitem(sys.modules, module_name, module)
    else:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    request.getfixturevalue("forbid_legacy_facades")

    with pytest.raises(ModuleNotFoundError) as caught:
        importlib.import_module(module_name)
    assert caught.value.name == module_name
    if loaded:
        with pytest.raises(pytest.fail.Exception, match="legacy facade"):
            Facade()
