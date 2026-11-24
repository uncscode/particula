"""Regression checks for installed-package provenance in the Jia replay."""

import json
from importlib.metadata import PathDistribution

import pytest

import study


@pytest.mark.parametrize(
    ("vcs_info", "expected"),
    [
        (None, None),
        ({"vcs": "git", "commit_id": "a" * 40}, "a" * 40),
    ],
)
def test_installed_revision(tmp_path, monkeypatch, vcs_info, expected):
    """Use the installed distribution's origin without walking to parent Git."""
    site = tmp_path / "site-packages"
    module = site / "particula" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.touch()
    info = site / "particula-0.3.6.dist-info"
    info.mkdir()
    if vcs_info:
        (info / "direct_url.json").write_text(
            json.dumps({"url": "file:///example", "vcs_info": vcs_info})
        )
    monkeypatch.setattr(study.par, "__file__", str(module))
    monkeypatch.setattr(
        study.metadata, "distribution", lambda name: PathDistribution(info)
    )

    def unexpected_git_call(*args, **kwargs):
        pytest.fail("A site-packages import queried Git")

    monkeypatch.setattr(study.subprocess, "check_output", unexpected_git_call)
    assert study.particula_revision() == expected
