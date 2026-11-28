"""Exercise the release producer and public fetch without external services."""

import io
import json
import zipfile
from hashlib import sha256
from urllib.error import HTTPError

import pytest
import yaml
from particula.dynamics.chemistry.catalog import (
    ChemistryCatalog,
    bundle,
    sources,
)
from particula.dynamics.chemistry.catalog.cli import main
from scripts import build_chemistry_catalog as builder

REACTION = b"Reactions\n[t] NO + O3 -> NO2 + O2 ; 3e-12,-1500\nEnd Reactions\n"


@pytest.fixture
def assets(tmp_path, monkeypatch):
    """Build an actual release archive from one synthetic E3SM source."""
    recipe = {
        "source": "e3sm",
        "release": "fixture",
        "mechanism": "test",
        "aerosol": None,
        "files": [
            {
                "name": "chem.in",
                "url": "https://example.org/chem.in",
                "sha256": sha256(REACTION).hexdigest(),
            }
        ],
    }
    monkeypatch.setattr(builder, "CITATIONS", {"e3sm": "fixture"})
    monkeypatch.setattr(sources, "recipes", lambda: {"e3sm": recipe})
    monkeypatch.setattr(
        sources, "urlopen", lambda url, timeout: io.BytesIO(REACTION)
    )
    out = tmp_path / "release"
    builder.build(out)
    original = (out / bundle.ASSET).read_bytes()
    builder.build(out)
    assert original == (out / bundle.ASSET).read_bytes()
    monkeypatch.setattr(
        sources, "urlopen", lambda *a, **kw: pytest.fail("unexpected import")
    )

    def download(url, timeout):
        assert url.startswith(
            f"https://github.com/uncscode/particula/releases/download/{bundle.RELEASE}/"
        )
        return (out / url.rsplit("/", 1)[1]).open("rb")

    monkeypatch.setattr(bundle, "urlopen", download)
    monkeypatch.setenv("PARTICULA_CHEM_CATALOG", str(tmp_path / "catalog.db"))
    return out


def test_default_fetch_and_offline_export(
    assets, tmp_path, capsys, monkeypatch
):
    """Install twice alongside a custom study, then search and export offline."""
    catalog = ChemistryCatalog()
    source = tmp_path / "case.in"
    source.write_bytes(REACTION)
    local = catalog.import_file("waccm", source, release="local", mechanism="x")
    main(["chem", "fetch", "--json"])
    manifest = json.loads(capsys.readouterr().out)
    assert catalog.fetch() == manifest
    assert catalog.show(local)["count"] == 1
    assert catalog.source_bytes(sha256(REACTION).hexdigest()) == REACTION
    monkeypatch.setattr(
        bundle, "urlopen", lambda *a, **kw: pytest.fail("offline")
    )
    main(["chem", "search", "--source", "e3sm", "--species", "NO", "--json"])
    (row,) = json.loads(capsys.readouterr().out)
    assert row["id"] == manifest["snapshots"]["e3sm"]["id"] + "/t"
    native = tmp_path / "native.json"
    main(["chem", "export", row["id"], "--output", str(native)])
    assert json.loads(native.read_bytes())
    with zipfile.ZipFile(assets / bundle.ASSET) as archive:
        assert (
            archive.read("notices/e3sm.txt")
            == (builder.NOTICES / "e3sm.txt").read_bytes()
        )


@pytest.mark.parametrize("damage", ["checksum", "count", "release", "archive"])
def test_damaged_bundle_preserves_local_catalog(assets, tmp_path, damage):
    """Reject invalid downloads before committing any imported rows."""
    archive = assets / bundle.ASSET
    if damage in {"count", "release"}:
        with zipfile.ZipFile(archive) as src:
            entries = {name: src.read(name) for name in src.namelist()}
        manifest = json.loads(entries["manifest.json"])
        if damage == "count":
            manifest["snapshots"]["e3sm"]["count"] += 1
        else:
            manifest["release"] = "v0"
        entries["manifest.json"] = json.dumps(manifest).encode()
        with zipfile.ZipFile(archive, "w") as dest:
            for name, data in entries.items():
                dest.writestr(name, data)
    else:
        archive.write_bytes(b"incomplete")
    if damage != "checksum":
        (assets / (bundle.ASSET + ".sha256")).write_text(
            f"{sha256(archive.read_bytes()).hexdigest()}  {bundle.ASSET}\n"
        )
    source = tmp_path / "case.in"
    source.write_bytes(REACTION)
    c = ChemistryCatalog()
    local = c.import_file("waccm", source, release="mine", mechanism="trial")
    with pytest.raises(ValueError):
        c.fetch()
    assert [row["id"] for row in c.search()] == [local + "/t"]


def test_missing_release(tmp_path, monkeypatch, capsys):
    """Explain how to fetch a named source on a release without a bundle."""

    def missing(url, timeout):
        raise HTTPError(url, 404, "Not found", {}, io.BytesIO())

    monkeypatch.setattr(bundle, "urlopen", missing)
    monkeypatch.setenv("PARTICULA_CHEM_CATALOG", str(tmp_path / "catalog.db"))
    with pytest.raises(SystemExit, match="2"):
        main(["chem", "fetch", "--json"])
    assert "fetch RECIPE" in json.loads(capsys.readouterr().err)["error"]


def test_release_job_publishes_both_catalog_assets():
    """Require the builder and its checked output in the base release path."""
    workflow = builder.NOTICES.parents[1] / ".github/workflows/pypi.yml"
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    steps = jobs["pyping"]["steps"]
    build_step = next(
        step
        for step in steps
        if "scripts.build_chemistry_catalog" in step.get("run", "")
    )
    assert build_step["if"] == "needs.release-plan.outputs.publish == 'true'"
    assert (
        f"sha256sum --strict --check {bundle.ASSET}.sha256" in build_step["run"]
    )
    assert "sha256sum catalog/* >> SHA256SUMS" in build_step["run"]
    publish = "\n".join(
        step.get("run", "") for step in jobs["publish"]["steps"]
    )
    assert (
        f"cp candidates/catalog/{bundle.ASSET}{{,.sha256}} release-assets/"
        in publish
    )
