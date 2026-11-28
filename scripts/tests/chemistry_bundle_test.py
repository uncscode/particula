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
from particula.dynamics.chemistry.catalog import (
    native as converter,
)
from particula.dynamics.chemistry.catalog.cli import main
from particula.dynamics.chemistry.mechanism import ReactionMechanism
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
    native_data = tmp_path / "native-data"
    builder.refresh(ChemistryCatalog(tmp_path / "inputs.db"), native_data)
    monkeypatch.setattr(
        sources, "urlopen", lambda *a, **kw: pytest.fail("unexpected import")
    )
    builder.build(out, native_data)
    original = (out / bundle.ASSET).read_bytes()
    builder.build(out, native_data)
    assert original == (out / bundle.ASSET).read_bytes()

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
    source.write_bytes(REACTION + b"\n")
    local = catalog.import_file("waccm", source, release="local", mechanism="x")
    main(["chem", "fetch", "--json"])
    manifest = json.loads(capsys.readouterr().out)
    assert catalog.fetch() == manifest
    snap = manifest["snapshots"]["e3sm"]
    assert catalog.show(snap["id"]) == snap
    assert catalog.show(local)["count"] == 1
    assert catalog.source_bytes(sha256(REACTION).hexdigest()) is None
    monkeypatch.setattr(
        converter, "compile_channel", lambda *a, **kw: pytest.fail("recompiled")
    )
    monkeypatch.setattr(
        bundle, "urlopen", lambda *a, **kw: pytest.fail("offline")
    )
    main(["chem", "search", "--source", "e3sm", "--species", "NO", "--json"])
    (row,) = json.loads(capsys.readouterr().out)
    assert row["id"] == manifest["snapshots"]["e3sm"]["id"] + "/t"
    native = tmp_path / "native.json"
    main(["chem", "export", row["id"], "--output", str(native)])
    lock = tmp_path / "native.lock.json"
    catalog.export([row["id"]], lock, target="lock")
    replay = tmp_path / "replay.json"
    ChemistryCatalog(tmp_path / "empty").export([], replay, lock_path=lock)
    assert replay.read_bytes() == native.read_bytes()
    expected = catalog.show(row["id"])["mechanism"]
    assert ReactionMechanism.load(native).to_dict() == (
        ReactionMechanism.from_dict(expected).to_dict()
    )
    with zipfile.ZipFile(assets / bundle.ASSET) as archive:
        assert not any(
            n.endswith((".sqlite3", ".in")) for n in archive.namelist()
        )
        assert REACTION not in archive.read("e3sm.jsonl")
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


def test_committed_native_inventory(tmp_path, monkeypatch):
    """Validate all tracked mechanisms and both reservoirs for MCM titration."""

    def offline(*args, **kwargs):
        pytest.fail("network or source converter called")

    monkeypatch.setattr(sources, "urlopen", offline)
    monkeypatch.setattr(converter, "compile_channel", offline)
    builder.build(tmp_path)
    monkeypatch.setattr(
        bundle,
        "urlopen",
        lambda url, timeout: (tmp_path / url.rsplit("/", 1)[1]).open("rb"),
    )
    catalog = ChemistryCatalog(tmp_path / "catalog.db")
    snapshots = catalog.fetch()["snapshots"]
    assert sum(s["source_count"] for s in snapshots.values()) == 19655
    assert sum(s["count"] for s in snapshots.values()) == 2311
    assert sum(s["requires_closed_oxygen"] for s in snapshots.values()) == 7
    for key, snapshot in snapshots.items():
        rows = catalog.search(snapshot=snapshot["id"])
        assert len(rows) == snapshot["count"]
        for line in (builder.DATA / f"{key}.jsonl").read_bytes().splitlines():
            item = json.loads(line)
            mechanism = ReactionMechanism.from_dict(item["mechanism"])
            assert len(mechanism.reactions) == 1
    path = tmp_path / "ozone.json"
    for key, label in (("waccm", "NO_O3"), ("mcm", "7"), ("mcm-full", "7")):
        rid = snapshots[key]["id"] + "/" + label
        if key.startswith("mcm"):
            with pytest.raises(ValueError, match="close-oxygen"):
                catalog.export([rid], path)
        catalog.export([rid], path, close_oxygen=key.startswith("mcm"))
        rxn = ReactionMechanism.load(path).to_dict()["reactions"][0]
        assert rxn["reactants"] == {"gas:NO": 1, "gas:O3": 1}
        assert rxn["products"] == {"gas:NO2": 1, "gas:O2": 1}


def test_source_and_native_snapshot_coexist(assets, monkeypatch):
    """Keep both pinned selections and return one match in ordinary searches."""
    catalog = ChemistryCatalog()
    snapshot = catalog.fetch()["snapshots"]["e3sm"]
    monkeypatch.setattr(
        sources, "urlopen", lambda *a, **kw: io.BytesIO(REACTION)
    )
    source_id = catalog.fetch("e3sm")
    assert source_id == snapshot["source_snapshot"]
    assert [r["id"] for r in catalog.search()] == [source_id + "/t"]
    (native_row,) = catalog.search(snapshot=snapshot["id"])
    assert native_row["id"] == snapshot["id"] + "/t"
    assert (
        catalog.show(native_row["id"])["native_reaction"]["id"]
        == source_id + "/t"
    )
