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
from scripts import publish_chemistry_catalog as publisher

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
    (native_data / "VERSION").write_text("1.0.0")
    builder.build(out, native_data)
    original = (out / bundle.ASSET).read_bytes()
    builder.build(out, native_data)
    assert original == (out / bundle.ASSET).read_bytes()

    def download(url, timeout):
        if url.startswith(bundle.RELEASES_API):
            return io.BytesIO(
                b'[{"tag_name":"chemistry-catalog-v1.0.0","draft":false,"prerelease":false}]'
            )
        assert url.startswith(
            f"{bundle.DOWNLOAD_URL}/chemistry-catalog-v1.0.0/"
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
        main(["chem", "fetch", "--catalog-version", "1.0.0", "--json"])
    assert "fetch RECIPE" in json.loads(capsys.readouterr().err)["error"]


def test_release_job_publishes_both_catalog_assets():
    """Catalog publication is gated on its own version and tested build."""
    workflows = builder.NOTICES.parents[1] / ".github/workflows"
    config = yaml.safe_load((workflows / "chemistry-catalog.yml").read_text())
    triggers = config.get("on", config.get(True))
    assert triggers["push"] == {
        "branches": ["main"],
        "paths": ["chemistry_catalog/VERSION"],
    }
    assert config["jobs"]["publish"]["needs"] == "build"
    assert "build_chemistry_catalog" not in (workflows / "pypi.yml").read_text()


def test_latest_catalog_version(monkeypatch):
    """Ignore software, draft and prerelease tags and search all pages."""

    def item(tag, draft=False, pre=False):
        return dict(tag_name=tag, draft=draft, prerelease=pre)

    first = [item("v9.0.0")] * 97 + [
        item("chemistry-catalog-v1.9.0"),
        item("chemistry-catalog-v4.0.0", draft=True),
        item("chemistry-catalog-v3.0.0", pre=True),
    ]
    pages = iter([first, [item("chemistry-catalog-v1.10.0")]])
    monkeypatch.setattr(
        bundle,
        "urlopen",
        lambda url, timeout: io.BytesIO(json.dumps(next(pages)).encode()),
    )
    assert bundle.latest_version() == "1.10.0"


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
    snapshots = catalog.fetch(
        catalog_version=(builder.DATA / "VERSION").read_text().strip()
    )["snapshots"]
    if (builder.DATA / "VERSION").read_text().strip() == "1.0.0":
        assert sum(s["count"] for s in snapshots.values()) == 19655
        assert sum(s["native_count"] for s in snapshots.values()) == 2311
        assert sum(s["requires_closed_oxygen"] for s in snapshots.values()) == 7
    assert all(s["count"] == s["source_count"] for s in snapshots.values())
    for key, snapshot in snapshots.items():
        rows = catalog.search(snapshot=snapshot["id"])
        assert len(rows) == snapshot["count"]
        native_count = 0
        for line in (builder.DATA / f"{key}.jsonl").read_bytes().splitlines():
            item = json.loads(line)
            if item["mechanism"] is None:
                assert item["native_issue"]
                continue
            native_count += 1
            mechanism = ReactionMechanism.from_dict(item["mechanism"])
            assert len(mechanism.reactions) == 1
        assert native_count == snapshot["native_count"]
    soap_id = snapshots["pmcamx-soap"]["id"] + "/SOAP1"
    assert soap_id in {
        r["id"] for r in catalog.search(species="CAS3", role="product")
    }
    soap = catalog.show(soap_id)
    assert soap["metadata"]["yield_unit"] == "ppm/ppm"
    assert soap["expression"] and soap["native_reaction"] is None
    assert snapshots["mcm"]["RO2"]
    lock = tmp_path / "soap.lock.json"
    catalog.export([soap_id], lock, target="lock")
    assert (
        json.loads(lock.read_bytes())["reactions"][0]["metadata"]
        == soap["metadata"]
    )
    with pytest.raises(ValueError, match="NOx-dependent SOAP"):
        catalog.export([], tmp_path / "soap.json", lock_path=lock)
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


@pytest.mark.parametrize("exists", [False, True])
def test_independent_publisher(assets, tmp_path, monkeypatch, exists):
    """Use the catalog tag and preserve assets on retries."""
    from shutil import copytree

    release = publisher.release
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_REPOSITORY", release.REPOSITORY)
    version = tmp_path / "chemistry_catalog/VERSION"
    version.parent.mkdir()
    version.write_text("1.0.0")
    target = copytree(assets, tmp_path / "dist/catalog")
    calls = []

    def command(*args):
        calls.append(args)
        return "a" * 40

    monkeypatch.setattr(release, "command", command)
    monkeypatch.setattr(release, "ensure_tag", lambda *args: calls.append(args))
    existing = dict(
        draft=False,
        prerelease=False,
        assets=[
            dict(name=p.name, digest="sha256:" + release.digest(p))
            for p in target.iterdir()
        ],
    )
    monkeypatch.setattr(
        release, "fetch_json", lambda *a, **kw: existing if exists else None
    )
    publisher.publish()
    assert ("1.0.0", "a" * 40, "chemistry-catalog") in calls
    creates = [cmd for cmd in calls if cmd[:3] == ("gh", "release", "create")]
    if exists:
        assert not creates
        assert not any(cmd[:3] == ("gh", "release", "upload") for cmd in calls)
    else:
        assert creates[0][3] == "chemistry-catalog-v1.0.0"
        assert "--latest=false" in creates[0] and "--verify-tag" in creates[0]
    version.write_text("1.0.1")
    with pytest.raises(ValueError, match="version differs"):
        publisher.publish()
