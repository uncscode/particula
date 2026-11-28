"""Cache source bytes, reviewed chemistry records, and immutable selections."""

import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from particula.dynamics.chemistry.catalog import camx, cmc
from particula.dynamics.chemistry.catalog.native import (
    compile_channel,
    native_mechanism,
)
from particula.dynamics.chemistry.catalog.readers import (
    SOURCES,
    labeled,
    mozart,
    ro2_members,
)

IMPORT_VERSION = 4
LOCK_FORMAT = "particula.chemical-selection/1"


def _json(data):
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _fingerprint(data):
    return _digest(_json(data).encode())


def _read_entries(source, text, digest, companions):
    if source in {"waccm", "e3sm"}:
        return mozart(text)
    if source in {"mcm", "wrf-chem", "cmaq"}:
        return labeled(text, source, companions)
    if source == "pmcamx" and re.search(
        r"^\s*subroutine\s+soachem\(", text, re.I | re.M
    ):
        return camx.soap(text, digest)
    texts = [data.decode("utf-8-sig") for data in companions]
    maps = [m for content in texts if (m := camx.equations(content))]
    expected = (
        {"camx": "1", "pmcamx": "7"}.get(source)
        if digest in camx.TITRATION_DECKS[source]
        else None
    )
    tables = cmc.read_cmc(texts, expected)
    if len(maps) > 1 or (maps and tables):
        raise ValueError("provide one matching CAMx mechanism listing")
    return camx.rate_deck(text, source, digest, maps[0] if maps else {}, tables)


def _revision(entry, record, files, manifest):
    model = {
        k: v
        for k, v in record.items()
        if k not in {"label", "line", "equation", "expression"}
    }
    if (
        not entry.kinetics
        or not entry.reactants
        or not entry.products
        or entry.metadata.get("annotated_equation")
    ):
        model.update(equation=entry.equation, expression=entry.expression)
    expressions = (
        entry.expression,
        *entry.metadata.get("rate_definitions", {}).values(),
    )
    uses_pool = "RO2" in manifest and any(
        re.search(r"\bRO2\b", expr, re.I) for expr in expressions
    )
    if entry.reason or uses_pool:
        model["companions"] = [f["sha256"] for f in files[1:]]
    if uses_pool:
        model["RO2"] = manifest["RO2"]
    return _fingerprint(model)


def _inputs(path, companions, origins):
    paths = [
        Path(path),
        *sorted(map(Path, companions), key=lambda p: p.name),
    ]
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("source and companion basenames must be distinct")
    payloads = [p.read_bytes() for p in paths]
    files = [
        {"name": p.name, "sha256": _digest(data)}
        for p, data in zip(paths, payloads, strict=True)
    ]
    if origins:
        origin_map = {item["name"]: item for item in origins}
        for item in files:
            origin = origin_map[item["name"]]
            if origin["sha256"] != item["sha256"]:
                raise ValueError(f"source digest mismatch for {item['name']}")
            item.update(origin)

    return paths, payloads, files


class ChemistryCatalog:
    """Import local files once and query the resulting persistent records."""

    def __init__(self, path=None):
        """Open or create the local SQLite catalog at path."""
        default = (
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "particula"
            / "chemistry.sqlite3"
        )
        self.path = Path(
            path or os.environ.get("PARTICULA_CHEM_CATALOG", default)
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS files(
                   digest TEXT PRIMARY KEY, data BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots(
                   id TEXT PRIMARY KEY, manifest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reactions(
                   id TEXT PRIMARY KEY, snapshot TEXT NOT NULL,
                   source TEXT NOT NULL, record TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS by_snapshot ON reactions(snapshot);
                CREATE TABLE IF NOT EXISTS terms(
                   species TEXT, role TEXT, id TEXT,
                   PRIMARY KEY(species, role, id));
                CREATE INDEX IF NOT EXISTS by_term_id ON terms(id);
                """)

    @contextmanager
    def _db(self):
        with closing(sqlite3.connect(self.path, timeout=60)) as db, db:
            yield db

    def source_bytes(self, digest):
        """Read a cached upstream file by its SHA-256 digest."""
        with self._db() as db:
            row = db.execute(
                "SELECT data FROM files WHERE digest=?", (digest,)
            ).fetchone()
        return row[0] if row else None

    def fetch(self, recipe):
        """Import the verified files of a bundled source recipe."""
        from particula.dynamics.chemistry.catalog.sources import fetch

        return fetch(self, recipe)

    def import_file(
        self,
        source,
        path,
        *,
        release,
        mechanism,
        companions=(),
        aerosol=None,
        origins=(),
    ):
        """Pin a source and optional companion files; return its snapshot ID."""
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        if not all(re.fullmatch(r"[\w.-]+", v) for v in (release, mechanism)):
            raise ValueError(
                "release and mechanism require names using letters, "
                "numbers, dots, underscores or hyphens"
            )
        paths, payloads, files = _inputs(path, companions, origins)
        manifest = dict(
            source=source,
            release=release,
            mechanism=mechanism,
            aerosol=aerosol,
            files=files,
            importer_version=IMPORT_VERSION,
        )
        snap = f"{source}:{mechanism}@{_fingerprint(manifest)}"
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM snapshots WHERE id=?", (snap,)
            ).fetchone():
                return snap
            text = payloads[0].decode("utf-8-sig")
            if source == "mcm":
                manifest["RO2"] = ro2_members(text)
            for f, data in zip(files, payloads, strict=True):
                db.execute(
                    "INSERT OR IGNORE INTO files VALUES (?,?)",
                    (f["sha256"], data),
                )
            count = 0
            for entry in _read_entries(
                source, text, files[0]["sha256"], payloads[1:]
            ):
                record = entry.record()
                rid = f"{snap}/{entry.label}"
                record.update(
                    id=rid,
                    snapshot=snap,
                    source=source,
                    revision=_revision(entry, record, files, manifest),
                )
                try:
                    db.execute(
                        "INSERT INTO reactions VALUES (?,?,?,?)",
                        (rid, snap, source, _json(record)),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"duplicate source label {entry.label}"
                    ) from exc
                for role, terms in (
                    ("reactant", entry.reactants),
                    ("product", entry.products),
                    ("product", entry.metadata.get("product_bins", ())),
                ):
                    db.executemany(
                        "INSERT OR IGNORE INTO terms VALUES (?,?,?)",
                        ((s, role, rid) for s in terms),
                    )
                count += 1
            if not count:
                raise ValueError(
                    f"no {source} reactions found in {paths[0].name}"
                )
            manifest["count"] = count
            db.execute(
                "INSERT INTO snapshots VALUES (?,?)", (snap, _json(manifest))
            )
        return snap

    def search(self, *, species=None, source=None, role=None, snapshot=None):
        """Search exact source species and roles across imported snapshots."""
        if role and role not in {"reactant", "product"}:
            raise ValueError("role must be reactant or product")
        # Updated readers can index the same byte-identical case differently.
        # Keep older snapshots addressable while showing the latest parse once.
        query = """
            WITH versions AS (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY json_remove(manifest, '$.importer_version',
                        '$.count', '$.RO2')
                    ORDER BY json_extract(manifest,'$.importer_version') DESC
                ) AS rank FROM snapshots
            )
            SELECT record FROM reactions r
            WHERE (:source IS NULL OR r.source=:source)
            AND ((:snapshot IS NULL AND r.snapshot IN
                  (SELECT id FROM versions WHERE rank=1))
                 OR r.snapshot=:snapshot)
            AND ((:species IS NULL AND :role IS NULL) OR EXISTS (
                SELECT 1 FROM terms t WHERE t.id=r.id
                AND (:species IS NULL OR t.species=:species)
                AND (:role IS NULL OR t.role=:role)))
            ORDER BY r.id
        """
        with self._db() as db:
            rows = db.execute(
                query,
                dict(
                    source=source, species=species, role=role, snapshot=snapshot
                ),
            )
            result = []
            for (data,) in rows:
                record = json.loads(data)
                result.append(
                    {
                        k: record[k]
                        for k in (
                            "id",
                            "label",
                            "equation",
                            "source",
                            "revision",
                        )
                    }
                )
            return result

    def show(self, identifier):
        """Inspect source parameters, provenance and export diagnostics."""
        with self._db() as db:
            row = db.execute(
                "SELECT manifest FROM snapshots WHERE id=?", (identifier,)
            ).fetchone()
            if row:
                return dict(id=identifier, **json.loads(row[0]))
            row = db.execute(
                "SELECT record,manifest FROM reactions JOIN snapshots "
                "ON snapshots.id=reactions.snapshot WHERE reactions.id=?",
                (identifier,),
            ).fetchone()
        if not row:
            raise ValueError(f"unknown catalog ID: {identifier}")
        r = json.loads(row[0])
        try:
            native, _ = compile_channel(r)
            issue = ""
        except ValueError as exc:
            native = None
            issue = str(exc)
        return dict(
            **r,
            provenance=json.loads(row[1]),
            export_issue=issue,
            native_reaction=native,
            implementation=None
            if issue
            else (
                "particula.dynamics.chemistry.mechanism.ReactionMechanism.bind"
            ),
        )

    def selection(self, identifiers, *, close_oxygen=False):
        """Pin individual channels or every channel in an imported source."""
        records = {}
        manifests = {}
        with self._db() as db:
            for ident in identifiers:
                rows = db.execute(
                    "SELECT record FROM reactions WHERE id=? OR snapshot=? "
                    "ORDER BY id",
                    (ident, ident),
                ).fetchall()
                if not rows:
                    raise ValueError(f"unknown catalog ID: {ident}")
                for (data,) in rows:
                    r = json.loads(data)
                    records[r["id"]] = r
                    s = r["snapshot"]
                    if s not in manifests:
                        manifests[s] = json.loads(
                            db.execute(
                                "SELECT manifest FROM snapshots WHERE id=?",
                                (s,),
                            ).fetchone()[0]
                        )
        if not records:
            raise ValueError("select at least one reaction")
        lock = dict(
            format=LOCK_FORMAT,
            close_oxygen=close_oxygen,
            snapshots=manifests,
            reactions=[records[k] for k in sorted(records)],
        )
        return dict(**lock, sha256=_fingerprint(lock))

    def export(
        self,
        identifiers,
        output,
        *,
        target="native",
        lock_path=None,
        close_oxygen=False,
    ):
        """Write a study selection lock or an executable native mechanism.

        Args:
            identifiers: Reaction or snapshot IDs to select. Pass an empty
                sequence when replaying a lock.
            output: Destination for the JSON lock or native JSON/YAML file.
                A successful export atomically replaces this file.
            target: Use ``"lock"`` to preserve the source records and oxygen
                assumption, or ``"native"`` to compile the selected channels.
            lock_path: Path to a saved selection lock. Replay uses the oxygen
                treatment recorded at lock creation.
            close_oxygen: For NO + O₃ → NO₂ + O₂, include O₂ as an evolving
                product when the source omits or prescribes it. Bind the
                resulting mechanism to a gas inventory with an O₂ column.
                Defaults to False. To change a saved assumption, create a
                new lock from the source IDs.

        Raises:
            ValueError: If the selection, lock, or target is invalid, or a
                selected channel cannot be represented by the native solver.
        """
        if target not in {"lock", "native"}:
            raise ValueError("target must be lock or native")
        dest = Path(output)
        if dest.resolve() == self.path.resolve() or (
            lock_path and dest.resolve() == Path(lock_path).resolve()
        ):
            raise ValueError(
                "destination must differ from the catalog and input lock"
            )
        if lock_path:
            if identifiers:
                raise ValueError("select IDs or supply one lock")
            lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
            if not isinstance(lock, dict) or lock.get("format") != LOCK_FORMAT:
                raise ValueError("unsupported lock format")
            checksum = lock.pop("sha256", None)
            if checksum != _fingerprint(lock):
                raise ValueError("selection lock checksum mismatch")
            lock["sha256"] = checksum
        else:
            lock = self.selection(identifiers, close_oxygen=close_oxygen)
        if lock_path and close_oxygen and not lock.get("close_oxygen"):
            raise ValueError("select --close-oxygen when creating the lock")
        mechanism = (
            native_mechanism(
                lock["reactions"], close_oxygen=lock.get("close_oxygen", False)
            )
            if target == "native"
            else None
        )
        with NamedTemporaryFile(
            dir=dest.parent, suffix=dest.suffix, delete=False
        ) as f:
            tmp = Path(f.name)
        try:
            if mechanism:
                mechanism.save(tmp)
            else:
                tmp.write_text(
                    json.dumps(lock, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
