# particula.dynamics.chemistry.catalog

*module*

Local, versioned catalog of atmospheric model chemistry.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/__init__.py#L1-L1)

## particula.dynamics.chemistry.catalog.ChemistryCatalog

*class*

```python
class ChemistryCatalog
```

Re-exported from: `particula.dynamics.chemistry.catalog.store`

Import local files once and query the resulting persistent records.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L115-L471)

### particula.dynamics.chemistry.catalog.ChemistryCatalog.path

*attribute* · *instance attribute*

```python
path = Path(path or os.environ.get('PARTICULA_CHEM_CATALOG', default))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L125-L127)

### particula.dynamics.chemistry.catalog.ChemistryCatalog.__init__

*method*

```python
def __init__(path=None)
```

Open or create the local SQLite catalog at path.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L118-L145)

### particula.dynamics.chemistry.catalog.ChemistryCatalog._db

*method*

```python
def _db()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L147-L150)

### particula.dynamics.chemistry.catalog.ChemistryCatalog.export

*method*

```python
def export(identifiers, output, *, target='native', lock_path=None, close_oxygen=False)
```

Write a study selection lock or an executable native mechanism.

**Parameters**

- `identifiers` — Reaction or snapshot IDs to select. Pass an empty sequence when replaying a lock.
- `output` — Destination for the JSON lock or native JSON/YAML file. A successful export atomically replaces this file.
- `target` (default: `'native'`) — Use ``"lock"`` to preserve the source records and oxygen assumption, or ``"native"`` to assemble the native channels.
- `lock_path` (default: `None`) — Path to a saved selection lock. Replay uses the oxygen treatment recorded at lock creation.
- `close_oxygen` (default: `False`) — For NO + O₃ → NO₂ + O₂, include O₂ as an evolving product when the source omits or prescribes it. Bind the resulting mechanism to a gas inventory with an O₂ column. Defaults to False. To change a saved assumption, create a new lock from the source IDs.

**Raises**

- `ValueError` — If the selection, lock, or target is invalid, or a
selected channel cannot be represented by the native solver.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L394-L471)

### particula.dynamics.chemistry.catalog.ChemistryCatalog.fetch

*method*

```python
def fetch(recipe=None, *, catalog_version=None)
```

Fetch catalog records for offline search and export.

**Parameters**

- `recipe` (default: `None`) — Optional source key such as `waccm`. Omitting the key downloads the latest stable chemistry catalog release.
- `catalog_version` (default: `None`) — Exact catalog data version, such as ``1.0.0``, when fetching the prebuilt artifact.

**Returns**

- The release manifest for a prebuilt catalog, or the immutable
- snapshot ID for a named source import.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L160-L184)

### particula.dynamics.chemistry.catalog.ChemistryCatalog.import_file

*method*

```python
def import_file(source, path, *, release, mechanism, companions=(), aerosol=None, origins=())
```

Pin a source and optional companion files; return its snapshot ID.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L186-L268)

### particula.dynamics.chemistry.catalog.ChemistryCatalog.search

*method*

```python
def search(species=None, source=None, role=None, snapshot=None)
```

Search exact source species and roles across imported snapshots.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L270-L322)

### particula.dynamics.chemistry.catalog.ChemistryCatalog.selection

*method*

```python
def selection(identifiers, *, close_oxygen=False)
```

Pin individual channels or every channel in an imported source.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L358-L392)

### particula.dynamics.chemistry.catalog.ChemistryCatalog.show

*method*

```python
def show(identifier)
```

Inspect source parameters, provenance and export diagnostics.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L324-L356)

### particula.dynamics.chemistry.catalog.ChemistryCatalog.source_bytes

*method*

```python
def source_bytes(digest)
```

Read a cached upstream file by its SHA-256 digest.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L152-L158)

**Modules**

- `particula.dynamics.chemistry.catalog.bundle`
- `particula.dynamics.chemistry.catalog.camx`
- `particula.dynamics.chemistry.catalog.cli`
- `particula.dynamics.chemistry.catalog.cmc`
- `particula.dynamics.chemistry.catalog.native`
- `particula.dynamics.chemistry.catalog.readers`
- `particula.dynamics.chemistry.catalog.sources`
- `particula.dynamics.chemistry.catalog.store`
