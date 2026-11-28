# particula.dynamics.chemistry.catalog.store

*module*

Cache source bytes, reviewed chemistry records, and immutable selections.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L1-L1)

## particula.dynamics.chemistry.catalog.store.IMPORT_VERSION

*attribute* · *module attribute*

```python
IMPORT_VERSION = 5
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L24-L24)

## particula.dynamics.chemistry.catalog.store.LOCK_FORMAT

*attribute* · *module attribute*

```python
LOCK_FORMAT = 'particula.chemical-selection/1'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L25-L25)

## particula.dynamics.chemistry.catalog.store.NATIVE_LOCK_FORMAT

*attribute* · *module attribute*

```python
NATIVE_LOCK_FORMAT = 'particula.chemical-selection/2'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L26-L26)

## particula.dynamics.chemistry.catalog.store.ChemistryCatalog

*class*

```python
class ChemistryCatalog
```

Import local files once and query the resulting persistent records.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L115-L471)

### particula.dynamics.chemistry.catalog.store.ChemistryCatalog.path

*attribute* · *instance attribute*

```python
path = Path(path or os.environ.get('PARTICULA_CHEM_CATALOG', default))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L125-L127)

### particula.dynamics.chemistry.catalog.store.ChemistryCatalog.__init__

*method*

```python
def __init__(path=None)
```

Open or create the local SQLite catalog at path.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L118-L145)

### particula.dynamics.chemistry.catalog.store.ChemistryCatalog._db

*method*

```python
def _db()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L147-L150)

### particula.dynamics.chemistry.catalog.store.ChemistryCatalog.export

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

### particula.dynamics.chemistry.catalog.store.ChemistryCatalog.fetch

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

### particula.dynamics.chemistry.catalog.store.ChemistryCatalog.import_file

*method*

```python
def import_file(source, path, *, release, mechanism, companions=(), aerosol=None, origins=())
```

Pin a source and optional companion files; return its snapshot ID.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L186-L268)

### particula.dynamics.chemistry.catalog.store.ChemistryCatalog.search

*method*

```python
def search(species=None, source=None, role=None, snapshot=None)
```

Search exact source species and roles across imported snapshots.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L270-L322)

### particula.dynamics.chemistry.catalog.store.ChemistryCatalog.selection

*method*

```python
def selection(identifiers, *, close_oxygen=False)
```

Pin individual channels or every channel in an imported source.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L358-L392)

### particula.dynamics.chemistry.catalog.store.ChemistryCatalog.show

*method*

```python
def show(identifier)
```

Inspect source parameters, provenance and export diagnostics.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L324-L356)

### particula.dynamics.chemistry.catalog.store.ChemistryCatalog.source_bytes

*method*

```python
def source_bytes(digest)
```

Read a cached upstream file by its SHA-256 digest.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L152-L158)

## particula.dynamics.chemistry.catalog.store._digest

*function*

```python
def _digest(data)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L35-L36)

## particula.dynamics.chemistry.catalog.store._fingerprint

*function*

```python
def _fingerprint(data)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L39-L40)

## particula.dynamics.chemistry.catalog.store._inputs

*function*

```python
def _inputs(path, companions, origins)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L92-L112)

## particula.dynamics.chemistry.catalog.store._json

*function*

```python
def _json(data)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L29-L32)

## particula.dynamics.chemistry.catalog.store._read_entries

*function*

```python
def _read_entries(source, text, digest, companions)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L43-L62)

## particula.dynamics.chemistry.catalog.store._revision

*function*

```python
def _revision(entry, record, files, manifest)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/store.py#L65-L89)
