# particula.dynamics.chemistry.catalog.bundle

*module*

Install the prebuilt reaction catalog from a versioned catalog release.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/bundle.py#L1-L1)

## particula.dynamics.chemistry.catalog.bundle.ASSET

*attribute* · *module attribute*

```python
ASSET = 'particula-chemistry-catalog.zip'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/bundle.py#L15-L15)

## particula.dynamics.chemistry.catalog.bundle.DOWNLOAD_URL

*attribute* · *module attribute*

```python
DOWNLOAD_URL = 'https://github.com/uncscode/particula/releases/download'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/bundle.py#L20-L20)

## particula.dynamics.chemistry.catalog.bundle.FORMAT

*attribute* · *module attribute*

```python
FORMAT = 'particula.chemistry-catalog/2'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/bundle.py#L16-L16)

## particula.dynamics.chemistry.catalog.bundle.RELEASES_API

*attribute* · *module attribute*

```python
RELEASES_API = 'https://api.github.com/repos/uncscode/particula/releases'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/bundle.py#L19-L19)

## particula.dynamics.chemistry.catalog.bundle.TAG_PREFIX

*attribute* · *module attribute*

```python
TAG_PREFIX = 'chemistry-catalog-v'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/bundle.py#L17-L17)

## particula.dynamics.chemistry.catalog.bundle.VERSION_PATTERN

*attribute* · *module attribute*

```python
VERSION_PATTERN = '(?:0|[1-9][0-9]*)(?:\\.(?:0|[1-9][0-9]*)){2}'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/bundle.py#L18-L18)

## particula.dynamics.chemistry.catalog.bundle._download

*function*

```python
def _download(archive, tag)
```

Download both assets from the selected catalog tag and verify the ZIP.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/bundle.py#L60-L81)

## particula.dynamics.chemistry.catalog.bundle._install_snapshot

*function*

```python
def _install_snapshot(db, snapshot, data)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/bundle.py#L84-L126)

## particula.dynamics.chemistry.catalog.bundle.catalog_tag

*function*

```python
def catalog_tag(version)
```

Return the stable GitHub release tag for a catalog data version.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/bundle.py#L23-L27)

## particula.dynamics.chemistry.catalog.bundle.install_release

*function*

```python
def install_release(catalog, version=None)
```

Index every released source entry and native conversion atomically.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/bundle.py#L129-L156)

## particula.dynamics.chemistry.catalog.bundle.latest_version

*function*

```python
def latest_version()
```

Find the highest stable catalog version across release pages.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/bundle.py#L30-L57)
