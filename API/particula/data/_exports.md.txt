# particula.data._exports

*module*

Resolve public data imports without loading optional packages at startup.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_exports.py#L1-L1)

## particula.data._exports._MODULES

*attribute* · *module attribute*

```python
_MODULES = {'Measurement': 'measurement', 'MobilityNumberSpectrum': 'measurement', 'ParticleCountRate': 'measurement', 'migrate_settings': 'migration', 'ProfileError': 'profile', 'describe_profile': 'profile', 'validate_profile': 'profile', 'MeasurementReader': 'reader', 'inspect_source': 'reader', 'import_to_parquet': 'storage', 'open_measurement': 'storage', 'read_measurement': 'storage'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_exports.py#L7-L20)

## particula.data._exports.load_export

*function*

```python
def load_export(name: str) -> Any
```

Load a public object on first access and cache it on the data package.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_exports.py#L23-L31)
