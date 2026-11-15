# particula.execution.checkpoint_extensions

*module*

Canonical registered state and versioned prepared-checkpoint validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_extensions.py#L1-L1)

## particula.execution.checkpoint_extensions._payload_map

*function*

```python
def _payload_map(payloads, expected)
```

Validate full inventory, primitive dtype, shape and finite bytes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_extensions.py#L93-L112)

## particula.execution.checkpoint_extensions.configurations

*function*

```python
def configurations(plans)
```

Strip live adapter implementations from checkpoint configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_extensions.py#L20-L25)

## particula.execution.checkpoint_extensions.restore_host_fields

*function*

```python
def restore_host_fields(saved, host)
```

Rebuild validated CPU fields from authoritative bytes before upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_extensions.py#L115-L150)

## particula.execution.checkpoint_extensions.restore_host_rng

*function*

```python
def restore_host_rng(saved, plans)
```

Validate transfer mode and host stream identities before upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_extensions.py#L225-L265)

## particula.execution.checkpoint_extensions.restore_targets

*function*

```python
def restore_targets(owner)
```

Resolve existing registered arrays; restart never replaces bindings.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_extensions.py#L197-L222)

## particula.execution.checkpoint_extensions.save_extensions

*function*

```python
def save_extensions(owner)
```

Download registered state into immutable byte payloads after sync.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_extensions.py#L28-L64)

## particula.execution.checkpoint_extensions.validate_envelope

*function*

```python
def validate_envelope(saved)
```

Explicitly admit legacy built-in-only v1 or the extension-aware v2.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_extensions.py#L67-L90)

## particula.execution.checkpoint_extensions.validate_extension_state

*function*

```python
def validate_extension_state(saved, plans)
```

Check declared adapter state and device clock before any upload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_extensions.py#L153-L194)
