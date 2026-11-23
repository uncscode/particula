# particula.particles.prepared_activity

*module*

Optional direct activity-provider contract for native condensation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/prepared_activity.py#L1-L1)

## particula.particles.prepared_activity.PreparedActivityProvider

*class*

```python
class PreparedActivityProvider(Protocol)
```

Bases: `Protocol`

Version 1 provides primitive checkpoint identity and device binding.

A binding owns all workspace and a float64 (box, slot, material)
activity buffer. Its evaluate(masses, counts, temperature, invalid)
enqueues on the current Warp stream, setting invalid[0] on a failed row.
Only mapped neutral targets have meaningful output. Status retains original
box and slot order for diagnosis at an observation or failure boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/prepared_activity.py#L6-L24)

### particula.particles.prepared_activity.PreparedActivityProvider.prepared_provider_version

*attribute* · *instance attribute*

```python
prepared_provider_version: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/prepared_activity.py#L16-L16)

### particula.particles.prepared_activity.PreparedActivityProvider.bind_device

*method*

```python
def bind_device(shape, targets, device) -> Any
```

Allocate owner-specific workspace and mapped outputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/prepared_activity.py#L22-L24)

### particula.particles.prepared_activity.PreparedActivityProvider.prepared_identity

*method*

```python
def prepared_identity() -> tuple
```

Return immutable layout, physics and parameter identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/prepared_activity.py#L18-L20)

## particula.particles.prepared_activity.provider_identity

*function*

```python
def provider_identity(activity)
```

Identify an external strategy that opts into preparation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/prepared_activity.py#L27-L34)

## particula.particles.prepared_activity.restore_provider

*function*

```python
def restore_provider(identity)
```

Rebuild a registered backend and verify its scientific identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/prepared_activity.py#L37-L43)
