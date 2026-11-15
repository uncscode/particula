# particula.execution.extension_rng

*module*

Stable logical-box random words for explicitly stochastic GPU extensions.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_rng.py#L1-L1)

## particula.execution.extension_rng.initial_words

*function*

```python
def initial_words(configuration, root_seed, box_ids)
```

Derive versioned uint32 words without depending on lane or process order.

These are Warp random-state words, separate from CPU PCG64 continuation.
Reject collisions within a process instead of silently sharing a stream.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_rng.py#L9-L34)

## particula.execution.extension_rng.validate_streams

*function*

```python
def validate_streams(plans, root_seed, box_ids)
```

Check initial derivation before uploading any simulation buffers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/extension_rng.py#L37-L43)
