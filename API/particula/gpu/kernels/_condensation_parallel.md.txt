# particula.gpu.kernels._condensation_parallel

*module*

Deterministic chunk reductions for large native condensation populations.

No atomics or particle-pair storage are used. Grouping changes roundoff relative
to the serial route; representable transfer reconciliation still uses the same
four downward corrections followed by suppression of excessive uptake. All
scratch is allocated at preparation and overwritten before each use.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L1-L1)

## particula.gpu.kernels._condensation_parallel.CHUNK_SIZE

*attribute* · *module attribute*

```python
CHUNK_SIZE = 256
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L16-L16)

## particula.gpu.kernels._condensation_parallel.MIN_SLOTS

*attribute* · *module attribute*

```python
MIN_SLOTS = 4096
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L17-L17)

## particula.gpu.kernels._condensation_parallel._ReductionWork

*class* · *dataclass*

```python
class _ReductionWork
```

Private scratch, retained by the prepared call and its captured graph.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L167-L174)

### particula.gpu.kernels._condensation_parallel._ReductionWork.negative

*attribute* · *instance attribute*

```python
negative: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L172-L172)

### particula.gpu.kernels._condensation_parallel._ReductionWork.partial

*attribute* · *instance attribute*

```python
partial: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L173-L173)

### particula.gpu.kernels._condensation_parallel._ReductionWork.positive

*attribute* · *instance attribute*

```python
positive: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L171-L171)

### particula.gpu.kernels._condensation_parallel._ReductionWork.status

*attribute* · *instance attribute*

```python
status: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L174-L174)

### particula.gpu.kernels._condensation_parallel._ReductionWork.__init__

*method*

```python
def __init__(positive: Any, negative: Any, partial: Any, status: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L0-L0)

## particula.gpu.kernels._condensation_parallel._candidate_chunks

*function*

```python
def _candidate_chunks(candidate: Any, concentration: Any, masses: Any, mapping: Any, volume: Any, kind: wp.int32, uptake: Any, release: Any)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L20-L52)

## particula.gpu.kernels._condensation_parallel._correct_transfer

*function*

```python
def _correct_transfer(finalized: Any, masses: Any, mapping: Any, status: Any, attempt: wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L107-L130)

## particula.gpu.kernels._condensation_parallel._energy_chunks

*function*

```python
def _energy_chunks(total: Any, concentration: Any, mapping: Any, volume: Any, kind: wp.int32, partial: Any)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L133-L154)

## particula.gpu.kernels._condensation_parallel._finish_candidates

*function*

```python
def _finish_candidates(positive: Any, negative: Any, demand: Any, release: Any)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L55-L65)

## particula.gpu.kernels._condensation_parallel._finish_energy

*function*

```python
def _finish_energy(partial: Any, latent: Any, energy: Any)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L157-L164)

## particula.gpu.kernels._condensation_parallel._finish_transfer

*function*

```python
def _finish_transfer(partial: Any, gas: Any, delta: Any, status: Any)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L91-L104)

## particula.gpu.kernels._condensation_parallel._transfer_chunks

*function*

```python
def _transfer_chunks(transfer: Any, concentration: Any, mapping: Any, volume: Any, kind: wp.int32, partial: Any)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L68-L88)

## particula.gpu.kernels._condensation_parallel.candidates

*function*

```python
def candidates(p)
```

Reduce proposals into the same full-gas demand/release outputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L186-L209)

## particula.gpu.kernels._condensation_parallel.energy

*function*

```python
def energy(p)
```

Reduce accumulated transfer to extensive energy in full gas order.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L255-L276)

## particula.gpu.kernels._condensation_parallel.finalized

*function*

```python
def finalized(p)
```

Reconcile rounded uptake with a fixed capture-safe sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L212-L252)

## particula.gpu.kernels._condensation_parallel.prepare

*function*

```python
def prepare(boxes, slots, gases, device)
```

Allocate O(B G ceil(N/256)) storage once, outside stepping.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_parallel.py#L177-L183)
