# particula.execution.analysis_io

*module*

Portable analysis results with explicit scientific and numerical provenance.

ZIP/NPY with a JSON manifest retains exact numeric dtypes and arbitrary named
axes without a new runtime dependency. HDF5/Zarr offer partial/chunked access
but require additional libraries and more complex atomic multi-object writes.
This bounded single-file format suits selected histories; it is not a streaming
store for unlimited particle snapshots, nor an exact-restart format.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L1-L1)

## particula.execution.analysis_io.AnalysisResult

*class* · *dataclass*

```python
class AnalysisResult
```

Own loaded numeric observations and JSON scientific provenance.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L21-L26)

### particula.execution.analysis_io.AnalysisResult.arrays

*attribute* · *instance attribute*

```python
arrays: dict[str, np.ndarray]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L26-L26)

### particula.execution.analysis_io.AnalysisResult.metadata

*attribute* · *instance attribute*

```python
metadata: dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L25-L25)

### particula.execution.analysis_io.AnalysisResult.__init__

*method*

```python
def __init__(metadata: dict, arrays: dict[str, np.ndarray]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L0-L0)

## particula.execution.analysis_io.AnalysisWriter

*class*

```python
class AnalysisWriter
```

Compose accepted recording with atomic replacement of an analysis file.

Each new output saves the bounded history so far. A write failure leaves
accepted physics and the new in-memory row intact; the prior file remains
usable. flush() retries that row without advancing physics or duplicating
history. File writing is a host observation action, outside CUDA capture.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L29-L81)

### particula.execution.analysis_io.AnalysisWriter._options

*attribute* · *instance attribute*

```python
_options = deepcopy(dict(configuration=configuration, numerical_settings=numerical_settings, source_identity=source_identity, compressed=compressed))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L53-L60)

### particula.execution.analysis_io.AnalysisWriter._saved_time

*attribute* · *instance attribute*

```python
_saved_time = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L65-L65)

### particula.execution.analysis_io.AnalysisWriter.path

*attribute* · *instance attribute*

```python
path = Path(path)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L52-L52)

### particula.execution.analysis_io.AnalysisWriter.recorder

*attribute* · *instance attribute*

```python
recorder = recorder
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L51-L51)

### particula.execution.analysis_io.AnalysisWriter.last_time

*property*

```python
last_time
```

Return the last recorded time, including an unsaved row.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L68-L70)

### particula.execution.analysis_io.AnalysisWriter.__init__

*method*

```python
def __init__(recorder, path, *, configuration, numerical_settings, source_identity, compressed=True)
```

Detach and validate provenance without creating an output file.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L38-L65)

### particula.execution.analysis_io.AnalysisWriter.flush

*method*

```python
def flush()
```

Save an unsaved row; successful repeats do not replace the file.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L77-L81)

### particula.execution.analysis_io.AnalysisWriter.record

*method*

```python
def record(time, state)
```

Record a complete accepted state and save its accumulated history.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L72-L75)

## particula.execution.analysis_io._validate

*function*

```python
def _validate(metadata, arrays)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L178-L212)

## particula.execution.analysis_io._validate_header

*function*

```python
def _validate_header(metadata)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L126-L140)

## particula.execution.analysis_io._validate_science

*function*

```python
def _validate_science(science)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L143-L175)

## particula.execution.analysis_io._validate_species

*function*

```python
def _validate_species(science, dimensions)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L231-L254)

## particula.execution.analysis_io.load_analysis

*function*

```python
def load_analysis(path, *, max_bytes=128 * 1024 * 1024)
```

Load and validate analysis data without executable deserialization.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L215-L228)

## particula.execution.analysis_io.save_analysis

*function*

```python
def save_analysis(recorder, path, *, configuration, numerical_settings, source_identity, compressed=True)
```

Save observations atomically with reproducibility inputs.

configuration, numerical_settings and source_identity are JSON objects, not
executable process instances. The caller supplies physical assumptions,
parameters, process order, tolerances, source revision and input identities.
Time is seconds; the scientific metadata carries all other units and axes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/analysis_io.py#L84-L123)
