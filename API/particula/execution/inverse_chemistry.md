# particula.execution.inverse_chemistry

*module*

Differentiate a fixed SiMBA network through deterministic chamber growth.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L1-L1)

## particula.execution.inverse_chemistry.SeededChamber

*class* · *dataclass*

```python
class SeededChamber
```

Fixed chemistry, seed composition, instrument and chamber conditions.

Densities (kg/m³) and initial seed volume fractions are keyed by particle
species. Gas and particle species require declared molar masses. A flat
interface, ideal activities and unit accommodation define vapor exchange.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L27-L44)

### particula.execution.inverse_chemistry.SeededChamber.cohorts

*attribute* · *instance attribute*

```python
cohorts: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L44-L44)

### particula.execution.inverse_chemistry.SeededChamber.density

*attribute* · *instance attribute*

```python
density: dict[str, float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L37-L37)

### particula.execution.inverse_chemistry.SeededChamber.edges_m

*attribute* · *instance attribute*

```python
edges_m: tuple[float, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L42-L42)

### particula.execution.inverse_chemistry.SeededChamber.exchange

*attribute* · *instance attribute*

```python
exchange: tuple[VaporExchange, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L39-L39)

### particula.execution.inverse_chemistry.SeededChamber.log_response_width

*attribute* · *instance attribute*

```python
log_response_width: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L43-L43)

### particula.execution.inverse_chemistry.SeededChamber.measured_gases

*attribute* · *instance attribute*

```python
measured_gases: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L41-L41)

### particula.execution.inverse_chemistry.SeededChamber.mechanism

*attribute* · *instance attribute*

```python
mechanism: ReactionMechanism
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L36-L36)

### particula.execution.inverse_chemistry.SeededChamber.organic

*attribute* · *instance attribute*

```python
organic: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L40-L40)

### particula.execution.inverse_chemistry.SeededChamber.seed_fractions

*attribute* · *instance attribute*

```python
seed_fractions: dict[str, float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L38-L38)

### particula.execution.inverse_chemistry.SeededChamber.__init__

*method*

```python
def __init__(mechanism: ReactionMechanism, density: dict[str, float], seed_fractions: dict[str, float], exchange: tuple[VaporExchange, ...], organic: tuple[str, ...], measured_gases: tuple[str, ...], edges_m: tuple[float, ...], log_response_width: float, cohorts: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L0-L0)

## particula.execution.inverse_chemistry.Trajectory

*class* · *dataclass*

```python
class Trajectory
```

Predictions, final inventories, loss and optional derivatives.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L47-L56)

### particula.execution.inverse_chemistry.Trajectory.final_state

*attribute* · *instance attribute*

```python
final_state: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L52-L52)

### particula.execution.inverse_chemistry.Trajectory.gradients

*attribute* · *instance attribute*

```python
gradients: dict[str, np.ndarray]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L55-L55)

### particula.execution.inverse_chemistry.Trajectory.loss

*attribute* · *instance attribute*

```python
loss: float | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L54-L54)

### particula.execution.inverse_chemistry.Trajectory.predictions

*attribute* · *instance attribute*

```python
predictions: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L51-L51)

### particula.execution.inverse_chemistry.Trajectory.species

*attribute* · *instance attribute*

```python
species: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L53-L53)

### particula.execution.inverse_chemistry.Trajectory.transfer_bytes

*attribute* · *instance attribute*

```python
transfer_bytes: dict[str, int]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L56-L56)

### particula.execution.inverse_chemistry.Trajectory.__init__

*method*

```python
def __init__(predictions: np.ndarray, final_state: np.ndarray, species: tuple[str, ...], loss: float | None, gradients: dict[str, np.ndarray], transfer_bytes: dict[str, int]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L0-L0)

## particula.execution.inverse_chemistry.VaporExchange

*class* · *dataclass*

```python
class VaporExchange
```

Paired phases of one compound and its constant transport properties.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L17-L24)

### particula.execution.inverse_chemistry.VaporExchange.diffusivity_m2_s

*attribute* · *instance attribute*

```python
diffusivity_m2_s: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L23-L23)

### particula.execution.inverse_chemistry.VaporExchange.gas

*attribute* · *instance attribute*

```python
gas: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L21-L21)

### particula.execution.inverse_chemistry.VaporExchange.mean_free_path_m

*attribute* · *instance attribute*

```python
mean_free_path_m: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L24-L24)

### particula.execution.inverse_chemistry.VaporExchange.particle

*attribute* · *instance attribute*

```python
particle: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L22-L22)

### particula.execution.inverse_chemistry.VaporExchange.__init__

*method*

```python
def __init__(gas: str, particle: str, diffusivity_m2_s: float, mean_free_path_m: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L0-L0)

## particula.execution.inverse_chemistry._compile

*function*

```python
def _compile(chamber)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L72-L119)

## particula.execution.inverse_chemistry._finish

*function*

```python
def _finish(bad, outputs, observations, tape, floats, params, gas_ids, state, names, download, traffic)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L382-L412)

## particula.execution.inverse_chemistry._integrate

*function*

```python
def _integrate(state, duration, max_step, slope, advance)
```

Take fixed midpoint steps ending exactly at an observation time.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L185-L192)

## particula.execution.inverse_chemistry._measurement_grid

*function*

```python
def _measurement_grid(chamber, times, max_step, observations, gradient, seed)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L153-L182)

## particula.execution.inverse_chemistry._transfers

*function*

```python
def _transfers(device)
```

Count explicit input and output tensor bytes for a CUDA replay.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L195-L214)

## particula.execution.inverse_chemistry._validate_exchange

*function*

```python
def _validate_exchange(chamber, gases, particles, masses, composition)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L122-L150)

## particula.execution.inverse_chemistry._vector

*function*

```python
def _vector(values, length, label)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L59-L69)

## particula.execution.inverse_chemistry.simulate_chamber

*function*

```python
def simulate_chamber(chamber: SeededChamber, times, *, dose, rates, saturation, seed, max_step, device, observations: ObservationSeries | None = None, gradient=False) -> Trajectory
```

Integrate a fixed candidate with explicit midpoint and optional Warp AD.

Dose is mol/m³ air in sorted gas-species order. Rates follow native reaction
order, saturation (mol/m³) follows exchange order. Seed controls are total
number/m³, median diameter (m), geometric standard deviation, and first-
order deposition at 100 nm (1/s). Times and max_step are in seconds.
Predictions consist of measured gases, suspended organic kg/m³, and the
number spectrum including both tails. A NaN observation masks a channel.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/inverse_chemistry.py#L217-L379)
