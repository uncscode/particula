# particula.optics.retrieval

*module*

Optical closure with prescribed size distributions and measurement errors.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L1-L1)

## particula.optics.retrieval.CHANNELS

*attribute* · *module attribute*

```python
CHANNELS = ('extinction', 'scattering', 'absorption')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L16-L16)

## particula.optics.retrieval.EffectiveIndexModel

*class*

```python
class EffectiveIndexModel(_OpticalModel)
```

Bases: `particula.optics.retrieval._OpticalModel`

Fit a common complex index, with optional water uptake.

With wet sizes use the default configuration for an index at the optical
RH. With dry sizes supply relative_humidity and either a fixed kappa or
leave kappa open for fitting. Alternatively supply a measured diameter
growth factor to determine water volume directly. Supply sizing RH if
the measured distribution retains water at a known equilibrium RH.
This model shares one dry index across all wavelengths and RH groups.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L158-L254)

### particula.optics.retrieval.EffectiveIndexModel._kappa

*attribute* · *instance attribute*

```python
_kappa = kappa
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L191-L191)

### particula.optics.retrieval.EffectiveIndexModel._ratio

*attribute* · *instance attribute*

```python
_ratio = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L190-L190)

### particula.optics.retrieval.EffectiveIndexModel._rh

*attribute* · *instance attribute*

```python
_rh = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L189-L189)

### particula.optics.retrieval.EffectiveIndexModel._sizing_rh

*attribute* · *instance attribute*

```python
_sizing_rh = np.zeros(groups)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L205-L205)

### particula.optics.retrieval.EffectiveIndexModel.grid

*attribute* · *instance attribute*

```python
grid = grid
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L182-L182)

### particula.optics.retrieval.EffectiveIndexModel.kelvin

*attribute* · *instance attribute*

```python
kelvin = _freeze(np.broadcast_to(_real(kelvin_length), (groups)))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L188-L188)

### particula.optics.retrieval.EffectiveIndexModel.parameters

*attribute* · *instance attribute*

```python
parameters = ('n', 'k', 'kappa') if self._rh is not None and self._ratio is None and kappa is None else ('n', 'k')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L217-L221)

### particula.optics.retrieval.EffectiveIndexModel.water_index

*attribute* · *instance attribute*

```python
water_index = _freeze(np.broadcast_to(_physical_index(water_index), (groups)))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L185-L187)

### particula.optics.retrieval.EffectiveIndexModel.wavelength

*attribute* · *instance attribute*

```python
wavelength = grid.wavelength
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L183-L183)

### particula.optics.retrieval.EffectiveIndexModel.__init__

*method*

```python
def __init__(grid, *, relative_humidity=None, sizing_relative_humidity=None, kappa=None, growth_factor=None, kelvin_length=1.05e-09, water_index=1.333, device='numpy')
```

Prepare the measured grid and a CPU or explicit Warp backend.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L169-L222)

### particula.optics.retrieval.EffectiveIndexModel._parameters

*method*

```python
def _parameters(values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L224-L228)

### particula.optics.retrieval.EffectiveIndexModel._population

*method*

```python
def _population(values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L230-L254)

### particula.optics.retrieval.EffectiveIndexModel._eval

*method*

```python
def _eval(parameters, seeds)
```

Inherited from: `particula.optics.retrieval._OpticalModel`

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L131-L137)

### particula.optics.retrieval.EffectiveIndexModel._setup

*method*

```python
def _setup(wavelength, shape, backend)
```

Inherited from: `particula.optics.retrieval._OpticalModel`

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L118-L125)

### particula.optics.retrieval.EffectiveIndexModel.predictions

*method*

```python
def predictions(parameters)
```

Inherited from: `particula.optics.retrieval._OpticalModel`

Return bulk extinction, scattering and absorption [m⁻¹].

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L127-L129)

## particula.optics.retrieval.MixingStateModel

*class*

```python
class MixingStateModel(_OpticalModel)
```

Bases: `particula.optics.retrieval._OpticalModel`

Prescribed composition with internal/external number fractions by size.

Each dry bin has a monodisperse internal population and size-matched
pure-species external populations. The endpoints have the same dry
number density and per-bin species inventories. Volume fractions may
be uniform across bins or specified per group and bin.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L257-L367)

### particula.optics.retrieval.MixingStateModel._band

*attribute* · *instance attribute*

```python
_band = _freeze(np.zeros(grid.radius.shape, dtype=np.int32) if dry_boundary is None else 2 * grid.radius >= dry_boundary.astype(np.int32))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L330-L334)

### particula.optics.retrieval.MixingStateModel._endpoint_density

*attribute* · *instance attribute*

```python
_endpoint_density = _freeze(grid.number_density[..., None] * np.concatenate((np.ones_like(fraction[..., :1]), fraction), axis=-1).reshape(shape))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L322-L329)

### particula.optics.retrieval.MixingStateModel._index

*attribute* · *instance attribute*

```python
_index = _freeze(wet_index.reshape(shape))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L321-L321)

### particula.optics.retrieval.MixingStateModel._radius

*attribute* · *instance attribute*

```python
_radius = _freeze(wet.reshape(shape))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L320-L320)

### particula.optics.retrieval.MixingStateModel._subpopulations

*attribute* · *instance attribute*

```python
_subpopulations = species + 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L318-L318)

### particula.optics.retrieval.MixingStateModel.grid

*attribute* · *instance attribute*

```python
grid = grid
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L316-L316)

### particula.optics.retrieval.MixingStateModel.parameters

*attribute* · *instance attribute*

```python
parameters = ('internal_fraction')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L335-L335)

### particula.optics.retrieval.MixingStateModel.wavelength

*attribute* · *instance attribute*

```python
wavelength = grid.wavelength
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L317-L317)

### particula.optics.retrieval.MixingStateModel.__init__

*method*

```python
def __init__(grid, *, volume_fractions, species_index, species_kappa, relative_humidity, water_index=1.333, kelvin_length=1.05e-09, dry_boundary=None, device='numpy')
```

Prepare optics inputs for both endpoint populations.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L266-L349)

### particula.optics.retrieval.MixingStateModel._parameters

*method*

```python
def _parameters(values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L351-L355)

### particula.optics.retrieval.MixingStateModel._population

*method*

```python
def _population(values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L357-L367)

### particula.optics.retrieval.MixingStateModel._eval

*method*

```python
def _eval(parameters, seeds)
```

Inherited from: `particula.optics.retrieval._OpticalModel`

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L131-L137)

### particula.optics.retrieval.MixingStateModel._setup

*method*

```python
def _setup(wavelength, shape, backend)
```

Inherited from: `particula.optics.retrieval._OpticalModel`

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L118-L125)

### particula.optics.retrieval.MixingStateModel.predictions

*method*

```python
def predictions(parameters)
```

Inherited from: `particula.optics.retrieval._OpticalModel`

Return bulk extinction, scattering and absorption [m⁻¹].

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L127-L129)

## particula.optics.retrieval.OpticalFit

*class* · *dataclass*

```python
class OpticalFit
```

Best bounded fit, with sensitivity diagnostics in measurement space.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L464-L479)

### particula.optics.retrieval.OpticalFit.active_bounds

*attribute* · *instance attribute*

```python
active_bounds: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L475-L475)

### particula.optics.retrieval.OpticalFit.backward_evaluations

*attribute* · *instance attribute*

```python
backward_evaluations: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L479-L479)

### particula.optics.retrieval.OpticalFit.chi_square

*attribute* · *instance attribute*

```python
chi_square: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L476-L476)

### particula.optics.retrieval.OpticalFit.forward_evaluations

*attribute* · *instance attribute*

```python
forward_evaluations: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L478-L478)

### particula.optics.retrieval.OpticalFit.names

*attribute* · *instance attribute*

```python
names: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L468-L468)

### particula.optics.retrieval.OpticalFit.parameters

*attribute* · *instance attribute*

```python
parameters: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L469-L469)

### particula.optics.retrieval.OpticalFit.predictions

*attribute* · *instance attribute*

```python
predictions: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L470-L470)

### particula.optics.retrieval.OpticalFit.residual

*attribute* · *instance attribute*

```python
residual: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L471-L471)

### particula.optics.retrieval.OpticalFit.sensitivity

*attribute* · *instance attribute*

```python
sensitivity: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L473-L473)

### particula.optics.retrieval.OpticalFit.singular_values

*attribute* · *instance attribute*

```python
singular_values: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L474-L474)

### particula.optics.retrieval.OpticalFit.success

*attribute* · *instance attribute*

```python
success: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L477-L477)

### particula.optics.retrieval.OpticalFit.whitened_residual

*attribute* · *instance attribute*

```python
whitened_residual: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L472-L472)

### particula.optics.retrieval.OpticalFit.__init__

*method*

```python
def __init__(names: tuple[str, ...], parameters: np.ndarray, predictions: np.ndarray, residual: np.ndarray, whitened_residual: np.ndarray, sensitivity: np.ndarray, singular_values: np.ndarray, active_bounds: np.ndarray, chi_square: float, success: bool, forward_evaluations: int, backward_evaluations: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L0-L0)

## particula.optics.retrieval.OpticalGrid

*class* · *dataclass*

```python
class OpticalGrid
```

One particle spectrum per optical wavelength and measurement window.

Radius and number_density have shape (group, bin). A one-dimensional
spectrum broadcasts over groups. Radii and vacuum wavelengths are in
metres and number densities are integrated counts per bin in m⁻³.
Groups can repeat a wavelength at different humidity or time conditions.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L45-L105)

### particula.optics.retrieval.OpticalGrid.number_density

*attribute* · *instance attribute*

```python
number_density: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L56-L56)

### particula.optics.retrieval.OpticalGrid.radius

*attribute* · *instance attribute*

```python
radius: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L55-L55)

### particula.optics.retrieval.OpticalGrid.wavelength

*attribute* · *instance attribute*

```python
wavelength: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L57-L57)

### particula.optics.retrieval.OpticalGrid.__init__

*method*

```python
def __init__(radius: np.ndarray, number_density: np.ndarray, wavelength: np.ndarray) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L0-L0)

### particula.optics.retrieval.OpticalGrid.__post_init__

*method*

```python
def __post_init__()
```

Validate and detach fixed measurement coordinates.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L59-L79)

### particula.optics.retrieval.OpticalGrid.from_diameter_bins

*method* · *classmethod*

```python
def from_diameter_bins(edges, values, wavelength, *, measure)
```

Build a log-midpoint grid from mobility-diameter edges in metres.

Use `bin_density` for number per m³ already integrated across the
bin or `dN_dlog10D` for number per m³ per decade. The spectrum
must already use the same volume basis as the optical coefficients.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L81-L105)

## particula.optics.retrieval.OpticalObservations

*class*

```python
class OpticalObservations
```

Calibrated measurements projected from (group, E/S/A) coefficients.

Response has shape (observation, group, 3). Use it to select a channel
or average predicted spectra across measured time or spectral windows.
Use only measured channels with an independent uncertainty model.
Observations and standard deviations have units m⁻¹. Covariance entries
have units m⁻². Include environmental and sizing uncertainty through
sensitivity cases or a supplied covariance.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L384-L461)

### particula.optics.retrieval.OpticalObservations._factor

*attribute* · *instance attribute*

```python
_factor = _freeze(factor)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L435-L435)

### particula.optics.retrieval.OpticalObservations.response

*attribute* · *instance attribute*

```python
response = _freeze(response)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L433-L433)

### particula.optics.retrieval.OpticalObservations.values

*attribute* · *instance attribute*

```python
values = _freeze(values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L434-L434)

### particula.optics.retrieval.OpticalObservations.__init__

*method*

```python
def __init__(response, values, *, standard_deviation=None, covariance=None)
```

Validate the response and factor its measurement covariance.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L395-L435)

### particula.optics.retrieval.OpticalObservations._whiten

*method*

```python
def _whiten(residual)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L460-L461)

### particula.optics.retrieval.OpticalObservations.channels

*method* · *classmethod*

```python
def channels(groups, selectors, values, **uncertainty)
```

Construct observations from (group index, channel name) selectors.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L437-L451)

### particula.optics.retrieval.OpticalObservations.project

*method*

```python
def project(coefficients)
```

Apply the calibrated linear response in measurement coordinates.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L453-L458)

## particula.optics.retrieval._OpticalModel

*class*

```python
class _OpticalModel
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L117-L137)

### particula.optics.retrieval._OpticalModel._eval

*method*

```python
def _eval(parameters, seeds)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L131-L137)

### particula.optics.retrieval._OpticalModel._setup

*method*

```python
def _setup(wavelength, shape, backend)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L118-L125)

### particula.optics.retrieval._OpticalModel.predictions

*method*

```python
def predictions(parameters)
```

Return bulk extinction, scattering and absorption [m⁻¹].

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L127-L129)

## particula.optics.retrieval._Residual

*class*

```python
class _Residual
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L482-L521)

### particula.optics.retrieval._Residual.autodiff

*attribute* · *instance attribute*

```python
autodiff = method == 'autodiff'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L492-L492)

### particula.optics.retrieval._Residual.backwards

*attribute* · *instance attribute*

```python
backwards = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L497-L497)

### particula.optics.retrieval._Residual.forwards

*attribute* · *instance attribute*

```python
forwards = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L496-L496)

### particula.optics.retrieval._Residual.jacobian

*attribute* · *instance attribute*

```python
jacobian = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L495-L495)

### particula.optics.retrieval._Residual.model

*attribute* · *instance attribute*

```python
model = model
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L490-L490)

### particula.optics.retrieval._Residual.observations

*attribute* · *instance attribute*

```python
observations = observations
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L491-L491)

### particula.optics.retrieval._Residual.point

*attribute* · *instance attribute*

```python
point = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L493-L493)

### particula.optics.retrieval._Residual.values

*attribute* · *instance attribute*

```python
values = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L494-L494)

### particula.optics.retrieval._Residual.__init__

*method*

```python
def __init__(model, observations, method)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L483-L497)

### particula.optics.retrieval._Residual.calculate

*method*

```python
def calculate(point, *, jacobian=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L499-L513)

### particula.optics.retrieval._Residual.derivative

*method*

```python
def derivative(point)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L519-L521)

### particula.optics.retrieval._Residual.residual

*method*

```python
def residual(point)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L515-L517)

## particula.optics.retrieval._bulk

*function*

```python
def _bulk(radius, index, density, wave)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L108-L114)

## particula.optics.retrieval._fit_inputs

*function*

```python
def _fit_inputs(model, bounds, starts, method)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L524-L542)

## particula.optics.retrieval._freeze

*function*

```python
def _freeze(value)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L28-L31)

## particula.optics.retrieval._humidity

*function*

```python
def _humidity(value, groups)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L140-L144)

## particula.optics.retrieval._measured_water

*function*

```python
def _measured_water(growth_factor, shape, kappa)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L147-L155)

## particula.optics.retrieval._physical_index

*function*

```python
def _physical_index(value)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L34-L42)

## particula.optics.retrieval._real

*function*

```python
def _real(value)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L19-L25)

## particula.optics.retrieval.dry_volume_fractions

*function*

```python
def dry_volume_fractions(mass, density)
```

Convert measured dry species mass and density to volume fractions.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L370-L381)

## particula.optics.retrieval.fit_optical

*function*

```python
def fit_optical(model, observations, *, bounds, starts, method='autodiff', max_nfev=None)
```

Fit optically constrained parameters from one or more bounded starts.

Sensitivity is the whitened Jacobian per parameter in its native units.
Singular values include structural zeros when there are fewer independent
observations than unknowns. Inspect profiles and held-out predictions
before treating a least-squares solution as a unique retrieval.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L545-L596)

## particula.optics.retrieval.profile_optical

*function*

```python
def profile_optical(model, observations, fit, parameter, coordinates, *, bounds, method='autodiff', delta_chi_square=3.841459)
```

Profile a fitted parameter by optimizing all others at fixed values.

Returns the grid, profiled chi square, and the intervals resolved by the
grid and threshold. A component touching either search bound is limited
by that bound. The default delta is the asymptotic 95% cutoff for one
parameter under a regular Gaussian measurement model.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/retrieval.py#L599-L686)
