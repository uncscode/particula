# particula.gpu.dynamics.ion_pair_data

*module*

Derived ion-pair rates shared by the ordinary majorant and sampler paths.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_pair_data.py#L1-L1)

## particula.gpu.dynamics.ion_pair_data.IonPairData

*class*

```python
class IonPairData
```

Derived ion data; disabled instances need no device allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_pair_data.py#L8-L15)

### particula.gpu.dynamics.ion_pair_data.IonPairData.attachment

*attribute* · *instance attribute*

```python
attachment: wp.array3d(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_pair_data.py#L14-L14)

### particula.gpu.dynamics.ion_pair_data.IonPairData.enabled

*attribute* · *instance attribute*

```python
enabled: wp.int32
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_pair_data.py#L12-L12)

### particula.gpu.dynamics.ion_pair_data.IonPairData.kinds

*attribute* · *instance attribute*

```python
kinds: wp.array2d(dtype=wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_pair_data.py#L13-L13)

### particula.gpu.dynamics.ion_pair_data.IonPairData.recombination

*attribute* · *instance attribute*

```python
recombination: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_pair_data.py#L15-L15)

## particula.gpu.dynamics.ion_pair_data.replace_ion_pair_rate

*function*

```python
def replace_ion_pair_rate(data: IonPairData, box: wp.int32, first: wp.int32, second: wp.int32, aerosol_rate: wp.float64) -> wp.float64
```

Select one physical law after setup validates rates.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_pair_data.py#L18-L48)
