# Rates

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Rates

> Auto-generated documentation for [particula.rates](https://github.com/uncscode/particula/blob/main/particula/rates.py) module.

## Rates

[Show source in rates.py:12](https://github.com/uncscode/particula/blob/main/particula/rates.py#L12)

The class to calculate the rates

#### Signature

```python
class Rates:
    def __init__(self, particle=None, lazy=True): ...
```

### Rates()._coag_loss_gain

[Show source in rates.py:40](https://github.com/uncscode/particula/blob/main/particula/rates.py#L40)

get both loss and gain

#### Signature

```python
def _coag_loss_gain(self): ...
```

### Rates().coagulation_gain

[Show source in rates.py:56](https://github.com/uncscode/particula/blob/main/particula/rates.py#L56)

get coagulation gain rate

#### Signature

```python
def coagulation_gain(self): ...
```

### Rates().coagulation_loss

[Show source in rates.py:48](https://github.com/uncscode/particula/blob/main/particula/rates.py#L48)

get the coagulation loss rate

#### Signature

```python
def coagulation_loss(self): ...
```

### Rates().coagulation_rate

[Show source in rates.py:64](https://github.com/uncscode/particula/blob/main/particula/rates.py#L64)

get the coagulation rate by summing the loss and gain rates

#### Signature

```python
def coagulation_rate(self): ...
```

### Rates().condensation_growth_rate

[Show source in rates.py:74](https://github.com/uncscode/particula/blob/main/particula/rates.py#L74)

condensation rate

#### Signature

```python
def condensation_growth_rate(self): ...
```

### Rates().condensation_growth_speed

[Show source in rates.py:69](https://github.com/uncscode/particula/blob/main/particula/rates.py#L69)

condensation speed

#### Signature

```python
def condensation_growth_speed(self): ...
```

### Rates().dilution_rate

[Show source in rates.py:94](https://github.com/uncscode/particula/blob/main/particula/rates.py#L94)

dilution rate

#### Signature

```python
def dilution_rate(self): ...
```

### Rates().nucleation_rate

[Show source in rates.py:85](https://github.com/uncscode/particula/blob/main/particula/rates.py#L85)

nucleation rate

#### Signature

```python
def nucleation_rate(self): ...
```

### Rates().sum_rates

[Show source in rates.py:107](https://github.com/uncscode/particula/blob/main/particula/rates.py#L107)

Sum rates, with options to disable individual rate terms.

#### Arguments

----------
coagulation : bool, optional
    does the coagulation calcuation, by default True
condensation : bool, optional
    does the condensation calculation, by default True
nucleation : bool, optional
    does the nucleation calculation, by default True
dilution : bool, optional
    does the dilution calculation, by default False
wall_loss : bool, optional
    does the wall loss calculation, by default False

#### Signature

```python
def sum_rates(
    self,
    coagulation=True,
    condensation=True,
    nucleation=True,
    dilution=False,
    wall_loss=False,
): ...
```

### Rates().wall_loss_rate

[Show source in rates.py:101](https://github.com/uncscode/particula/blob/main/particula/rates.py#L101)

wall loss rate

#### Signature

```python
def wall_loss_rate(self): ...
```
