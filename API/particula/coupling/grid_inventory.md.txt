# particula.coupling.grid_inventory

*module*

Conservative grid inventories and native discrete boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L1-L1)

## particula.coupling.grid_inventory.InventoryLayout

*class* · *dataclass*

```python
class InventoryLayout
```

Ordered gases, dry size classes and particle materials.

Lanes contain extensive gas kg, then represented counts by bin, then
particulate kg in bin-major/material-minor order. Native discrete boxes
store gas kg/m³, represented counts and kg per individual particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L8-L111)

### particula.coupling.grid_inventory.InventoryLayout.bins

*attribute* · *class attribute* · *instance attribute*

```python
bins: tuple = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L18-L18)

### particula.coupling.grid_inventory.InventoryLayout.gases

*attribute* · *instance attribute*

```python
gases: tuple
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L17-L17)

### particula.coupling.grid_inventory.InventoryLayout.materials

*attribute* · *class attribute* · *instance attribute*

```python
materials: tuple = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L19-L19)

### particula.coupling.grid_inventory.InventoryLayout.width

*property*

```python
width
```

Number of extensive lanes per cell.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L36-L38)

### particula.coupling.grid_inventory.InventoryLayout.__init__

*method*

```python
def __init__(gases: tuple, bins: tuple = (), materials: tuple = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L0-L0)

### particula.coupling.grid_inventory.InventoryLayout.__post_init__

*method*

```python
def __post_init__()
```

Require unique named lanes and coherent bin/material axes.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L21-L33)

### particula.coupling.grid_inventory.InventoryLayout._check_native

*method*

```python
def _check_native(grid, gas, particles)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L58-L73)

### particula.coupling.grid_inventory.InventoryLayout.pack

*method*

```python
def pack(grid, gas, particles)
```

Convert native neutral discrete boxes to extensive quantities.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L75-L86)

### particula.coupling.grid_inventory.InventoryLayout.unpack

*method*

```python
def unpack(grid, inventory, gas, particles)
```

Return detached native gas and particle copies in the same order.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L88-L111)

### particula.coupling.grid_inventory.InventoryLayout.validate

*method*

```python
def validate(inventory, n)
```

Reject nonphysical inventories including mass in an empty bin.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L40-L56)

## particula.coupling.grid_inventory.Uptake

*class* · *dataclass*

```python
class Uptake
```

First-order gas uptake (s⁻¹) into one occupied particle class.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L114-L134)

### particula.coupling.grid_inventory.Uptake.gas

*attribute* · *instance attribute*

```python
gas: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L118-L118)

### particula.coupling.grid_inventory.Uptake.material

*attribute* · *instance attribute*

```python
material: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L120-L120)

### particula.coupling.grid_inventory.Uptake.rate

*attribute* · *instance attribute*

```python
rate: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L121-L121)

### particula.coupling.grid_inventory.Uptake.size_bin

*attribute* · *instance attribute*

```python
size_bin: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L119-L119)

### particula.coupling.grid_inventory.Uptake.__init__

*method*

```python
def __init__(gas: str, size_bin: str, material: str, rate: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L0-L0)

### particula.coupling.grid_inventory.Uptake.bind

*method*

```python
def bind(layout)
```

Resolve named lanes once before transport.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_inventory.py#L123-L134)
