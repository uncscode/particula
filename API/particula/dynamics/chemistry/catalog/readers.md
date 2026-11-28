# particula.dynamics.chemistry.catalog.readers

*module*

Read source mechanisms as data; preserve source expressions and locations.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L1-L1)

## particula.dynamics.chemistry.catalog.readers.NUMBER

*attribute* · *module attribute*

```python
NUMBER = '[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eEdD][+-]?\\d+)?'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L8-L8)

## particula.dynamics.chemistry.catalog.readers.SOURCES

*attribute* · *module attribute*

```python
SOURCES = ('waccm', 'e3sm', 'mcm', 'wrf-chem', 'cmaq', 'camx', 'pmcamx')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L9-L9)

## particula.dynamics.chemistry.catalog.readers._KPP_COMMENTS

*attribute* · *module attribute*

```python
_KPP_COMMENTS = re.compile('\\{[^}]*\\}|//[^\\n]*|![^\\n]*')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L235-L235)

## particula.dynamics.chemistry.catalog.readers._KPP_SECTION

*attribute* · *module attribute*

```python
_KPP_SECTION = re.compile('^[ \\t]*#(\\w+)\\b', re.M)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L236-L236)

## particula.dynamics.chemistry.catalog.readers.Entry

*class* · *dataclass*

```python
class Entry
```

One source channel; kinetics are (A, C, b, Tref) for exp(C/T).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L12-L46)

### particula.dynamics.chemistry.catalog.readers.Entry.equation

*attribute* · *instance attribute*

```python
equation: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L18-L18)

### particula.dynamics.chemistry.catalog.readers.Entry.expression

*attribute* · *instance attribute*

```python
expression: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L19-L19)

### particula.dynamics.chemistry.catalog.readers.Entry.kinetics

*attribute* · *instance attribute*

```python
kinetics: tuple | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L22-L22)

### particula.dynamics.chemistry.catalog.readers.Entry.label

*attribute* · *instance attribute*

```python
label: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L16-L16)

### particula.dynamics.chemistry.catalog.readers.Entry.line

*attribute* · *instance attribute*

```python
line: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L17-L17)

### particula.dynamics.chemistry.catalog.readers.Entry.metadata

*attribute* · *class attribute* · *instance attribute*

```python
metadata: dict = field(default_factory=dict)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L24-L24)

### particula.dynamics.chemistry.catalog.readers.Entry.products

*attribute* · *instance attribute*

```python
products: dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L21-L21)

### particula.dynamics.chemistry.catalog.readers.Entry.reactants

*attribute* · *instance attribute*

```python
reactants: dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L20-L20)

### particula.dynamics.chemistry.catalog.readers.Entry.reason

*attribute* · *class attribute* · *instance attribute*

```python
reason: str = ''
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L23-L23)

### particula.dynamics.chemistry.catalog.readers.Entry.__init__

*method*

```python
def __init__(label: str, line: int, equation: str, expression: str, reactants: dict, products: dict, kinetics: tuple | None, reason: str = '', metadata: dict = dict()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L0-L0)

### particula.dynamics.chemistry.catalog.readers.Entry.record

*method*

```python
def record()
```

Return a JSON-compatible source record.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L26-L46)

## particula.dynamics.chemistry.catalog.readers._cmaq_equations

*function*

```python
def _cmaq_equations(text)
```

Yield labeled semicolon-terminated statements, and continuations.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L151-L181)

## particula.dynamics.chemistry.catalog.readers._expression

*function*

```python
def _expression(rate, source)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L184-L208)

## particula.dynamics.chemistry.catalog.readers._fixed_species

*function*

```python
def _fixed_species(source, clean, companions)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L358-L385)

## particula.dynamics.chemistry.catalog.readers._kpp_equations

*function*

```python
def _kpp_equations(text, clean, source)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L252-L304)

## particula.dynamics.chemistry.catalog.readers._kpp_sections

*function*

```python
def _kpp_sections(clean)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L245-L249)

## particula.dynamics.chemistry.catalog.readers._kpp_settings

*function*

```python
def _kpp_settings(content, fixed)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L334-L355)

## particula.dynamics.chemistry.catalog.readers._kpp_text

*function*

```python
def _kpp_text(text)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L239-L242)

## particula.dynamics.chemistry.catalog.readers._mcm_elements

*function*

```python
def _mcm_elements(text)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L315-L331)

## particula.dynamics.chemistry.catalog.readers._mcm_rates

*function*

```python
def _mcm_rates(companions)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L211-L220)

## particula.dynamics.chemistry.catalog.readers._mozart_rate

*function*

```python
def _mozart_rate(section, rate)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L96-L108)

## particula.dynamics.chemistry.catalog.readers._resolve

*function*

```python
def _resolve(rate, source, definitions)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L223-L232)

## particula.dynamics.chemistry.catalog.readers._terms

*function*

```python
def _terms(side)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L54-L69)

## particula.dynamics.chemistry.catalog.readers.channel

*function*

```python
def channel(label, line, equation, rate, kinetics=None, reason='', **metadata)
```

Keep the original equation even when its terms need a new dialect.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L72-L93)

## particula.dynamics.chemistry.catalog.readers.labeled

*function*

```python
def labeled(text, source, companions=())
```

Read York/WRF KPP equations or CMAQ MECHDEF in molecular units.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L388-L419)

## particula.dynamics.chemistry.catalog.readers.mozart

*function*

```python
def mozart(text)
```

Read MOZART preprocessor photolysis, kinetic and fixed-species blocks.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L111-L148)

## particula.dynamics.chemistry.catalog.readers.number

*function*

```python
def number(value)
```

Convert atmospheric-model Fortran numeric literals.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L49-L51)

## particula.dynamics.chemistry.catalog.readers.ro2_members

*function*

```python
def ro2_members(text)
```

Read the explicit KPP radical pool and its continuation lines.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/catalog/readers.py#L307-L312)
