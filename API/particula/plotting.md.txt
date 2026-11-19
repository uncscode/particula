# particula.plotting

*module*

Opt-in Matplotlib tools for paper-sized Particula figures.

Use `plt.style.context("particula.default")` around figure construction.

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L1-L1)

## particula.plotting.COLORS

*attribute* · *module attribute*

```python
COLORS = {'teal': '#08786C', 'rust': '#C2410C', 'indigo': '#4338CA', 'berry': '#BE185D'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L14-L19)

## particula.plotting._RAMPS

*attribute* · *module attribute*

```python
_RAMPS = {'magnitude': ['#f0fdfa', '#9edfd3', '#45a89b', '#08786c', '#073d39'], 'difference': ['#823216', '#db906b', '#faf8f3', '#6cbcaf', '#00534d']}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L21-L24)

## particula.plotting._SIZES

*attribute* · *module attribute*

```python
_SIZES = {'square': (3.0, 3.0), 'wide': (5.0, 3.5)}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L20-L20)

## particula.plotting._PaperLayout

*class*

```python
class _PaperLayout(ConstrainedLayoutEngine)
```

Bases: `ConstrainedLayoutEngine`

Keep the panel sizing contract with its native layout engine.

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L63-L77)

### particula.plotting._PaperLayout.finished

*attribute* · *instance attribute*

```python
finished = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L70-L70)

### particula.plotting._PaperLayout.grid

*attribute* · *instance attribute*

```python
grid = np.array([ncols, nrows])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L69-L69)

### particula.plotting._PaperLayout.gutters

*attribute* · *instance attribute*

```python
gutters = np.zeros(ncols)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L72-L72)

### particula.plotting._PaperLayout.outside

*attribute* · *instance attribute*

```python
outside = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L71-L71)

### particula.plotting._PaperLayout.panel

*attribute* · *instance attribute*

```python
panel = np.array([width, height])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L68-L68)

### particula.plotting._PaperLayout.__init__

*method*

```python
def __init__(width, height, nrows, ncols)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L66-L72)

### particula.plotting._PaperLayout.execute

*method*

```python
def execute(fig)
```

Preserve solved positions when redrawing for PNG, PDF or display.

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L74-L77)

## particula.plotting._check_panel_aspect

*function*

```python
def _check_panel_aspect(ax, panel)
```

Reject incompatible data and box aspect constraints before resizing.

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L117-L126)

## particula.plotting._draw_for_fit

*function*

```python
def _draw_for_fit(fig)
```

Resize a collapsed layout before measuring the data panels.

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L129-L151)

## particula.plotting.colormap

*function*

```python
def colormap(name)
```

Return a fresh sequential magnitude or diverging difference colormap.

Use with Matplotlib's own normalization, levels and colorbar functions.

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L27-L34)

## particula.plotting.fit

*function*

```python
def fit(fig)
```

Size the canvas to labels, legends and colorbars at fixed panel size.

Call after creating a figure with subplots() and adding all artists.
Read the finished canvas dimensions in inches with fig.get_size_inches().
The grid must have equally sized data axes. For images, use aspect='auto'
unless the data aspect agrees with the requested physical panel shape.

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L154-L198)

## particula.plotting.legend

*function*

```python
def legend(ax, *, placement='auto', **kwargs)
```

Place a measured legend inside the axes or in a right-hand gutter.

Auto uses at most 40% of panel width and 35% of height inside. Right
legends add columns if necessary. Pass other legend options by keyword.
Call fit() after the final legend, axis label and colorbar are present.

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L80-L114)

## particula.plotting.subplots

*function*

```python
def subplots(nrows=1, ncols=1, *, size='square', **kwargs)
```

Create an equal-panel grid. Finish with fit() after adding artists.

**Parameters**

- `nrows` (default: `1`) — Number of rows.
- `ncols` (default: `1`) — Number of columns.
- `size` (default: `'square'`) — Physical size of each data axes in inches. Square is 3 by 3 and wide is 5 by 3.5.
- `**kwargs` (default: `{}`) — Matplotlib subplot options, e.g. sharex or squeeze.

**Returns**

- A Matplotlib figure and its axes with ordinary subplots semantics.

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/__init__.py#L37-L60)
