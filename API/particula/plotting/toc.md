# particula.plotting.toc

*module*

Corner branding for graphical abstracts using the particula.toc style.

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/toc.py#L1-L1)

## particula.plotting.toc.add_toc_logo

*function*

```python
def add_toc_logo(fig, *, corner='bottom right')
```

Place the Particula dots inside the figure margin.

**Parameters**

- `fig` — Matplotlib figure, normally three inches square.
- `corner` (default: `'bottom right'`) — "top left", "top right", "bottom left", or "bottom right".

**Returns**

- The logo axes, excluded from the scientific layout.

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/toc.py#L13-L62)

## particula.plotting.toc.save_toc_svg

*function*

```python
def save_toc_svg(fig, path, *, title, description)
```

Save the full TOC canvas as a reproducible SVG with accessible metadata.

**Parameters**

- `fig` — Completed Matplotlib figure.
- `path` — Destination SVG file.
- `title` — Short title for the standalone artwork.
- `description` — Accessible description of the scientific comparison.

[View source](https://github.com/uncscode/particula/blob/main/particula/plotting/toc.py#L65-L88)
