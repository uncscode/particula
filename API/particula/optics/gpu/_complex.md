# particula.optics.gpu._complex

*module*

Float64 complex arithmetic for optical Warp kernels.

Complex values are represented by (real, imaginary) wp.vec2d components.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_complex.py#L1-L1)

## particula.optics.gpu._complex.divide

*function*

```python
def divide(a: wp.vec2d, b: wp.vec2d) -> wp.vec2d
```

Return a/b using scaled division to avoid squaring large components.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_complex.py#L17-L36)

## particula.optics.gpu._complex.multiply

*function*

```python
def multiply(a: wp.vec2d, b: wp.vec2d) -> wp.vec2d
```

Return the complex product.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_complex.py#L11-L14)

## particula.optics.gpu._complex.norm_squared

*function*

```python
def norm_squared(a: wp.vec2d) -> wp.float64
```

Return the squared complex modulus.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_complex.py#L39-L42)

## particula.optics.gpu._complex.real_product_conjugate

*function*

```python
def real_product_conjugate(a: wp.vec2d, b: wp.vec2d) -> wp.float64
```

Return Re(a conjugate(b)).

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_complex.py#L45-L48)
