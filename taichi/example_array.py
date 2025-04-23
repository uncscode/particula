import taichi as ti
import numpy as np

ti.init()

a = np.zeros((500, 500), dtype=np.float64)


@ti.kernel
def test(a: ti.types.ndarray()):
    for i in range(a.shape[0]):  # a parallel for loop
        for j in range(a.shape[1]):
            a[i, j] = i + j


test(a)
print(a)
