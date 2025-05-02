# Accelerating Python

Python’s ease-of-use and rich ecosystem make it ideal for scientific computing—but pure-Python loops and numeric operations can be orders of magnitude slower than optimized native code. To bridge this gap, projects routinely **offload** performance-critical kernels to compiled libraries rather than rewriting entire applications in lower-level languages.  

This pattern is nothing new: classic Fortran and C programs have long relied on optimized BLAS/LAPACK routines, FFT libraries, or custom C modules to handle the heavy lifting. In the same spirit, we aim to explore the **right approach** for accelerating our own Python codebase, weighing trade-offs in portability, maintenance, and raw speed.

---

## What We’ll Explore

### **C++ Extensions**  
   - Write custom C++ modules (via `pybind11`, CPython C-API, or `cffi`) for maximum control and access to state-of-the-art C++ libraries.  
   - Pros: Full language power, mature toolchains, seamless integration with existing C++ code.  
   - Cons: Steeper learning curve, manual memory management, more boilerplate.
   - Example: Not yet available.

### **Taichi**  
   - A data-oriented language that compiles Python-like kernels into optimized vectorized code for CPU/GPU.  
   - Pros: High-level syntax, automatic parallelization, built-in profiler and GPU backends.  
   - Cons: Requires learning Taichi’s data-model and kernel abstraction.
   - Example: [Taichi Exploration](Details/Taichi_Exploration.ipynb) notebook.

### **NVIDIA Warp**  
   - A Python API for data-parallel C-style kernels, especially suited for physics and graphics simulations.  
   - Pros: Easy dispatch to multicore CPUs or CUDA GPUs, familiar C-like syntax.  
   - Cons: Niche use cases, less mature ecosystem than CUDA or Taichi.
   - Example: Not yet available.

### **Cython (PyPy)**  
   - A superset of Python that compiles to C, enabling static type declarations and direct C-API calls.  
   - Pros: Incremental adoption via `*.pyx` files, excellent control over types and memory layout.  
   - Cons: Requires writing Cython annotations, manual build configuration.
   - Example: Not yet available.

### **Numba**  
   - A JIT compiler that decorates Python functions for high-performance machine code (LLVM).  
   - Pros: Minimal code changes for simple numeric loops, transparent parallelization options.  
   - Cons: Limited coverage of the full NumPy API—features like `np.any` or `isinstance` checks often fail, so non-trivial functions must be rewritten from scratch, just like with other options.
   - Example: This was explored (no example notebook) and found not to be a simple drop-in solution; due to incomplete `numpy` coverage. So if we needed to re-write the code anyway, we might as well use one of the other options.
---

## How We Envision This Working

Our vision is to have a simple enable statement for `particula` that allows users to choose the acceleration method. For example, a user could run the following command in their notebook:

```python
import particula as par

par.use_backend("taichi")  # or "warp", "cpp", etc.
```

The rest of the builder and other function calls would remain unchanged, and the user would be able to run their code as usual. The only difference would be that the kernels would be compiled to the selected backend (e.g., Taichi, Warp, etc.) and run there instead of in Python.

See one option for how this could be done in the [One-Line Vision](Details/One-Line_Vision.md) document.

---

## Why This Matters

- **Performance**: Offloading compute-intensive work can yield **10×–10,000×** speedups.  
- **Maintainability**: Choosing the right tool lets you keep most of your code in Python, while isolating optimized kernels.  
- **Portability**: Leveraging standard libraries ensures compatibility across platforms (Linux, Windows, macOS).  

In the notebooks under `Details/` on the left, we’ll benchmark each approach on real aerosol-simulation kernels, discuss integration strategies, and surface best practices.
