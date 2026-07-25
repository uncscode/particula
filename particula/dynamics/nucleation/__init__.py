"""CPU-only nucleation potential-rate strategy implementations.

The concrete strategy module is intentionally not re-exported through
``particula.dynamics``. Its strategies calculate potential formation rates;
they do not create particles or mutate gas, particle, or slot state.
"""
