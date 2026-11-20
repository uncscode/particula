# Adaptive integration method decisions

The [scientific comparison](../../../studies/validation/condensation/adaptive/index.md) describes the workload, assumptions and results supporting these decisions.

## Method decisions

| Candidate | Decision and evidence |
| --- | --- |
| Existing finite-step isothermal/staggered methods | Retain explicit compatibility choices. Native refinements expose trajectory error despite good budgets; fixed substeps are not adaptive accuracy control. |
| RK3(2) | Select for the supported CPU/CUDA continuous condensation profile. Four rates, resident trial buffers on GPU, clear domain rejection and qualified observable targets. |
| RK5(4) | Retain as an assessment comparator, not another production default. Preliminary results do not demonstrate a need for a second explicit production method in the chosen accuracy range. |
| ROS2 | Retain explicit CPU stiff branch and rate/Jacobian contract. Native cases are accurate but substantially more expensive; no native GPU ROS2 claim. |
| Radau/BDF | Independent stiff/reference solvers in the assessment. Robertson screening exposes the limitations of explicit candidates; these references do not qualify a production chemistry model. |
| MPRK22 | Retain research prototype only. Linear zero-donor limits conserve mass and stay nonnegative; smooth/depletion orders approach two, but stiff transient error remains 0.0591 at 256 steps. Positivity is insufficient for selection. |
| SSPRK | Defer until an owning process supplies a demonstrated forward-Euler positivity restriction and a need beyond the selected error-controlled methods. No SSP production or measured-cost claim. |
| Spectral deferred correction | Defer: the selected case meets the declared targets with simpler methods, with no measured unresolved accuracy gap justifying additional quadrature/sweep storage. No SDC benchmark claim. |

These are engineering decisions inferred from the retained measurements and
stated support boundary. The references describe method families, not Particula
qualification: [SciPy integration methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html),
[KPP ROS2 coefficients](https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html),
[modified Patankar order conditions](https://arxiv.org/abs/1702.04589), and
[spectral deferred correction convergence](https://arxiv.org/abs/1706.06245).
