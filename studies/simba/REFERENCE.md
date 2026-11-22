# Reference observations for the SiMBA reproductions

The [published SiMBA cases](https://doi.org/10.1039/D5SC01473E) use
synthetic concentration measurements. Compare the code-generated reference
profiles to the archived CSVs before comparing network rankings. Run from the
Particula checkout with NumPy and SciPy installed.

```bash
curl -fLsS https://codeload.github.com/OptiMaL-PSE-Lab/auto_react_mech_construct/tar.gz/8f0d16357193d79ea73f517540571fc6997f0941 -o /tmp/simba-source.tar.gz
python studies/simba/reference_inputs.py /tmp/simba-source.tar.gz
```

The [audit](reference_inputs.json) records each input checksum and the largest
absolute discrepancy across 30 observation times and all measured species.
All 15 profiles reproduce within 1e-7 in the original concentration units.

Two details affect comparisons with the published cases.

* The hypothetical generator writes a dimerization flux proportional to A
  squared but consumes only one A per B produced. Applying the stated
  stoichiometry of two A per B changes a measured-species prediction by up
  to 2.45 concentration units at the original rates. Preserve that difference
  when comparing a native mass-balanced model with the archived profiles.
* The HMF dataset associated with the paper's empirical rate law is
  `exp_data_fruc_HMF2`. Experiment four's archived trajectory reproduces
  from the initial concentrations [6, 0, 0]. The generating script on the
  pinned revision specifies [4, 0, 1]. The other four experiments reproduce
  using their scripted initial conditions.

[Hypothetical source](https://github.com/OptiMaL-PSE-Lab/auto_react_mech_construct/blob/8f0d16357193d79ea73f517540571fc6997f0941/hypothetical_network.py)
· [Fructose source](https://github.com/OptiMaL-PSE-Lab/auto_react_mech_construct/blob/8f0d16357193d79ea73f517540571fc6997f0941/fructose_to_HMF2.py)
