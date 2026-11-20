Galloway et al. (2009), Atmospheric Chemistry and Physics 9, 3331–3345.
https://doi.org/10.5194/acp-9-3331-2009
Figure 6 and Table 1, experiment 3. Original distributed under CC BY 3.0.
The three CSVs are digitizations of plotted points. They include the
pre-injection markers where shown. They are not original instrument files.

pdftocairo -f 7 -l 7 -svg acp-9-3331-2009.pdf page7.svg
Journal PDF SHA-256:
76932ed8b611c7d6ed25ed638f863fca2182dda4a645569cfb806b409e7e2228
Use vector marker centers, omit each series' final symbol (the legend).
In PDF page coordinates (points, y down), x=337.978661 and 505.412671
mark 0 and 1500 min. The black dilution line at x=417.23533 is 710.04 min.
y=179.135986 is zero and y=77.244982 is respectively 200 ppbv,
3.5 organic/sulfate, or 0.35 fragment/sulfate on the three axes.
Undo the printed factor of 16 for the purple m/z 68 diamonds. Green
squares give 2214 gas points, pink circles 780 organic points and purple
diamonds 773 fragment points. No smoothing or baseline correction.
A conservative ±0.15 page-point allowance in marker/tick position gives
±1.35 min, ±0.30 ppbv, ±0.0052 organic/sulfate, and ±0.000033 unscaled
marker/sulfate. This digitization precision excludes experimental error.
Table 1 quotes 20% uncertainty in gas glyoxal. Linear interpolation spans
instrument gaps, the longest about 57 min. Simulate only 0 to 1540 min.

Table 1 gives 182 ppbv, 56% RH, 293 K, 37% water (estimated by ISORROPIA)
and 87 µm³ seed/cm³ air. The measured gas trace drives the model. Treat water
as 37% of wet seed volume; retain initial water plus stoichiometric water.
The assumed monodisperse wet radius of 100 nm fixes represented count.
Material densities in kg/m³: water 1000, AS 1770, dry glyoxal equivalents
1690, acid equivalents 1830 and assumed IC 1400. Product densities are
model inputs. Only glyoxal exchanges with the gas.

Kampf et al. 2013 https://doi.org/10.1021/es400083d
Knote et al. 2014 https://doi.org/10.5194/acp-14-6213-2014 equations 1–4.
Compute AS molality from remaining salt moles per kg of current water.
At or above 12 mol/kg use times 44000 and 47000 s and bound/monomer 0.5;
below use 250 and 5500 s and ratio one. Acid equivalents conserve sulfur
but their salting effect is outside this AS relation. At each step match
the native transfer slope to tau1 at current size and composition.
The equilibrium relation is in mol per litre of aerosol water per atm.
A bound equivalent represents averaged reversible oligomer storage.

Yu et al. 2011 https://doi.org/10.1021/es200989n, supplement Table S2.
IC rates scaled by 1e-8 M/s. pH=3.5 uses 4.54 at [GL]=1 M and [NH4]=2 M.
The separate 1.5 M point, 9.74±0.90, checks the second-order extrapolation.
Other pH levels use 1.5 M GL and 2 M NH4 for their individual anchors.
For a rate r at glyoxal g M and 1 M ammonium sulfate, k_native =
r / g² / 1000³, for native mol/m³ of solution and formula-unit AS.
Using a fixed conditional activity coefficient at each pH extrapolates
the laboratory kinetics beyond its ionic-strength calibration. Lanes for
AS and H2SO4 track formula equivalents. Product material is IC (C4H4N2O).
mechanism.json is the native nominal-pH configuration, used by the example.

Saved history rows contain minutes, organic/sulfate, IC/sulfate, dissolved
and bound glyoxal/sulfate (kg/kg), net imposed gas, cumulative imposed gas
addition, removal, and cumulative evaporation (all kg/m³ of chamber air), followed by
remaining ammonium sulfate molality (mol/kg water).
Sulfate is SO4 mass including sulfate carried in acid equivalents.

The saved chamber_fit record is an empirical one-pool diagnostic. Its input
is the digitized gas history in ppbv. Its least-squares target is the
organic/sulfate ratio at native sampling times between seed injection and
710.04 min. The dilution observations are held out. Capacity is ratio per
ppbv, time constant is in hours, and fit RMSE is in organic/sulfate units.
The saved fitted/predicted history is time in minutes and organic/sulfate.
