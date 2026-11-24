# How much measurement uncertainty would change the observed size shift?

At 2 m/s the three-hour volume median exceeded its 30-minute value by 2.69 µm. The increase persisted when we varied the diameter calibration, suppressed weak channels and placed hypothetical undetected volume at either extreme of the measured size range. All four winds retained the observed sign of their shift in these tests. The archive provides averaged spectra. Recovering a sampling interval for the median, or testing a specific small change between model variants against measurement error, would require the individual tray spectra.

## Diameter calibration and weak-channel detection

The laser instrument's checks used 0.1 and 20 µm standards with a ±1% diameter acceptance criterion, and a standard suspension every four hours with recalibration triggered by a >2% shift in key parameters. Blank noise had to stay below 0.5%. Each tray sample was sized three times and averaged. The authors report instrument checks but provide no channel-specific limit of detection, response or calibration covariance for irregular mineral dust. The reported percentage for blank noise refers to a blank measurement and cannot directly supply an additive error for every size channel. [Jia et al., §2.4–2.5](https://acp.copernicus.org/articles/26/14245/2026/) · [Supplement](https://acp.copernicus.org/articles/26/14245/2026/acp-26-14245-2026-supplement.pdf) · [Jia and Liu, v2 archive](https://zenodo.org/records/22962214)

The diameter-only calculation shifts all channel diameters within an endpoint together and allows opposite shifts at the two endpoints. A common scale error at both times preserves the sign of the difference. The combined calculation also suppresses all channels originally contributing less than 0.5 percentage points and renormalizes the remaining spectrum to 99% of volume. It places a hypothetical missing 1% at the smallest or largest diameter and permits different placements at the two times. The censorship and hidden fractions are deliberately chosen stress assumptions. Neither is a measured instrument threshold. The measured diameter convention and log-scale interpolation of cumulative volume remain common throughout.

| Wind (m/s) | Measured change in D50 (µm) | Independent ±2% diameter scaling (µm) | Combined stress range (µm) |
|---:|---:|---:|---:|
| 0 | −2.66 | −3.15 to −2.18 | −3.43 to −1.71 |
| 2 | +2.69 | +2.35 to +3.02 | +2.23 to +3.39 |
| 4 | +1.14 | +0.87 to +1.42 | +0.71 to +1.73 |
| 6 | −11.45 | −12.39 to −10.50 | −12.67 to −10.13 |

These are conditional sensitivity ranges. They quantify the robustness of the measured median shift to those specific perturbations. They do not measure sampling variability across trays, run-to-run variation, aggregate breakup during sample preparation or optical inversion bias.

## Tray-to-tray variability in retained mass

The balance readout resolution was 0.1 mg. The SEM at 2 m/s was 1.94 mg per tray at 30 minutes and 2.82 mg at three hours for total retained mass below 63 µm. Combining the two independent endpoint groups gives a +15.12 mg/tray change and an approximate Welch 95% interval of +7.06 to +23.19 mg/tray. This treats trays as independent within each time group and describes the sampled experiment. Shared flow conditions and systematic spatial differences between the trays assigned to the two sampling times remain sources of uncertainty. For the narrower 10–20 µm band the group-mean gain was +6.13 mg/tray. Missing tray identifiers across the two cumulative thresholds allow a range of SEMs from 0.95 to 4.35 mg for that gain. [Endpoint mass SEMs (`total-mass.csv`)](study-results.zip) · [Feasible tray pairings (`middle-band-spread.csv`)](study-results.zip)

| Wind (m/s) | Early / late trays for <63 µm | Change in <63 µm mass (mg/tray) | Approximate 95% interval (mg/tray) |
|---:|---:|---:|---:|
| 0 | 5 / 5 | +13.42 | −1.26 to +28.10 |
| 2 | 5 / 5 | +15.12 | +7.06 to +23.19 |
| 4 | 5 / 4 | +8.03 | −1.57 to +17.63 |
| 6 | 5 / 5 | +13.65 | +5.67 to +21.63 |

At 6 m/s the three-hour laser fraction below 20 µm (56.21%) and the archived mass fraction under the authors’ conversion (19.82%) still need reconciliation at the tray level. A conclusion about absolute size-resolved retention at that wind depends on how the two archive series were obtained. The supplement’s separate edge-capture experiment detected markedly stronger mobilization at 6.01 m/s than at 5.87 m/s. That observation makes surface removal a relevant sensitivity at the highest wind. It supplies neither an error bar on the tray spectrum nor a size-resolved resuspension rate. [Cross-check (`consistency.csv`)](study-results.zip) · [Jia et al., Supplement Figure S1](https://acp.copernicus.org/articles/26/14245/2026/acp-26-14245-2026-supplement.pdf)

Run `python measurement_check.py` with NumPy and SciPy in the study environment. Inputs are the verified channel spectra and endpoint mass summary from the main replay. [Script](measurement_check.py) · [All seven scenarios and four winds (`measurement-sensitivity.csv`)](study-results.zip)

The optical follow-up tested scanning-beam transit and angular scattering, quantified the smallest synthetic signal changes that admit equal medians at 2 m/s, and checked every pairing of the five 6 m/s trays. [Optical study](optical-study.md)
