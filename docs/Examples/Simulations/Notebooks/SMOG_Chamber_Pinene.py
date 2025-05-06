# %%

import numpy as np

import matplotlib.pyplot as plt
import particula as par

# %%

M_gmol = np.array(
    [200.0, 188.0, 216.0, 368.0, 186.0, 204.0, 195.0, 368.0, 158.0, 206.0]
)

OC_ratio = np.array(
    [0.40, 0.444, 0.50, 0.368, 0.444, 0.556, 0.857, 0.368, 0.375, 0.75]
)

HC_ratio = np.array(
    [1.60, 1.78, 1.60, 1.47, 1.56, 1.78, 1.75, 1.56, 1.75, 1.75]
)

c_total_ug_per_m3 = np.array(
    [8.79, 3.98, 1.13, 4.07, 0.628, 0.919, 0.766, 1.02, 0.399, 0.313]
)

name = np.array(
    [
        "C107OOH",
        "C97OOH",
        "C108OOH",
        "ALDOL_dimer_C19H28O7",
        "PINIC",
        "C921OOH",
        "C812OOH",
        "ESTER_dimer",
        "C811OH",
        "C813OOH",
    ]
)

c_sat_ug_per_m3 = np.array(
    [
        8620.171693,
        522.7659518,
        231.757194,
        2.27e-06,
        24.13243017,
        3.131375563,
        1.107025816,
        2.97e-06,
        2197.484083,
        0.04398829,
    ]
)

# not needed for the moment
c_liquid_ug_per_m3 = np.array(
    [
        0.007057093,
        0.052085067,
        0.032911505,
        4.069606969,
        0.140058292,
        0.632546668,
        0.660729371,
        1.017406513,
        0.001254946,
        0.311210297,
    ]
)

temperature_K = 298.15

# %% create gas phase species

# vapor pressures
vapor_pressure_strategies = []
for i in range(len(name)):
    vapor_pressure_organic = (
        par.gas.SaturationConcentrationVaporPressureBuilder()
        .set_molar_mass(M_gmol[i], "g/mol")
        .set_temperature(temperature_K, "K")
        .set_saturation_concentration(
            c_sat_ug_per_m3[i], "ug/m^3"
        )
        .build()
    )
    vapor_pressure_strategies.append(vapor_pressure_organic)


    
# %%
