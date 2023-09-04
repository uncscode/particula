
gibbs_mix, dervative_gibbs = gibbs_of_mixing(
    molarmass_ratio,
    org_mole_fraction,
    O2C,
    density,
    FIT_lowO2C
)

organic_molecular_weight = np.linspace(50, 500, 500)

o2c_value = organic_water_single_phase(18.01528 / organic_molecular_weight)

fig, ax = plt.subplots()
ax.plot(
    organic_molecular_weight,
    o2c_value,
    label="initial",
    linestyle='dashed'
    )
ax.set_xlabel("Organic molecular weight (g/mol)")
ax.set_ylabel("O:C ratio")
ax.set_title("Organic Phase separation")
ax.legend()
#     org_mole_fraction, O2C, H2C, molarmass_ratio, BAT_functional_group, special_options, N2C_values_denistyOnly)





#     gibbs_real,
#     label="gibbs",
#     linestyle='dashed'
#     )
ax.plot(
    1-org_mole_fraction,
    activity_water,
    label="water",
    linestyle='dashed'
    )

ax.plot(
    1-org_mole_fraction,
    activity_organic,
    label="organic",
    )
ax.set_ylim((0,1))
ax.set_xlabel("water mole fraction")
ax.set_ylabel("activity")
ax.legend()
fig.show