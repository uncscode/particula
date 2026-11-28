# NASA POWER hourly surface solar data

The layered-column example replays the archived UTC day of 2022-07-15 for the grid cell requested at 67.36663° N, 26.62901° E near Sodankylä, Finland. The data were obtained from the NASA Langley Research Center Prediction Of Worldwide Energy Resources (POWER) Project, funded by the NASA Earth Science Division. The manifest records the API version, data sources, original request URL, access time, and file and value checksums. [POWER referencing guide](https://power.larc.nasa.gov/docs/referencing/) · [Hourly API](https://power.larc.nasa.gov/docs/services/api/temporal/hourly/)

Run this command from a checkout with Particula installed to download the three surface shortwave channels to a new directory. Change the coordinates and dates for another location. The fetch requires an Internet connection. The saved JSON can subsequently be used offline.

```bash
python studies/shortwave-radiation/download_power.py ./sodankyla-replay \
  --latitude 67.36663 --longitude 26.62901 --start 2022-07-15 --end 2022-07-15
```

The UTC label marks the hour's start. The reader divides hourly irradiation in Wh/m² by one hour to obtain mean W/m². Global and diffuse horizontal irradiance refer to the horizontal surface. Direct normal irradiance is perpendicular to the beam. The gridded, all-sky fields are satellite-constrained estimates. They include the attenuation of the intervening atmosphere. In a full-column simulation compare them at the surface. For a separate added slab prescribe the direct and diffuse fields at its upper face and document that placement. [Time conventions](https://power.larc.nasa.gov/docs/faqs/other/) · [CERES and POWER radiation methods](https://power.larc.nasa.gov/docs/methodology/data/derived/syn1deg/)

The example includes the original approximately 2 kB response with its manifest. Check the raw `sha256` for an exact-file replay and `parameter_sha256` when checking a fresh acquisition. Provider timing metadata can change between requests. Independent releases may also revise values. [Saved inputs](../../docs/data/shortwave/sodankyla-20220715/) · [Worked column example](../../docs/studies/examples/coupled-populations/radiation/layer_altitude.py)
