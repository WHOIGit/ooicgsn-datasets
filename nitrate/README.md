# OOI SUNA/NUTNR Nitrate Processing

[![Irminger dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21629872.svg)](https://doi.org/10.5281/zenodo.21629872)

[![Pioneer-NES dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14908031.svg)](https://doi.org/10.5281/zenodo.14908031)

This repo contains an end-to-end pipeline for downloading, correcting, quality-controlling, and CF-aligning OOI CGSN/Pioneer-NES and Irminger Sea Submersible Ultraviolet Nitrate Analyzer (SUNA) data. The result is a validated, deployment-merged dissolved nitrate dataset for a given reference designator, temperature/salinity/pressure corrected using the Plant et al. (2023) update to the Sakamoto et al. (2009) algorithm, drift-corrected against post-cruise calibrations, and offset-corrected against shipboard discrete (bottle) samples.

## Background

OOI has deployed both the In-Situ Ultraviolet Spectrophotometer (ISUS) and the Submersible Ultraviolet Nitrate Analyzer (SUNA) for continuous, in-situ measurement of nitrate at the Pioneer-New England Shelf, Pioneer-Mid Atlantic Bight, and Global Irminger Sea arrays. SUNA datasets are delivered by OOI as "NUTNR" (Nutrient Sensor). ISUS datasets (pre spring-2018) are not covered by this pipeline, since known measurement issues make a quantitative data quality assessment difficult. The SUNA sensor replaced the ISUS sensors spring 2018. The SUNA was a major improvement in technology, with significant improvements in accuracy and precision. However, it still suffers from calibration drift due to lamp fatigue and biofouling as well as spectral interference due to bromide and fluorometric CDOM. Drift is corrected by application of post-cruise calibrations to recalculate the temperature-and-salinity corrected nitrate concentration following Plant et al (2023) and estimating a linear drift between pre-and-post cruise deployments. Validation is performed by comparison with discrete water samples collected during deployment/recovery of the sensors.

## Published Data

The final, published datasets produced by this pipeline are archived on
Zenodo, one record per array:

| Array | DOI |
| --- | --- |
| Irminger Sea | [10.5281/zenodo.21629872](https://doi.org/10.5281/zenodo.21629872) |
| Pioneer-New England Shelf | [10.5281/zenodo.14908031](https://doi.org/10.5281/zenodo.14908031) |

Each record covers every instrument (`config/*.yaml`) belonging to that
array. When updating a published dataset, mint a new version under the
same Zenodo record rather than a new one, and update the DOI badges above
if Zenodo issues a new version-specific DOI.

## Overview

```
Download raw NUTNR streams + annotations (OOINet M2M)
    ↓
Process each deployment (suna_datalogger / suna_instrument) + annotation QC flags
    ↓
Instrument QC — RMSE, absorbance, dark value, spectrum average, range checks
    ↓
Recalculate corrected_nitrate_concentration — Plant et al. (2023) T-S(-P) algorithm
    ↓
Burst-average to 15-minute bins (median + median absolute deviation)
    ↓
Merge deployments for a reference designator
    ↓
Drift correction — linear drift between pre- and post-cruise calibrations
    ↓
Bottle offset correction — shipboard discrete samples at deployment/recovery
    ↓
Trim to final variables, CF-1.11 alignment, global metadata
    ↓
Save NetCDF
```

## Quickstart

```bash
# 1. Create the environment
conda env create -f environment.yaml
conda activate ooi-cgsn-nitrate
pip install -e /path/to/nitrate

# 2. Run the full pipeline for one instrument
python scripts/process_nitrate.py config/GI01SUMO-SBD11-08-NUTNRB000.yaml

# 3. Or step through interactively
jupyter lab notebooks/
```

## Repository Layout

```
nitrate/
├── README.md
├── environment.yaml
├── pyproject.toml
├── config/
│   ├── GI01SUMO-SBD11-08-NUTNRB000.yaml    # Irminger SUMO buoy (met-derived T/S)
│   └── CP01CNSM-RID26-07-NUTNRB000.yaml    # Pioneer-NES NSIF (CTD-colocated)
├── nitrate/                                # installable package
│   ├── __init__.py
│   ├── time_utils.py
│   ├── resample.py
│   ├── ts_correction.py
│   ├── calibration.py
│   ├── qc.py
│   ├── bottles.py
│   └── cf.py
├── notebooks/
│   ├── 00-download.ipynb
│   ├── 01-process.ipynb
│   ├── 02-bottle-correction.ipynb
│   └── 03-finalize.ipynb
├── scripts/
│   └── process_nitrate.py
├── data/                                   # gitignored
└── results/                                # gitignored
```

## Adding a New Instrument

1. Copy an existing config YAML and update the fields (`refdes`, `deployments`,
   `met_refdes` if the instrument is on a buoy without co-located CTD, and
   `mooring_pattern` for bottle matching).
2. Run `python scripts/process_nitrate.py config/<new-refdes>.yaml`.

No code changes are required for additional SUNA/NUTNR deployments of the
same instrument class.

## Dependencies

See `environment.yaml`. Key packages:
- `xarray`, `numpy`, `pandas`, `scipy`
- `ooi_data_explorations` (OOI processing utilities)
- `pyyaml`
- `h5netcdf` (NetCDF4 / HDF5 write backend)

## Algorithm Notes

- **T-S(-P) correction**: `nitrate.ts_correction.plant2023_tsp_correction`
  implements the Plant et al. (2023) update to Sakamoto et al. (2009),
  including the Sakamoto et al. (2017) pressure correction. The original
  Sakamoto (2009) implementation is retained
  (`nitrate.ts_correction.sakamoto_2009`) for comparison; the pipeline
  preserves the OOI-delivered Sakamoto (2009) value alongside the Plant
  (2023) value in the output (`corrected_nitrate_concentration_sakamoto2009`).
- **Drift correction**: `nitrate.calibration.plant_drift_correction` assumes
  linear drift between the pre-deployment and next (post-cruise)
  calibration, following Palevsky et al. (2023).
- **Bottle correction**: `nitrate.bottles.bottle_correction` smooths the
  sensor time series (6-hour centered rolling average) and compares it
  against deployment/recovery discrete bottle samples to compute a
  per-deployment offset.

## Citation

Ocean Observatories Initiative (OOI) data accessed via the M2M API. <br>
Plant, J.N., et al. (2023). Improved calculation of high-resolution
nitrate concentrations from an in-situ ultraviolet spectrophotometer,
including a temperature, salinity, and pressure correction. <br>
Sakamoto, C.M., Johnson, K.S., and Coletti, L.J. (2009). Improved
algorithm for the computation of nitrate concentrations in seawater
using an in situ ultraviolet spectrophotometer. Limnology and
Oceanography: Methods 7: 132-143. <br>
Palevsky, H.I., et al. (2023). Ocean Observatories Initiative biogeochemical
sensor data: Documenting drift and offsets from shipboard discrete
measurements. <br>
QARTOD flag conventions: https://ioos.noaa.gov/project/qartod <br>
CF Conventions 1.11: http://cfconventions.org
