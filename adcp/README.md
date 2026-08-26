# OOI ADCP Processing

This repo contains an end-to-end pipeline for downloading, merging, 
quality-controlling, and CF-aligning OOI CGSN fixed-depth moored 
ADCP data with co-located CTD observations. The result is a single
ADCP dataset for a given reference designator, regridded to a common
depth grid across deployments, and with associated flags indicating
quality of the primary observed data products, e.g. sea_water_velocities,
sea_water_temperature, etc.

## Overview

```
Download raw streams (OOINet M2M)
    ↓
Merge telemetered / recovered_host / recovered_inst
    ↓
Add co-located CTD  →  recalculate sound speed
    ↓
CTD calibration against ship bottle data
    ↓
Gap-fill CTD time series
    ↓
TRDI instrument QC  +  QARTOD tests
    ↓
Regrid to common depth grid
    ↓
CF-1.11 alignment, beam stacking, global metadata
    ↓
Save NetCDF
```

## Quickstart

```bash
# 1. Create the environment
conda env create -f environment.yml
conda activate ooi-cgsn-adcp
pip install -e /path/to/adcp

# 2. Run the full pipeline for one instrument
python scripts/process_adcp.py config/GI01SUMO-RII11-02-ADCPSN010.yml

# 3. Or step through interactively
jupyter lab notebooks/
```

## Repository Layout

```
adcp/
├── README.md
├── environment.yml
├── pyproject.toml
├── config/
│   └── GI01SUMO-RII11-02-ADCPSN010.yml     # one file per instrument
├── adcp/                                   # installable package
│   ├── __init__.py
│   ├── download.py
│   ├── merge.py
│   ├── ctd.py
│   ├── gap_fill.py
│   ├── qc.py
│   ├── regrid.py
│   ├── cf.py
│   └── physics.py
├── notebooks/
│   ├── 00-download.ipynb
│   ├── 01-process.ipynb
│   └── 02-explore.ipynb
├── scripts/
│   └── process_adcp.py
├── data/                                   # gitignored
└── results/                                # gitignored
```

## Adding a New Instrument

1. Copy an existing config YAML and update the fields.
2. Run `python scripts/process_adcp.py config/<new-refdes>.yaml`.

No code changes required for instruments of the same type (ADCPS-J/L/N).
For different ADCP models, extend the crosswalk in `adcp/merge.py`.

## Dependencies

See `environment.yaml`. Key packages:
- `xarray`, `numpy`, `pandas`, `scipy`
- `ooi_data_explorations` (OOI processing utilities)
- `pyyaml`
- `h5netcdf` (NetCDF4 / HDF5 write backend)

## Citation

Ocean Observatories Initiative (OOI) data accessed via the M2M API. <br>
QARTOD flag conventions: https://ioos.noaa.gov/project/qartod <br>
CF Conventions 1.11: http://cfconventions.org
