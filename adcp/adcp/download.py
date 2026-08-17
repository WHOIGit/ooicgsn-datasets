"""
adcp.download
=============
Download raw OOI ADCP and co-located CTD data via the M2M API,
assemble ship bottle data from local cruise directories, and save
raw NetCDF files to disk.

All functions accept a ``config`` dict loaded from a per-instrument
YAML file (see ``config/`` directory) so they are fully generic.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import xarray as xr

from ooi_data_explorations.common import load_kdata, get_sensor_information
from ooi_data_explorations.combine_data import combine_datasets
from ooi_data_explorations.uncabled import process_ctdbp

try:
    from ooi_data_explorations import bottles
except ImportError:
    bottles = None  # optional; only needed for bottle download


# ── Utilities ─────────────────────────────────────────────────────────────────

def _split_refdes(refdes: str) -> tuple[str, str, str]:
    """Split 'GI01SUMO-RII11-02-ADCPSN010' → (site, node, sensor)."""
    site, node, sensor = refdes.split("-", 2)
    return site, node, sensor


def sanitize_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Clean dataset and variable attributes for NetCDF serialization.
    Converts bytes values to strings; drops unsupported types.
    """
    def _clean(attrs: dict) -> dict:
        out = {}
        for k, v in attrs.items():
            if isinstance(v, bytes):
                out[k] = v.decode("utf-8")
            elif isinstance(v, (str, int, float, np.ndarray, list, tuple)):
                out[k] = v
            else:
                print(f"  Dropping attribute '{k}' — unsupported type {type(v)}: {v}")
        return out

    ds.attrs = _clean(ds.attrs)
    for var in ds.variables:
        ds[var].attrs = _clean(ds[var].attrs)
    return ds


def _get_serial_number(refdes, deployment) -> str:
    """
    Decode byte-array serial numbers in OOI metadata datasets to plain strings.
    """
    site, node, sensor = _split_refdes(refdes)
    sensor_info = get_sensor_information(site, node, sensor, deployment)
    serial_number = sensor_info[0].get('sensor').get('serialNumber', "-9999999")
    return serial_number


def _assign_serial_numbers(
    ds: xr.Dataset,
    deployments: np.ndarray,
) -> xr.Dataset:
    """
    Assign per-deployment serial numbers to a data dataset from the metadata
    dataset. Adds a string ``serial_number`` variable aligned on time.
    """
    ds["serial_number"] = (["time"], np.empty(ds["time"].shape, dtype="U20"))
    # Build refdes from the dataset attrs
    site, node, sensor = ds.attrs['subsite'], ds.attrs['node'], ds.attrs['sensor']
    refdes = "-".join((site, node, sensor))
    for d in deployments:
        # Get the serial number
        sn = _get_serial_number(refdes, d)
        idx_data, = np.where(ds["deployment"] == d)
        ds["serial_number"].values[idx_data] = sn
    return ds


# ── ADCP download ─────────────────────────────────────────────────────────────

def download_adcp(
    config: dict,
    output_dir: str = "data/",
) -> dict[str, str]:
    """
    Download all three ADCP data streams (telemetered, recovered_host,
    recovered_inst) plus the telemetered metadata stream.

    Parameters
    ----------
    config     : instrument config dict (loaded from YAML)
    output_dir : directory to write raw NetCDF files

    Returns
    -------
    paths : dict mapping stream name → output file path
    """
    os.makedirs(output_dir, exist_ok=True)
    refdes  = config["refdes"]
    streams = config["streams"]

    site, node, sensor = _split_refdes(refdes)

    print(f"Downloading ADCP data for {refdes} …")

    tdata = load_kdata(site, node, sensor, "telemetered",    streams["telemetered"],    tag=f"*{refdes}*.nc")
    hdata = load_kdata(site, node, sensor, "recovered_host", streams["recovered_host"], tag=f"*{refdes}*.nc")
    idata = load_kdata(site, node, sensor, "recovered_inst", streams["recovered_inst"], tag=f"*{refdes}*.nc")

    # Assign serial numbers across all streams
    deployments = np.unique(idata["deployment"])
    for ds in (tdata, hdata, idata):
        _assign_serial_numbers(ds, deployments)

    # Sanitize and save
    paths = {}
    for name, ds in [
        ("telemetered",    tdata),
        ("recovered_host", hdata),
        ("recovered_inst", idata),
    ]:
        ds   = sanitize_attrs(ds)
        path = os.path.join(output_dir, f"{refdes}.{name}.raw.nc")
        ds.to_netcdf(path, format="netcdf4", engine="h5netcdf")
        paths[name] = path
        print(f"  Saved {name} → {path}")

    return paths


# ── CTD download ──────────────────────────────────────────────────────────────

def download_ctd(
    config: dict,
    output_dir: str = "data/",
) -> str:
    """
    Download and merge CTD streams co-located with the ADCP mooring.

    Parameters
    ----------
    config     : instrument config dict
    output_dir : directory to write merged NetCDF file

    Returns
    -------
    path : output file path
    """
    os.makedirs(output_dir, exist_ok=True)
    ctd_refdes  = config["ctd_refdes"]
    ctd_streams = config["ctd_streams"]

    site, node, sensor = _split_refdes(ctd_refdes)

    print(f"Downloading CTD data for {ctd_refdes} …")

    ctd_tdata = load_kdata(site, node, sensor, "telemetered",    ctd_streams["telemetered"],    tag=f"*{ctd_refdes}*.nc")
    ctd_hdata = load_kdata(site, node, sensor, "recovered_host", ctd_streams["recovered_host"], tag=f"*{ctd_refdes}*.nc")
    ctd_idata = load_kdata(site, node, sensor, "recovered_inst", ctd_streams["recovered_inst"], tag=f"*{ctd_refdes}*.nc")

    # Process and merge
    ctd_tdata = process_ctdbp.ctdbp_datalogger(ctd_tdata)
    ctd_hdata = process_ctdbp.ctdbp_datalogger(ctd_hdata)
    ctd_idata = process_ctdbp.ctdbp_instrument(ctd_idata)

    ctd = combine_datasets(ctd_tdata, ctd_hdata, ctd_idata, None)

    ctd = sanitize_attrs(ctd)

    path = os.path.join(output_dir, f"{ctd_refdes}.merged.nc")
    ctd.to_netcdf(path, format="netcdf4", engine="h5netcdf")
    print(f"  Saved CTD → {path}")
    return path


# ── Ship bottle data ──────────────────────────────────────────────────────────

def download_bottle_data(
    ship_dir: str,
    output_dir: str = "data/",
    output_filename: str = "cleaned_bottle_data.csv",
) -> str:
    """
    Assemble and clean discrete water-sample data from local cruise directories.

    Expects each cruise directory to contain a ``Water_Sampling/`` subfolder
    with a file whose name includes ``Discrete_Summary.csv``.

    Parameters
    ----------
    ship_dir         : root directory containing one subfolder per cruise
    output_dir       : directory to write the combined CSV
    output_filename  : output CSV filename

    Returns
    -------
    path : output CSV file path
    """
    if bottles is None:
        raise ImportError(
            "ooi_data_explorations.bottles is required for bottle data download. "
            "Install ooi-data-explorations."
        )

    os.makedirs(output_dir, exist_ok=True)
    cruises = sorted(os.listdir(ship_dir))
    cleaned = []

    print(f"Processing bottle data from {len(cruises)} cruise(s) in {ship_dir} …")
    for cruise in cruises:
        sampling_dir = os.path.join(ship_dir, cruise, "Water_Sampling")
        if not os.path.isdir(sampling_dir):
            continue
        files = os.listdir(sampling_dir)
        matches = [f for f in files if "Discrete_Summary.csv" in f]
        if not matches:
            continue
        discrete_path = os.path.join(sampling_dir, matches[0])
        raw = pd.read_csv(discrete_path)
        cleaned.append(bottles.clean_data(raw))
        print(f"  {cruise}: {len(raw)} rows")

    if not cleaned:
        raise FileNotFoundError(f"No Discrete_Summary.csv files found under {ship_dir}")

    all_data = pd.concat(cleaned, ignore_index=True)
    path = os.path.join(output_dir, output_filename)
    all_data.to_csv(path, index=False)
    print(f"  Saved bottle data ({len(all_data)} rows) → {path}")
    return path