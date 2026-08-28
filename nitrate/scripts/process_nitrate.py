"""
process_nitrate.py
===================
End-to-end CLI pipeline for OOI SUNA/NUTNR nitrate processing.

Usage
-----
    python scripts/process_nitrate.py config/GI01SUMO-SBD11-08-NUTNRB000.yaml

    # Skip the download step if raw files already exist
    python scripts/process_nitrate.py config/GI01SUMO-SBD11-08-NUTNRB000.yaml --no-download

Steps
-----
1.  Download raw NUTNR streams (+ annotations, bottle data, and met data
    for buoy instruments) for each deployment (unless --no-download)
2.  Process each deployment: annotation QC flags, suna_datalogger /
    suna_instrument processing, burst-average, met T/S substitution
    (buoy only), Plant (2023) T-S(-P) correction, merge streams
3.  Drift-correct each deployment against its post-cruise calibration
4.  Merge all deployments for the reference designator
5.  Bottle-correct against shipboard discrete samples, deployment by
    deployment
6.  Trim to final variables, rename, fix dtypes and known metadata issues
7.  CF-1.11 alignment and global attribute cleanup
8.  Save output NetCDF
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from ooi_data_explorations.common import (
    get_annotations, load_kdata, add_annotation_qc_flags, list_deployments,
)
from ooi_data_explorations.combine_data import combine_datasets
from ooi_data_explorations.uncabled.process_nutnr import suna_datalogger, suna_instrument
from ooi_data_explorations.uncabled.process_metbk import metct_instrument

try:
    from ooi_data_explorations.bottles import clean_data
except ImportError:
    clean_data = None  # optional; only needed for Pioneer-NES cruise-dir bottle loading

# Allow running from the repo root without installing the package
import nitrate.resample as resample
import nitrate.calibration as calibration
import nitrate.bottles as bottles
import nitrate.qc as qc
import nitrate.cf as cf


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {msg}")
    print(f"{'─'*70}")


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_deployments(refdes: str, config: dict) -> list[int]:
    """Full deployment list from OOINet, minus deployments_to_drop."""
    site, node, sensor = refdes.split("-", 2)
    all_deployments = sorted(list_deployments(site, node, sensor))
    drop = set(config.get("deployments_to_drop", []))
    return [d for d in all_deployments if d not in drop]


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_download(config: dict, data_dir: str, deployments: list[int]) -> None:
    """Download raw NUTNR (+ met, if a buoy instrument) streams and
    annotations for every deployment. load_kdata() handles the actual M2M
    fetch/cache; this step is mostly here to keep the download phase
    separable from processing (e.g. so --no-download can skip it)."""
    _banner("Step 1 — Download")
    refdes = config["refdes"]
    site, node, sensor = refdes.split("-", 2)
    streams = config["streams"]

    get_annotations(site, node, sensor)

    for dN in deployments:
        dep = str(dN).zfill(4)
        for method, stream in streams.items():
            load_kdata(site, node, sensor, method, stream, tag=f"deployment{dep}_{refdes}*.nc")

    if config.get("met_refdes"):
        met_site, met_node, met_sensor = config["met_refdes"].split("-", 2)
        met_stream = config["met_streams"]["recovered_inst"]
        for dN in deployments:
            dep = str(dN).zfill(4)
            load_kdata(met_site, met_node, met_sensor, "recovered_inst", met_stream,
                       tag=f"deployment{dep}_{config['met_refdes']}*.nc")

    if config.get("ship_cruise_dir") and clean_data is not None:
        _download_cruise_bottle_data(config["ship_cruise_dir"], data_dir)


def _download_cruise_bottle_data(cruise_dir: str, data_dir: str) -> str:
    """Assemble and clean shipboard discrete nitrate samples from a
    directory of per-cruise Water_Sampling/*_Discrete_Summary.csv files
    (the Pioneer-NES layout), caching the cleaned result to
    data_dir/cleaned_bottle_data.csv."""
    bottle_data = None
    for cruise in sorted(os.listdir(cruise_dir)):
        cruise_path = os.path.join(cruise_dir, cruise, "Water_Sampling")
        if not os.path.exists(cruise_path):
            continue
        discrete_files = [f for f in os.listdir(cruise_path) if f.endswith("Discrete_Summary.csv")]
        if not discrete_files:
            continue
        cruise_data = pd.read_csv(os.path.join(cruise_path, discrete_files[0]), index_col=None)
        bottle_data = cruise_data if bottle_data is None else pd.concat([bottle_data, cruise_data], ignore_index=True)

    bottle_data = clean_data(bottle_data)
    outpath = os.path.join(data_dir, "cleaned_bottle_data.csv")
    bottle_data.to_csv(outpath, index=False)
    print(f"  Cleaned bottle data → {outpath}")
    return outpath


def _load_met_data(config: dict, dN: str) -> xr.Dataset | None:
    """Load and burst-average the co-located METBK data for a buoy (SBD)
    NUTNR deployment; returns None for instruments with a co-located CTD."""
    if not config.get("met_refdes"):
        return None
    met_refdes = config["met_refdes"]
    met_site, met_node, met_sensor = met_refdes.split("-", 2)
    met_stream = config["met_streams"]["recovered_inst"]
    met_data = load_kdata(met_site, met_node, met_sensor, "recovered_inst", met_stream,
                           tag=f"deployment{dN}_{met_refdes}*.nc")
    met_data = metct_instrument(met_data, burst=False)
    met_data = resample.met_burst_resample(met_data)
    return met_data


def _process_stream(config: dict, dN: str, method: str, processor, met_data) -> xr.Dataset:
    """Load, annotation-QC, instrument-process, burst-resample, and
    Plant-correct a single delivery method's stream for one deployment."""
    refdes = config["refdes"]
    site, node, sensor = refdes.split("-", 2)
    stream = config["streams"][method]

    annotations = get_annotations(site, node, sensor)
    data = load_kdata(site, node, sensor, method, stream, tag=f"deployment{dN}_{refdes}*.nc")
    data = add_annotation_qc_flags(data, annotations)

    # recovered_host serial_number sometimes carries an extra string4 dim
    if method == "recovered_host" and "string4" in data["serial_number"].dims:
        data["serial_number"] = data["serial_number"].isel(string4=0)

    data = processor(data, burst=False)
    data = resample.burst_resample(data)

    if met_data is not None:
        met_interp = met_data.interp_like(data)
        data["sea_water_practical_salinity"] = met_interp["sea_surface_salinity"]
        data["sea_water_temperature"] = met_interp["sea_surface_temperature"]

    data = calibration.add_plant_correction(data, site, node, sensor)
    return data


def step_process_deployment(config: dict, data_dir: str, dN: int) -> str:
    """Process one deployment end-to-end through stream merging, and save
    the merged (not yet drift- or bottle-corrected) dataset."""
    dep = str(dN).zfill(4)
    refdes = config["refdes"]
    site, node, sensor = refdes.split("-", 2)

    met_data = _load_met_data(config, dep)

    tdata = _process_stream(config, dep, "telemetered", suna_datalogger, met_data)
    hdata = _process_stream(config, dep, "recovered_host", suna_datalogger, met_data)
    idata = _process_stream(config, dep, "recovered_inst", suna_instrument, met_data)

    tdata = tdata.drop_vars("internal_timestamp", errors="ignore")
    hdata = hdata.drop_vars("internal_timestamp", errors="ignore")
    idata = idata.drop_vars("internal_timestamp", errors="ignore")
    data = combine_datasets(tdata, hdata, idata, None)

    outpath = os.path.join(data_dir, f"{refdes}_deployment{dep}_merged.nc")
    data.to_netcdf(outpath, format="netcdf4", engine="h5netcdf")
    print(f"  Deployment {dep}: merged → {outpath}")
    return outpath


def step_drift_correct(config: dict, data_dir: str, merged_path: str) -> str:
    """Apply the Plant (2023) drift correction to one deployment's merged
    dataset, using the post-cruise calibration."""
    refdes = config["refdes"]
    site, node, sensor = refdes.split("-", 2)

    data = xr.open_dataset(merged_path)
    data = calibration.plant_drift_correction(data, site, node, sensor)

    outpath = merged_path.replace("_merged.nc", "_drift_corrected.nc")
    data.to_netcdf(outpath, format="netcdf4", engine="h5netcdf")
    print(f"  Drift-corrected → {outpath}")
    return outpath


def step_merge_deployments(drift_corrected_paths: list[str]) -> xr.Dataset:
    """Merge all per-deployment drift-corrected datasets into a single
    dataset for the reference designator."""
    _banner("Step 4 — Merge deployments")
    data = None
    for path in drift_corrected_paths:
        ds = xr.open_dataset(path)
        data = ds if data is None else xr.concat([data, ds], dim="time")
    print(f"  Merged {len(drift_corrected_paths)} deployments: {dict(data.sizes)}")
    return data


def step_bottle_correct(config: dict, data: xr.Dataset, deployments: list[int]) -> xr.Dataset:
    """Bottle-correct the merged dataset, deployment by deployment, against
    shipboard discrete nitrate samples."""
    _banner("Step 5 — Bottle correction")
    refdes = config["refdes"]
    site, node, sensor = refdes.split("-", 2)

    bottle_data = pd.read_csv(config["bottle_csv"]) if config.get("bottle_csv") else None
    if bottle_data is not None:
        bottle_data["Cruise"] = bottle_data["Cruise"].apply(bottles.remove_last_letter)
        bottle_data["Start Time [UTC]"] = bottle_data["Start Time [UTC]"].apply(lambda x: x.strip("Z"))
        bottle_data["Start Time [UTC]"] = bottle_data["Start Time [UTC]"].astype("datetime64[ns]")

    deploy_info = pd.DataFrame(bottles.get_deployment_info(site, node, sensor, deployments))
    deploy_info.set_index(keys="deploymentNumber", drop=True, inplace=True)

    corrected = None
    for depNum in np.unique(data["deployment"]):
        depdata = data.where(data.deployment == depNum, drop=True)
        depdata, _, _, _ = bottles.bottle_correction(depdata, deploy_info, bottle_data)
        corrected = depdata if corrected is None else xr.concat([corrected, depdata], dim="time")

    return corrected


def step_finalize(config: dict, data: xr.Dataset) -> xr.Dataset:
    """Trim to the configured save_vars, rename for clarity, fix dtypes,
    patch known metadata issues, clean global attributes, and align to
    CF-1.11."""
    _banner("Step 6 — Finalize")
    save_vars = config["save_vars"]

    final = data[save_vars]
    final = final.rename({
        "corrected_nitrate_concentration_mad": "burst_median_absolute_deviation",
        "nitrate_sensor_quality_flag": "nitrate_concentration_qc_flag",
    })
    final["burst_median_absolute_deviation"].attrs["comment"] = (
        "The median absolute deviation calculated for each sampling burst.")
    final["drift_corrected_nitrate_qc_flag"] = final["drift_corrected_nitrate_qc_flag"].astype("int")
    final["nitrate_concentration_qc_flag"] = final["nitrate_concentration_qc_flag"].astype("int")
    final["deployment"] = final["deployment"].astype("int")
    final["serial_number"] = final["serial_number"].astype("int")

    for depNum, serial_number in config.get("serial_number_fixes", []):
        final = cf.fix_serial_numbers(final, depNum, serial_number)

    final = cf.clean_netcdf(final)
    final = qc.add_not_evaluated_flags(final, "bottle_corrected_nitrate")
    final = qc.add_not_evaluated_flags(final, "drift_corrected_nitrate")

    _banner("Step 7 — CF-1.11 alignment")
    final = cf.finalize_cf_compliance(final)

    return final


def step_save(final: xr.Dataset, config: dict, data_dir: str) -> str:
    """Save the final NetCDF."""
    _banner("Step 8 — Save")
    refdes = config["refdes"]
    outpath = os.path.join(data_dir, f"{refdes}.nc")
    final.to_netcdf(outpath, format="netcdf4", engine="h5netcdf")
    print(f"  Saved → {outpath}")
    return outpath


# ── Main entry point ──────────────────────────────────────────────────────────

def run(config_path: str, no_download: bool = False) -> str:
    """
    Run the full pipeline for one instrument config.

    Parameters
    ----------
    config_path : path to the YAML config file
    no_download : skip the download step (use existing raw/merged files)

    Returns
    -------
    outpath : path to the saved output NetCDF
    """
    config = _load_config(config_path)
    paths_cfg = config.get("paths", {})
    config_dir = os.path.dirname(os.path.abspath(config_path))
    data_dir = os.path.join(config_dir, paths_cfg.get("data_dir", "../data/"))
    results_dir = os.path.join(config_dir, paths_cfg.get("results_dir", "../results/"))
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    refdes = config["refdes"]
    deployments = _resolve_deployments(refdes, config)
    print(f"  Deployments: {deployments}")

    if not no_download:
        step_download(config, data_dir, deployments)

    _banner("Step 2/3 — Process + drift-correct each deployment")
    drift_corrected_paths = []
    for dN in deployments:
        merged_path = step_process_deployment(config, data_dir, dN)
        drift_corrected_paths.append(step_drift_correct(config, data_dir, merged_path))

    data = step_merge_deployments(drift_corrected_paths)
    data = step_bottle_correct(config, data, deployments)
    final = step_finalize(config, data)
    outpath = step_save(final, config, data_dir)

    _banner("Pipeline complete")
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end OOI SUNA/NUTNR nitrate processing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="Path to instrument YAML config file",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        default=False,
        help="Skip download; use existing raw/merged files in data/",
    )
    args = parser.parse_args()
    run(args.config, no_download=args.no_download)


if __name__ == "__main__":
    main()
