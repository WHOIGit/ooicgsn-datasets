"""
process_adcp.py
===============
End-to-end CLI pipeline for OOI ADCP processing.

Usage
-----
    python scripts/process_adcp.py config/GI01SUMO-RII11-02-ADCPSN010.yml

    # Skip the download step if raw files already exist
    python scripts/process_adcp.py config/GI01SUMO-RII11-02-ADCPSN010.yml --no-download

Steps
-----
1.  Download raw ADCP and CTD streams via M2M (unless --no-download)
2.  Download ship bottle data (unless --no-download)
3.  Merge ADCP streams + apply crosswalk
4.  Calibrate CTD against bottle data
5.  Gap-fill CTD temperature and salinity
6.  Merge CTD into ADCP; recalculate sound speed
7.  Sensor-engineering QC (TRDI + roll/pitch/sidelobe)
8.  Regrid to common depth grid
9.  Drop intermediate variables
10. QARTOD tests — scalar CTD variables
11. QARTOD tests — 2-D velocity variables
12. CF-1.11 alignment (names, attrs, beam stacking, global attrs, station)
13. Save output NetCDF
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import xarray as xr
import yaml

# Allow running from the repo root without installing the package
import adcp.download  as download
import adcp.merge     as merge
import adcp.ctd       as ctd_mod
import adcp.gap_fill  as gap_fill
import adcp.qc        as qc
import adcp.regrid    as regrid
import adcp.cf        as cf


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {msg}")
    print(f"{'─'*70}")


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_download(config: dict, data_dir: str, ship_dir: str | None) -> dict:
    """Download raw ADCP, CTD, and bottle data."""
    _banner("Step 1 — Download")
    paths = {}
    paths["adcp"] = download.download_adcp(config, output_dir=data_dir)
    paths["ctd"]  = download.download_ctd(config,  output_dir=data_dir)
    if ship_dir:
        paths["bottles"] = download.download_bottle_data(ship_dir, output_dir=data_dir)
    return paths


def step_merge(config: dict, data_dir: str) -> xr.Dataset:
    """Load raw streams and merge."""
    _banner("Step 2 — Merge streams")
    refdes = config["refdes"]
    tdata  = xr.open_dataset(os.path.join(data_dir, f"{refdes}.telemetered.raw.nc"))
    hdata  = xr.open_dataset(os.path.join(data_dir, f"{refdes}.recovered_host.raw.nc"))
    idata  = xr.open_dataset(os.path.join(data_dir, f"{refdes}.recovered_inst.raw.nc"))

    adcp = merge.merge_adcp_streams(
        tdata, hdata, idata,
        deployments_to_drop=config.get("deployments_to_drop"),
    )
    print(f"  Merged dataset: {dict(adcp.sizes)}")
    return adcp


def step_calibrate_ctd(config: dict, data_dir: str, results_dir: str) -> xr.Dataset:
    """Load CTD, calibrate, gap-fill, return filled dataset."""
    _banner("Step 3 — CTD calibration + gap fill")
    ctd_refdes  = config["ctd_refdes"]
    bottle_path = os.path.join(data_dir, "cleaned_bottle_data.csv")
    qc_cfg      = config.get("qc", {})

    ctd = xr.open_dataset(os.path.join(data_dir, f"{ctd_refdes}.merged.nc"))

    cal_ctd, _, _ = ctd_mod.calibrate_ctd(
        ctd,
        bottle_path             = bottle_path,
        output_dir              = results_dir,
        mooring_pressure        = qc_cfg.get("mooring_pressure",        500.0),
        mooring_pressure_window = qc_cfg.get("mooring_pressure_window", 100.0),
        mooring_pattern         = config.get("mooring_pattern",         "SUMO"),
    )

    ctd_filled, _ = gap_fill.fill_ctd_gaps(cal_ctd)
    ctd_filled.to_netcdf(
        os.path.join(data_dir, f"{ctd_refdes}.calibrated.filled.nc"),
        format="netcdf4", engine="h5netcdf",
    )
    return ctd_filled


def step_merge_ctd(adcp: xr.Dataset, ctd_filled: xr.Dataset) -> xr.Dataset:
    """Interpolate CTD onto ADCP time axis and recalculate sound speed."""
    _banner("Step 4 — Merge CTD into ADCP")
    return ctd_mod.merge_ctd_into_adcp(adcp, ctd_filled)


def step_sensor_qc(adcp: xr.Dataset, config: dict) -> xr.Dataset:
    """Apply TRDI instrument QC and add qc_flags variable."""
    _banner("Step 5 — Sensor engineering QC")
    combined_qc, combined_attrs = qc.compute_sensor_engineering_qc(adcp, config)
    adcp["qc_flags"] = (["time", "bin"], combined_qc)
    adcp["qc_flags"].attrs = combined_attrs
    return adcp


def step_regrid(adcp: xr.Dataset, config: dict) -> xr.Dataset:
    """Interpolate onto common depth grid."""
    _banner("Step 6 — Regrid")
    depth_grid = regrid.build_depth_grid(config)
    adcp_g     = regrid.regrid_adcp(adcp, depth_grid)
    adcp_g     = merge.drop_processing_vars(adcp_g)
    print(f"  Regridded: {dict(adcp_g.sizes)}")
    return adcp_g


def step_qartod_scalar(adcp_g: xr.Dataset, config: dict) -> xr.Dataset:
    """QARTOD tests on scalar CTD variables (temperature, salinity)."""
    _banner("Step 7 — QARTOD scalar tests")
    qc_cfg = config.get("qc", {})

    from ooi_data_explorations.qartod import gross_range, climatology

    for var, fail_key in [
        ("sea_water_temperature",         "temperature_fail_range"),
        ("sea_water_practical_salinity",  "salinity_fail_range"),
    ]:
        if var not in adcp_g:
            continue
        fail_range = tuple(qc_cfg.get(fail_key, [-999, 999]))

        # Fit gross range and climatology from the data
        gr = gross_range.GrossRange(*fail_range)
        gr.fit(adcp_g, var, sigma=3, check_normality=True)

        clm = climatology.Climatology()
        clm.fit(adcp_g[var])

        range_flags = qc.qartod_range_test(
            adcp_g[var].values,
            (gr.fail_min, gr.fail_max),
            (gr.suspect_min, gr.suspect_max),
        )
        clim_flags = qc.qartod_climatology_test(
            adcp_g[var].values,
            adcp_g["time"].values,
            np.array(clm.monthly_mu),
            np.array(clm.monthly_std),
            fail_range,
            qc_cfg.get("climatology_suspect_std", 2),
        )

        test_names = ["gross_range_test", "climatology_test"]
        long_name  = adcp_g[var].attrs.get("long_name", var)

        executed, exec_attrs = qc.zip_flags([range_flags, clim_flags], test_names, var, long_name)
        summary,  sum_attrs  = qc.combine_flags([range_flags, clim_flags], test_names, var, long_name)

        adcp_g[f"{var}_qartod_executed"] = (["time"], executed)
        adcp_g[f"{var}_qartod_executed"].attrs = exec_attrs
        adcp_g[f"{var}_qartod_results"]  = (["time"], summary)
        adcp_g[f"{var}_qartod_results"].attrs  = sum_attrs
        print(f"  {var}: done")

    return adcp_g


def step_qartod_velocity(adcp_g: xr.Dataset, config: dict) -> xr.Dataset:
    """QARTOD WOA-bin gross range tests on 2-D velocity variables."""
    _banner("Step 8 — QARTOD velocity tests")
    qc_cfg     = config.get("qc", {})
    fail_range = tuple(qc_cfg.get("velocity_fail_range", [-5, 5]))
    sus_std    = qc_cfg.get("velocity_suspect_std", 3)
    woa_depths = qc.WOA_STANDARD_DEPTHS[qc.WOA_STANDARD_DEPTHS <= adcp_g["bin_depths"].max().item()]

    for var in ("eastward_sea_water_velocity", "northward_sea_water_velocity"):
        if var not in adcp_g:
            continue
        executed, summary, exec_attrs, sum_attrs = qc.run_velocity_qartod(
            adcp_g, var, fail_range, woa_depths, sus_std
        )
        adcp_g[f"{var}_qartod_executed"] = (["time", "bin_depths"], executed)
        adcp_g[f"{var}_qartod_executed"].attrs = exec_attrs
        adcp_g[f"{var}_qartod_results"]  = (["time", "bin_depths"], summary)
        adcp_g[f"{var}_qartod_results"].attrs  = sum_attrs
        print(f"  {var}: done")

    return adcp_g


def step_cf_alignment(adcp_g: xr.Dataset) -> xr.Dataset:
    """Apply all CF-1.11 fixes, beam stacking, and global attribute cleanup."""
    _banner("Step 9 — CF alignment")

    adcp_g = cf.combine_adcp_beam_params(adcp_g)
    adcp_g = cf.fix_parameter_names(adcp_g)
    adcp_g = cf.fix_parameter_attrs(adcp_g)
    adcp_g = cf.fix_coordinates(adcp_g)
    adcp_g = cf.fix_cf_compliance(adcp_g)

    global_attrs  = cf.clean_global_attrs(adcp_g)
    adcp_g.attrs  = global_attrs

    adcp_g = cf.add_station_id(adcp_g)
    adcp_g = cf.sanitize_attrs(adcp_g)

    return adcp_g


def step_save(adcp_final: xr.Dataset, config: dict, data_dir: str) -> str:
    """Save the final NetCDF."""
    _banner("Step 10 — Save")
    refdes  = config["refdes"]
    outpath = os.path.join(data_dir, f"{refdes}.nc")
    adcp_final.to_netcdf(outpath, format="netcdf4", engine="h5netcdf")
    print(f"  Saved → {outpath}")
    return outpath


# ── Main entry point ──────────────────────────────────────────────────────────

def run(config_path: str, no_download: bool = False, ship_dir: str | None = None) -> str:
    """
    Run the full pipeline for one instrument config.

    Parameters
    ----------
    config_path  : path to the YAML config file
    no_download  : skip the download step (use existing raw files)
    ship_dir     : path to local cruise directory for bottle data

    Returns
    -------
    outpath : path to the saved output NetCDF
    """
    config      = _load_config(config_path)
    paths_cfg   = config.get("paths", {})
    config_dir  = os.path.dirname(os.path.abspath(config_path))
    data_dir    = os.path.join(config_dir, paths_cfg.get("data_dir",    "../data/"))
    results_dir = os.path.join(config_dir, paths_cfg.get("results_dir", "../results/"))
    os.makedirs(data_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if not no_download:
        step_download(config, data_dir, ship_dir)

    adcp       = step_merge(config, data_dir)
    ctd_filled = step_calibrate_ctd(config, data_dir, results_dir)
    adcp       = step_merge_ctd(adcp, ctd_filled)
    adcp       = step_sensor_qc(adcp, config)
    adcp_g     = step_regrid(adcp, config)
    adcp_g     = step_qartod_scalar(adcp_g, config)
    adcp_g     = step_qartod_velocity(adcp_g, config)
    adcp_final = step_cf_alignment(adcp_g)
    outpath    = step_save(adcp_final, config, data_dir)

    _banner("Pipeline complete")
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end OOI ADCP processing pipeline",
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
        help="Skip download; use existing raw files in data/",
    )
    parser.add_argument(
        "--ship-dir",
        default=None,
        help="Path to local cruise directory containing Water_Sampling/ subfolders",
    )
    args = parser.parse_args()
    run(args.config, no_download=args.no_download, ship_dir=args.ship_dir)


if __name__ == "__main__":
    main()