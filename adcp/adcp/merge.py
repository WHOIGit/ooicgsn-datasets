"""
adcp.merge
==========
Merge OOI ADCP telemetered / recovered_host / recovered_inst streams into
a single xarray Dataset, applying the variable crosswalk to align names
across streams and dropping variables that are all-NaN or instrument-internal.
"""

from __future__ import annotations

import xarray as xr
import numpy as np

from ooi_data_explorations.combine_data import combine_datasets


# ── Variable crosswalk ────────────────────────────────────────────────────────
# Maps the per-stream variable names onto a common set of more common names.
CROSSWALK: list[dict] = [
    {
        "host_name": "adcps_jln_upward_seawater_velocity2",
        "inst_name": "upward_seawater_velocity",
        "tele_name": "adcps_jln_upward_seawater_velocity2",
        "confidence": "high",
        "match_basis": "identical standard_name, data_product_identifier, units, long_name",
    },
    {
        "host_name": "adcps_jln_eastward_seawater_velocity2",
        "inst_name": "eastward_seawater_velocity",
        "tele_name": "adcps_jln_eastward_seawater_velocity2",
        "confidence": "high",
        "match_basis": "identical standard_name, data_product_identifier, units",
    },
    {
        "host_name": "adcps_jln_northward_seawater_velocity2",
        "inst_name": "northward_seawater_velocity",
        "tele_name": "adcps_jln_northward_seawater_velocity2",
        "confidence": "high",
        "match_basis": "identical standard_name, data_product_identifier, units",
    },
    {
        "host_name": "adcps_jln_error_velocity2",
        "inst_name": "error_seawater_velocity",
        "tele_name": "adcps_jln_error_velocity2",
        "confidence": "high",
        "match_basis": "identical data_product_identifier (VELPROF-EVL_L1)",
    },
    {
        "host_name": "adcps_jln_temp",
        "inst_name": "temperature",
        "tele_name": "adcps_jln_temp",
        "confidence": "high",
        "match_basis": "matching units (cdeg_C), long_name, QARTOD structure",
    },
    {
        "host_name": "adcps_jln_pitch",
        "inst_name": "pitch",
        "tele_name": "adcps_jln_pitch",
        "confidence": "high",
        "match_basis": "matching units (cdegrees), long_name",
    },
    {
        "host_name": "adcps_jln_roll",
        "inst_name": "roll",
        "tele_name": "adcps_jln_roll",
        "confidence": "high",
        "match_basis": "matching units (cdegrees), long_name",
    },
    {
        "host_name": "water_velocity_east",
        "inst_name": "water_velocity_east",
        "tele_name": "water_velocity_east",
        "confidence": "high",
        "match_basis": "identical name and attributes",
    },
    {
        "host_name": "water_velocity_north",
        "inst_name": "water_velocity_north",
        "tele_name": "water_velocity_north",
        "confidence": "high",
        "match_basis": "identical name and attributes",
    },
    {
        "host_name": "water_velocity_up",
        "inst_name": "water_velocity_up",
        "tele_name": "water_velocity_up",
        "confidence": "high",
        "match_basis": "identical name and attributes",
    },
    {
        "host_name": "error_velocity",
        "inst_name": "error_velocity",
        "tele_name": "error_velocity",
        "confidence": "high",
        "match_basis": "identical L0 variable",
    },
    {
        "host_name": "pressure",
        "inst_name": "pressure",
        "tele_name": None,
        "confidence": "medium",
        "match_basis": "matching long_name and units (daPa)",
    },
    {
        "host_name": "int_ctd_pressure",
        "inst_name": "int_ctd_pressure",
        "tele_name": None,
        "confidence": "high",
        "match_basis": "identical standard_name and data_product_identifier",
    },
    {
        "host_name": "bin_depths",
        "inst_name": "bin_depths",
        "tele_name": "bin_depths",
        "confidence": "high",
        "match_basis": "identical long_name and units",
    },
    {
        "host_name": "bin_depths_qartod_results",
        "inst_name": "bin_depths_qartod_results",
        "tele_name": "bin_depths_qartod_results",
        "confidence": "high",
        "match_basis": "identical QARTOD metadata",
    },
    {
        "host_name": "bin_depths_qartod_executed",
        "inst_name": "bin_depths_qartod_executed",
        "tele_name": "bin_depths_qartod_executed",
        "confidence": "high",
        "match_basis": "identical QARTOD metadata",
    },
    {
        "host_name": "non_zero_pressure",
        "inst_name": "non_zero_pressure",
        "tele_name": None,
        "confidence": "high",
        "match_basis": "identical long_name and units",
    },
    {
        "host_name": "non_zero_depth",
        "inst_name": "non_zero_depth",
        "tele_name": None,
        "confidence": "high",
        "match_basis": "identical long_name and units",
    },
    {
        "host_name": "depth_from_pressure",
        "inst_name": "depth_from_pressure",
        "tele_name": None,
        "confidence": "high",
        "match_basis": "identical long_name and units",
    },
    {
        "host_name": "depth",
        "inst_name": "depth",
        "tele_name": None,
        "confidence": "medium",
        "match_basis": "matching long_name and units",
    },
    {
        "host_name": "deployment",
        "inst_name": "deployment",
        "tele_name": "deployment",
        "confidence": "high",
        "match_basis": "identical name",
    },
    # Time-component metadata variables — no inst equivalent; dropped after merge
    {"host_name": "adcps_jln_hour",   "inst_name": None, "tele_name": "adcps_jln_hour",   "confidence": "none", "match_basis": "time metadata only"},
    {"host_name": "adcps_jln_minute", "inst_name": None, "tele_name": "adcps_jln_minute", "confidence": "none", "match_basis": "time metadata only"},
    {"host_name": "adcps_jln_second", "inst_name": None, "tele_name": "adcps_jln_second", "confidence": "none", "match_basis": "time metadata only"},
    {"host_name": "adcps_jln_year",   "inst_name": None, "tele_name": "adcps_jln_year",   "confidence": "none", "match_basis": "time metadata only"},
    {"host_name": "adcps_jln_month",  "inst_name": None, "tele_name": "adcps_jln_month",  "confidence": "none", "match_basis": "time metadata only"},
    {"host_name": "adcps_jln_day",    "inst_name": None, "tele_name": "adcps_jln_day",    "confidence": "none", "match_basis": "time metadata only"},
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_crosswalk(
    ds: xr.Dataset,
    stream: str,
    crosswalk: list[dict],
) -> xr.Dataset:
    """
    Rename variables in ``ds`` according to the crosswalk for ``stream``
    (one of 'host', 'tele', 'inst').
    """
    name_key = f"{stream}_name"
    rename_map = {}
    for row in crosswalk:
        if row["inst_name"] is None:
            continue
        old = row.get(name_key)
        new = row["inst_name"]
        if old and old != new and old in ds.variables:
            rename_map[old] = new
    if rename_map:
        ds = ds.rename_vars(rename_map)
    return ds


def _deduplicate_times(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove duplicate timestamps, preferring data in this order:
    recovered_inst > recovered_host > telemetered.

    Expects a 'stream' variable on the time axis encoding source priority.
    Falls back to keeping the first occurrence if 'stream' is absent.
    """
    times = ds["time"].values
    unique_times, counts = np.unique(times, return_counts=True)

    if not (counts > 1).any():
        return ds  # no duplicates — fast exit

    if "stream" in ds:
        # Lower number = higher priority
        priority_map = {"recovered_inst": 0, "recovered_host": 1, "telemetered": 2}
        priority = np.array([
            priority_map.get(s, 99) for s in ds["stream"].values
        ])
        # Sort by time first, then priority so that for each timestamp
        # the highest-priority record sorts to the top
        order    = np.lexsort((priority, times.astype("int64")))
    else:
        # No stream variable — sort by time only, keep first occurrence
        order = np.argsort(times.astype("int64"), kind="stable")

    ds = ds.isel(time=order)

    # Drop duplicates — after sorting, first occurrence of each timestamp
    # is the highest-priority one
    _, first_idx = np.unique(ds["time"].values, return_index=True)
    return ds.isel(time=first_idx)


def _fix_pressure_units(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert non_zero_pressure from daPa to dbar if needed.
    1 daPa = 0.001 dbar  (1 dbar = 1000 daPa)
    """
    if "non_zero_pressure" not in ds:
        return ds
    units = ds["non_zero_pressure"].attrs.get("units", "")
    if units.lower() in ("dapa", "dapa"):
        ds["non_zero_pressure"] = ds["non_zero_pressure"] / 1000.0
        ds["non_zero_pressure"].attrs["units"] = "dbar"
    return ds


# ── Public API ────────────────────────────────────────────────────────────────

def drop_null_vars(ds: xr.Dataset) -> xr.Dataset:
    """Drop variables that are entirely NaN."""
    return ds.drop_vars([v for v in ds.data_vars if ds[v].isnull().all()])


def drop_internal_vars(ds: xr.Dataset) -> xr.Dataset:
    """
    Drop OOI-internal variable names (adcps_* prefix) and pre-existing
    QC / QARTOD variables that will be recomputed.
    """
    drop = [
        v for v in ds.variables
        if "adcps_" in v or "qc" in v or "qartod" in v
    ]
    return ds.drop_vars(drop, errors="ignore")


def drop_processing_vars(ds: xr.Dataset, extra: list[str] | None = None) -> xr.Dataset:
    """
    Drop intermediate or redundant variables that are not needed in the
    final output.

    Parameters
    ----------
    extra : additional variable names to drop beyond the defaults
    """
    defaults = [
        "temperature", "salinity", "internal_timestamp",
        "non_zero_pressure", "pressure", "depth", "depth_from_pressure",
        "int_ctd_pressure", "non_zero_depth", "num_cells", "ensemble_number",
        "transducer_depth", "sysconfig_vertical_orientation",
        "ctdmo_ghqr_imodem_instrument_recovered-depth",
        "bin_depths_qc_summary_flag", "ctdmo_ghqr_sio_mule_instrument-depth",
        "subsampling_parameter", "sio_controller_timestamp", "velocity_po_up_flag",
        "velocity_po_error_flag", "velocity_po_north_flag", "unit_id",
        "firmware_version", "velocity_po_east_flag", "firmware_revision"
    ]
    to_drop = defaults + (extra or [])
    return ds.drop_vars([v for v in to_drop if v in ds.variables])


def merge_adcp_streams(
    tdata: xr.Dataset,
    hdata: xr.Dataset,
    idata: xr.Dataset,
    crosswalk: list[dict] | None = None,
    deployments_to_drop: list[int] | None = None,
) -> xr.Dataset:
    """
    Parameters
    ----------
    tdata               : telemetered Dataset
    hdata               : recovered_host Dataset
    idata               : recovered_inst Dataset
    crosswalk           : variable crosswalk list; defaults to module CROSSWALK
    deployments_to_drop : deployment numbers to exclude (e.g. incomplete metadata)

    Returns
    -------
    xr.Dataset : merged, cleaned Dataset on a unified time axis
    """
    cw = crosswalk or CROSSWALK
    hdata = _apply_crosswalk(hdata, "host", cw)
    tdata = _apply_crosswalk(tdata, "tele", cw)
    hdata = drop_null_vars(hdata)
    tdata = drop_null_vars(tdata)
    idata = drop_null_vars(idata)

    # Tag priority — lower = higher priority
    idata["_priority"] = xr.Variable("time", np.zeros(idata.sizes["time"], dtype=np.int8))
    hdata["_priority"] = xr.Variable("time", np.ones(hdata.sizes["time"],  dtype=np.int8))
    tdata["_priority"] = xr.Variable("time", np.full(tdata.sizes["time"], 2, dtype=np.int8))

    # Reset bin coord to plain 0-indexed integers so outer join
    # pads correctly across streams with different bin counts
    idata = idata.assign_coords(bin=np.arange(idata.sizes["bin"]))
    hdata = hdata.assign_coords(bin=np.arange(hdata.sizes["bin"]))
    tdata = tdata.assign_coords(bin=np.arange(tdata.sizes["bin"]))

    # Fix pressure if needed
    idata = _fix_pressure_units(idata)
    hdata = _fix_pressure_units(hdata)
    tdata = _fix_pressure_units(tdata)

    # Concatenate all streams — time axis will have duplicates
    combined = xr.concat(
        [idata, hdata, tdata],
        dim="time",
        join="outer",
        combine_attrs="override",
    )

    # Sort by time then priority so best source sorts first per timestamp
    order    = np.lexsort((combined["_priority"].values, combined["time"].values.astype("int64")))
    combined = combined.isel(time=order)

    # np.unique with return_index=True returns the first occurrence of each
    # unique time — since we sorted by priority, first = highest priority.
    # This replaces groupby().first() with a single O(n log n) numpy call.
    _, first_idx = np.unique(combined["time"].values, return_index=True)
    ds = combined.isel(time=first_idx)
    
    ds = ds.drop_vars("_priority", errors="ignore")
    ds = drop_internal_vars(ds)

    if deployments_to_drop:
        mask = ~np.isin(ds["deployment"].values, deployments_to_drop)
        ds   = ds.isel(time=mask)

    return ds