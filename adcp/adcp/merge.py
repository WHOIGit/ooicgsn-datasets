"""
adcp.merge
==========
Merge OOI ADCP telemetered / recovered_host / recovered_inst streams into
a single xarray Dataset, applying the variable crosswalk to align names
across streams and dropping variables that are all-NaN or instrument-internal.
"""

from __future__ import annotations

import xarray as xr

from ooi_data_explorations.combine_data import combine_datasets


# ── Variable crosswalk ────────────────────────────────────────────────────────
# Maps the per-stream variable names onto a common set of canonical names.
# Only variables with a non-None inst_name are kept after renaming.
# Add rows here to support new ADCP models / firmware versions.

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
        "bin_depths_qc_summary_flag",
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
    Rename, clean, and merge the three ADCP data streams.

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

    # Apply crosswalk renames
    hdata = _apply_crosswalk(hdata, "host", cw)
    tdata = _apply_crosswalk(tdata, "tele", cw)
    # idata variable names are already canonical

    # Drop all-NaN variables before merging
    tdata = drop_null_vars(tdata)
    hdata = drop_null_vars(hdata)
    idata = drop_null_vars(idata)

    # Merge streams
    ds = combine_datasets(tdata, hdata, idata, None)

    # Drop OOI-internal and existing QC variables
    ds = drop_internal_vars(ds)

    # Drop specified deployments
    if deployments_to_drop:
        for dep in deployments_to_drop:
            ds = ds.where(ds["deployment"] != dep, drop=True)

    return ds