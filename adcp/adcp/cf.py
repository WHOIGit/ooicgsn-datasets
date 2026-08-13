"""
adcp.cf
=======
CF-1.11 compliance, metadata alignment, and beam variable stacking for
OOI ADCP datasets.

Functions
---------
fix_parameter_names       — rename adcps_jln_* and seawater → sea_water
fix_parameter_attrs       — units, dtypes, and missing standard_names
fix_coordinates           — update coordinates attribute on each variable
fix_cf_compliance         — axis/positive on bin_depths, history, station var
combine_adcp_beam_params  — stack per-beam variables into (beam, time, bin_depths)
clean_global_attrs        — filter and update dataset-level attributes
add_station_id            — add scalar cf_role=timeseries_id variable
sanitize_attrs            — remove bytes / unsupported attribute types
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

try:
    from ooi_data_explorations.common import get_vocabulary
    _HAS_VOCAB = True
except ImportError:
    _HAS_VOCAB = False


# ── Attribute sanitization ────────────────────────────────────────────────────

def sanitize_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Clean dataset and variable attributes for NetCDF serialization.
    Bytes values are decoded to str; unsupported types are dropped.
    """
    def _clean(attrs: dict) -> dict:
        out = {}
        for k, v in attrs.items():
            if isinstance(v, bytes):
                out[k] = v.decode("utf-8")
            elif isinstance(v, (str, int, float, np.ndarray, list, tuple)):
                out[k] = v
            else:
                print(f"  Dropping attribute '{k}' — unsupported type {type(v)}")
        return out

    ds.attrs = _clean(ds.attrs)
    for var in ds.variables:
        ds[var].attrs = _clean(ds[var].attrs)
    return ds


# ── Variable name fixes ───────────────────────────────────────────────────────

def fix_parameter_names(ds: xr.Dataset) -> xr.Dataset:
    """
    Rename OOI-prefixed variable names and fix common naming inconsistencies.

    Transformations applied (in order):
      - Strip ``adcps_jln_`` prefix
      - Replace ``velocity2`` → ``velocity``
      - Replace ``seawater`` → ``sea_water``

    Also updates ``standard_name`` and ``ancillary_variables`` attributes
    to reflect the renames.
    """
    old_params = list(ds.data_vars)
    new_params = {
        p: p.replace("adcps_jln_", "")
            .replace("velocity2",  "velocity")
            .replace("seawater",   "sea_water")
        for p in old_params
    }

    ds = ds.rename(new_params)

    parameters = list(ds.variables)

    # Update standard_name attributes
    for param in parameters:
        if "standard_name" in ds[param].attrs:
            ds[param].attrs["standard_name"] = (
                ds[param].attrs["standard_name"]
                .replace("adcps_jln_", "")
                .replace("velocity2", "velocity")
                .replace("seawater",  "sea_water")
            )

    # Update ancillary_variables attributes
    for param in parameters:
        if "ancillary_variables" in ds[param].attrs:
            anc = ds[param].attrs["ancillary_variables"].split()
            anc = [new_params[p] if p in old_params else p for p in anc]
            anc = [p for p in anc if p in parameters]
            ds[param].attrs["ancillary_variables"] = " ".join(anc)

    return ds


# ── Variable attribute fixes ──────────────────────────────────────────────────

def fix_parameter_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Fix variable dtypes, units, and missing CF attributes.

    Changes applied
    ---------------
    - deployment → int8, minimal attrs
    - qc_flags   → fillna(9) → int8
    - bin_depths → full CF Z-axis attrs
    - time       → standard_name + long_name
    - heading / pitch / roll : divide by 100, units → degrees
    - cell_length / bin_1_distance : divide by 100, units → m
    - water_velocity_* L0 : divide by 1000, units → m s-1
    """
    ds = ds.copy()

    # Deployment
    ds["deployment"].values = ds["deployment"].values.astype(np.int8)
    ds["deployment"].attrs  = {"long_name": "Deployment Number"}

    # QC flags
    if "qc_flags" in ds:
        ds["qc_flags"] = ds["qc_flags"].fillna(9).astype(np.int8)

    # Bin depths — Z-axis identity
    ds["bin_depths"].attrs = {
        "standard_name": "depth",
        "long_name":     "Depth of Cell Center",
        "positive":      "down",
        "axis":          "Z",
        "units":         "m",
    }

    # Time
    ds["time"].attrs = {
        "standard_name":  "time",
        "long_name":      "Time",
        "units_metadata": "leap_seconds: utc",
    }

    # Orientation — convert centidegrees → degrees
    for param in ("heading", "pitch", "roll"):
        if param not in ds:
            continue
        ds[param].data = ds[param].data / 100.0
        ds[param].attrs["units"] = "degrees"
        if param != "heading":
            ds[param].attrs.pop("comment", None)

    # Cell geometry — convert cm → m
    for param in ("cell_length", "bin_1_distance"):
        if param not in ds:
            continue
        ds[param].data = ds[param].data / 100.0
        ds[param].attrs["units"] = "m"

    # L0 velocity — convert mm s⁻¹ → m s⁻¹
    for param in ("water_velocity_north", "water_velocity_east", "water_velocity_up"):
        if param not in ds:
            continue
        ds[param].data = ds[param].data / 1000.0
        ds[param].attrs["units"] = "m s-1"

    return ds


# ── Coordinate attributes ─────────────────────────────────────────────────────

def fix_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Update the ``coordinates`` attribute of every variable to match its
    actual dimension coordinates."""
    ds = ds.copy()
    for param in ds.variables:
        coords = list(ds[param].coords)
        if coords:
            ds[param].attrs["coordinates"] = " ".join(coords)
    return ds


# ── CF §2.4 / §2.6 / §9.1 compliance ────────────────────────────────────────

def fix_cf_compliance(ds: xr.Dataset) -> xr.Dataset:
    """
    Fix the three warning groups reported by the IOOS compliance checker:

    §2.4  Give bin_depths axis='Z' and positive='down' so the checker
          recognises it and stops flagging all (time, bin_depths) variables.
    §2.6  Ensure a non-empty ``history`` global attribute exists.
    §9.1  Add a scalar station variable with cf_role='timeseries_id' and
          reference it in the coordinates attribute of all 1-D time variables.
    """
    ds = ds.copy()

    # §2.4 — bin_depths Z-axis (idempotent with fix_parameter_attrs)
    ds["bin_depths"].attrs.update({
        "standard_name": "depth",
        "axis":          "Z",
        "positive":      "down",
        "units":         "m",
    })

    # §2.6 — history
    existing = ds.attrs.get("history", "")
    new_entry = f"{pd.Timestamp.now(tz='UTC').isoformat()} CF compliance fixes applied."
    ds.attrs["history"] = f"{new_entry}\n{existing}".strip()

    # §9.1 — scalar station variable
    station_id = ds.attrs.get("source", "unknown")
    ds["station"] = xr.Variable(
        [],
        station_id,
        attrs={
            "cf_role":  "timeseries_id",
            "long_name": "Station identifier",
        },
    )

    # Add 'station' to coordinates of all 1-D time variables
    one_d_vars = [v for v in ds.data_vars if ds[v].dims == ("time",)]
    for var in one_d_vars:
        coords = ds[var].attrs.get("coordinates", "time")
        if "station" not in coords:
            ds[var].attrs["coordinates"] = f"station {coords}"

    ds.attrs["featureType"] = "timeSeriesProfile"
    ds.attrs["Conventions"] = "CF-1.11"

    return ds


# ── Beam variable stacking ────────────────────────────────────────────────────

def combine_adcp_beam_params(ds: xr.Dataset) -> xr.Dataset:
    """
    Stack the four per-beam echo_intensity, corrected_echo_intensity, and
    correlation_magnitude variables into 3-D arrays with a ``beam`` dimension,
    then transpose to (beam, time, bin_depths) per CF §2.4.

    Also fixes percent_good, error_velocity units and removes the duplicate
    error_seawater_velocity variable.

    Parameters
    ----------
    ds : regridded Dataset with individual beam variables

    Returns
    -------
    ds : Dataset with stacked beam variables and cleaned attributes
    """
    ds    = ds.copy()
    beams = np.array([1, 2, 3, 4], dtype=np.int8)

    # ── Stack ─────────────────────────────────────────────────────────────────
    echo_raw  = np.stack([ds[f"echo_intensity_beam{b}"].data        for b in beams], axis=-1)
    echo_corr = np.stack([ds[f"corrected_echo_intensity_beam{b}"].data for b in beams], axis=-1)
    corr_mag  = np.stack([ds[f"correlation_magnitude_beam{b}"].data  for b in beams], axis=-1)

    # ── Add beam coordinate ───────────────────────────────────────────────────
    ds = ds.assign_coords(beam=beams)   # ← uses ds, not any outer-scope name
    ds["beam"].attrs = {"long_name": "ADCP Beam Number", "units": "1"}

    # ── Create stacked variables (time, bin_depths, beam) ─────────────────────
    shared_coords = "time bin_depths beam"

    ds["echo_intensity"] = (["time", "bin_depths", "beam"], echo_raw)
    ds["echo_intensity"].attrs = {
        "standard_name": "signal_intensity_from_multibeam_acoustic_doppler_velocity_sensor_in_sea_water",
        "long_name":     "Echo Intensity",
        "units":         "counts",
        "coordinates":   shared_coords,
        "comment": (
            "Raw acoustic return signal per beam, 0–255 counts. "
            "Echo intensity can be used as an indicator of sediment or organisms "
            "in the water column and as a proxy for measurement quality."
        ),
    }

    ds["corrected_echo_intensity"] = (["time", "bin_depths", "beam"], echo_corr)
    ds["corrected_echo_intensity"].attrs = {
        "standard_name": "signal_intensity_from_multibeam_acoustic_doppler_velocity_sensor_in_sea_water",
        "long_name":     "Echo Intensity Corrected for Spreading and Attenuation",
        "units":         "dB",
        "coordinates":   shared_coords,
        "comment": (
            "Echo intensity corrected for beam-spreading and water-column attenuation."
        ),
    }

    ds["correlation_magnitude"] = (["time", "bin_depths", "beam"], corr_mag)
    ds["correlation_magnitude"].attrs = {
        "standard_name": "beam_consistency_indicator_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water",
        "long_name":     "Correlation Magnitude",
        "units":         "counts",
        "coordinates":   shared_coords,
        "comment": (
            "Magnitude of the normalised echo auto-correlation at the lag used for "
            "estimating the Doppler phase change. 0 = bad, 255 = perfect correlation."
        ),
    }

    # ── Transpose to CF-recommended order (beam left of T, Z) ────────────────
    for var in ("echo_intensity", "corrected_echo_intensity", "correlation_magnitude"):
        ds[var] = ds[var].transpose("beam", "time", "bin_depths")

    # ── Remove individual beam variables ─────────────────────────────────────
    drop = (
        [f"echo_intensity_beam{b}"         for b in beams] +
        [f"corrected_echo_intensity_beam{b}" for b in beams] +
        [f"correlation_magnitude_beam{b}"   for b in beams]
    )
    ds = ds.drop_vars([v for v in drop if v in ds])

    # ── Fix percent_good / percent_bad units ──────────────────────────────────
    for var, sn, ln in [
        ("percent_good_3beam", "proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water", "Percent Good 3 Beam Solutions"),
        ("percent_good_4beam", "proportion_of_acceptable_signal_returns_from_acoustic_instrument_in_sea_water", "Percent Good 4 Beam Solutions"),
        ("percent_bad_beams",  None, None),
        ("percent_transforms_reject", None, None),
    ]:
        if var not in ds:
            continue
        ds[var].attrs["units"] = "1"
        if sn:
            ds[var].attrs["standard_name"] = sn
        if ln:
            ds[var].attrs["long_name"] = ln

    # ── Fix error velocity ────────────────────────────────────────────────────
    if "error_velocity" in ds:
        ds["error_velocity"].data = ds["error_velocity"].data / 1000.0
        ds["error_velocity"].attrs["standard_name"] = (
            "indicative_error_from_multibeam_acoustic_doppler_velocity_profiler_in_sea_water"
        )
        ds["error_velocity"].attrs["units"] = "m s-1"

    # Drop duplicate error_seawater_velocity (L1 duplicate of error_velocity)
    ds = ds.drop_vars(["error_seawater_velocity"], errors="ignore")

    return ds


# ── Global attribute cleanup ──────────────────────────────────────────────────

_GLOBAL_ATTR_KEEP = [
    "title", "comment", "sourceUrl", "featureType", "publisher_name",
    "references", "Metadata_Conventions", "nodc_template_version",
    "creator_name", "history", "standard_name_vocabulary", "acknowledgement",
    "project", "source", "subsite", "node", "sensor",
    "Manufacturer", "ModelNumber", "Description",
    "time_coverage_start", "time_coverage_end",
    "geospatial_lat_min", "geospatial_lat_max",
    "geospatial_lat_units", "geospatial_lat_resolution",
    "geospatial_lon_min", "geospatial_lon_max",
    "geospatial_lon_units", "geospatial_lon_resolution",
    "geospatial_vertical_units", "geospatial_vertical_resolution",
    "geospatial_vertical_positive", "lat", "lon",
]


def _make_title(ds: xr.Dataset) -> str:
    """Build a dataset title from OOI vocabulary if available."""
    if not _HAS_VOCAB:
        return " : ".join(filter(None, [
            ds.attrs.get("subsite"),
            ds.attrs.get("node"),
            ds.attrs.get("sensor"),
        ]))
    subsite = ds.attrs.get("subsite", "")
    node    = ds.attrs.get("node", "")
    sensor  = ds.attrs.get("sensor", "")
    try:
        vocab = get_vocabulary(subsite, node, sensor)[0]
        return " : ".join([vocab["tocL1"], vocab["tocL2"], vocab["tocL3"], vocab["instrument"]])
    except Exception:
        return f"{subsite}-{node}-{sensor}"


def clean_global_attrs(ds: xr.Dataset) -> dict:
    """
    Return a cleaned global attribute dict suitable for CF-compliant output.

    Keeps a curated subset of OOI global attributes, updates time coverage,
    sets Conventions and featureType, and appends a history entry.
    """
    old = ds.attrs.copy()

    new = {k: old[k] for k in _GLOBAL_ATTR_KEEP if k in old}

    # Update time coverage
    new["time_coverage_start"] = str(ds["time"].min().values) + "Z"
    new["time_coverage_end"]   = str(ds["time"].max().values) + "Z"

    # Shorten source to refdes only (drop method suffix)
    if "source" in old:
        new["source"] = "-".join(old["source"].split("-")[:4])

    new["Conventions"]  = "CF-1.11"
    new["featureType"]  = "timeSeriesProfile"
    new["title"]        = _make_title(ds)

    # Append history entry
    existing = new.get("history", "")
    entry    = f"{pd.Timestamp.now(tz='UTC').isoformat()} Regridded and quality-control applied."
    new["history"] = f"{entry}\n{existing}".strip()

    return new


# ── Station variable ──────────────────────────────────────────────────────────

def add_station_id(ds: xr.Dataset) -> xr.Dataset:
    """
    Add a scalar station variable with cf_role='timeseries_id' and reference
    it in the ``coordinates`` attribute of all 1-D time variables.
    """
    ds = ds.copy()
    site, node, sensor = (
        ds.attrs.get("subsite", ""),
        ds.attrs.get("node",    ""),
        ds.attrs.get("sensor",  ""),
    )
    station_id = "-".join(filter(None, [site, node, sensor])) or ds.attrs.get("source", "unknown")

    ds["station"] = xr.Variable(
        [],
        station_id,
        attrs={
            "cf_role":  "timeseries_id",
            "long_name": "Reference Designator",
        },
    )

    one_d_vars = [v for v in ds.data_vars if ds[v].dims == ("time",)]
    for var in one_d_vars:
        coords = ds[var].attrs.get("coordinates", "time")
        if "station" not in coords:
            ds[var].attrs["coordinates"] = f"station {coords}"

    return ds