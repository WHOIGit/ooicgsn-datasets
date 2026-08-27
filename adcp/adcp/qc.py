"""
adcp.qc
=======
QARTOD and TRDI instrument quality-control tests for OOI ADCP data.

QARTOD primitives
-----------------
qartod_range_test              — gross range test
qartod_climatology_test        — monthly climatology test
compute_gross_range_from_woa_bins — data-driven range test per WOA depth bin
zip_flags                      — combine flag arrays into per-datum strings
combine_flags                  — roll up flags to a summary flag

TRDI instrument QC
------------------
correlation_magnitude_qc
error_velocity_qc
percent_good_qc
sidelobe_qc
merge_qc

These wrap the equivalent functions in ``ooi_data_explorations.uncabled.process_adcp``
where they exist, and are implemented locally otherwise.

WOA depth grid
--------------
WOA_STANDARD_DEPTHS
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ooi_data_explorations.uncabled import process_adcp

try:
    from ooi_data_explorations.qartod import gross_range, climatology
    _HAS_OOI_QARTOD = True
except ImportError:
    _HAS_OOI_QARTOD = False


# ── Constants ─────────────────────────────────────────────────────────────────

QARTOD_FLAG_VALUES   = np.array([1, 2, 3, 4, 9], dtype=np.int8)
QARTOD_FLAG_MEANINGS = "pass not_evaluated suspect_or_of_high_interest fail missing_data"
QARTOD_REFERENCES    = "https://ioos.noaa.gov/project/qartod https://github.com/ioos/ioos_qc"

WOA_STANDARD_DEPTHS = np.array([
    0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250,
    *np.arange(300, 1600, 100),      # 300, 400, …, 1500
    1750,
    *np.arange(2000, 10500, 500),    # 2000, 2500, …, 10000
])


# ── QARTOD primitives ─────────────────────────────────────────────────────────

def qartod_range_test(
    values: np.ndarray,
    fail_range: tuple[float, float],
    suspect_range: tuple[float, float],
) -> np.ndarray:
    """
    QARTOD gross range test.

    Parameters
    ----------
    values        : data array (any shape)
    fail_range    : (min, max) — outside → flag 4
    suspect_range : (min, max) — outside (within fail) → flag 3

    Returns
    -------
    flags : int8 array, same shape as values
        1=pass  3=suspect  4=fail  9=missing
    """
    conditions = [
        ~np.isfinite(values),
        (values < fail_range[0])    | (values > fail_range[1]),
        (values < suspect_range[0]) | (values > suspect_range[1]),
    ]
    return np.select(conditions, [9, 4, 3], default=1).astype(np.int8)


def qartod_climatology_test(
    values: np.ndarray,
    times: "pd.DatetimeIndex | np.ndarray",
    clim_mean: np.ndarray,
    clim_std: np.ndarray,
    fail_range: tuple[float, float],
    suspect_scale: float = 2.0,
) -> np.ndarray:
    """
    QARTOD climatology test using monthly means and standard deviations.

    Fail bounds are fixed (``fail_range``); suspect bounds are
    ``mean ± suspect_scale * std`` for the datum's calendar month.

    Parameters
    ----------
    values        : (time,) or (time, bin) array
    times         : time axis — datetime64 or DatetimeIndex
    clim_mean     : (12,) or (12, bin) monthly means, index 0 = January
    clim_std      : (12,) or (12, bin) monthly std devs
    fail_range    : (min, max) absolute fail bounds
    suspect_scale : number of std devs for suspect threshold

    Returns
    -------
    flags : int8 array, same shape as values
    """
    times  = pd.DatetimeIndex(times)
    months = times.month - 1   # 0-indexed

    mean = clim_mean[months]
    std  = clim_std[months]

    fail_min,    fail_max    = fail_range
    suspect_min = mean - suspect_scale * std
    suspect_max = mean + suspect_scale * std

    conditions = [
        ~np.isfinite(values),
        (values < fail_min)    | (values > fail_max),
        (values < suspect_min) | (values > suspect_max),
    ]
    return np.select(conditions, [9, 4, 3], default=1).astype(np.int8)


def compute_gross_range_from_woa_bins(
    values: np.ndarray,
    bin_depths: np.ndarray,
    woa_depths: np.ndarray,
    fail_range: tuple[float, float],
    suspect_std: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """
    Data-driven gross range test: compute per-WOA-depth-bin statistics and
    apply QARTOD range flags.

    Fail bounds are fixed (``fail_range``); suspect bounds are derived from
    mean ± suspect_std × std within each WOA depth bin.

    Parameters
    ----------
    values      : (time, bin) data array
    bin_depths  : (time, bin) depth of each bin in metres
    woa_depths  : (n_woa,) WOA standard depths used as bin edges
    fail_range  : (min, max) absolute fail bounds applied globally
    suspect_std : number of std devs for suspect threshold

    Returns
    -------
    flags      : int8 array, same shape as values
    thresholds : dict mapping depth → {fail_range, suspect_range, n, mean, std}
    """
    flat_depths = bin_depths.ravel()
    flat_values = values.ravel()

    bin_indices = np.digitize(flat_depths, woa_depths) - 1  # 0-indexed
    n_woa       = len(woa_depths)
    flags       = np.full(flat_values.shape, 9, dtype=np.int8)
    thresholds  = {}

    fail_min, fail_max = fail_range

    for i in range(n_woa):
        in_bin = (bin_indices == i) & np.isfinite(flat_depths) & np.isfinite(flat_values)
        if in_bin.sum() < 2:
            continue

        bin_vals    = flat_values[in_bin]
        mean        = np.nanmean(bin_vals)
        std         = np.nanstd(bin_vals)
        suspect_min = mean - suspect_std * std
        suspect_max = mean + suspect_std * std

        thresholds[woa_depths[i]] = {
            "fail_range"    : (fail_min, fail_max),
            "suspect_range" : (suspect_min, suspect_max),
            "n"             : int(in_bin.sum()),
            "mean"          : float(mean),
            "std"           : float(std),
        }

        conditions = [
            ~np.isfinite(flat_values[in_bin]),
            (flat_values[in_bin] < fail_min)    | (flat_values[in_bin] > fail_max),
            (flat_values[in_bin] < suspect_min) | (flat_values[in_bin] > suspect_max),
        ]
        flags[in_bin] = np.select(conditions, [9, 4, 3], default=1).astype(np.int8)

    return flags.reshape(values.shape), thresholds


def zip_flags(
    flag_arrays: list[np.ndarray],
    test_names: list[str],
    var_name: str,
    long_name: str,
    coordinates: str = "time lat lon depth",
) -> tuple[np.ndarray, dict]:
    """
    Combine flag arrays into per-datum comma-separated strings.

    Parameters
    ----------
    flag_arrays  : list of equal-shape int8 arrays
    test_names   : QARTOD test names in the same order as flag_arrays
    var_name     : CF standard_name base e.g. 'sea_water_temperature'
    long_name    : descriptive variable name
    coordinates  : space-separated coordinate names for the attribute

    Returns
    -------
    zipped : string array, same shape as inputs  e.g. '1,3,4'
    attrs  : IOOS QARTOD metadata dict
    """
    assert len(flag_arrays) == len(test_names), \
        "flag_arrays and test_names must have the same length"

    str_arrays = [arr.astype(str) for arr in flag_arrays]
    result = str_arrays[0]
    for arr in str_arrays[1:]:
        result = np.char.add(np.char.add(result, ","), arr)

    attrs = {
        "tests_executed" : " ".join(test_names),
        "standard_name"  : f"{var_name} status_flag",
        "long_name"      : f"{long_name} Individual QARTOD Flags",
        "flag_values"    : QARTOD_FLAG_VALUES,
        "flag_meanings"  : QARTOD_FLAG_MEANINGS,
        "references"     : QARTOD_REFERENCES,
        "comment"        : (
            "Individual QARTOD test flags. For each datum, flags are listed in a string "
            "matching the order of the tests_executed attribute. Flags should be interpreted "
            "using the standard QARTOD mapping: [1: pass, 2: not_evaluated, "
            "3: suspect_or_of_high_interest, 4: fail, 9: missing_data]."
        ),
        "coordinates"    : coordinates,
    }
    return result, attrs


def combine_flags(
    flag_arrays: list[np.ndarray],
    test_names: list[str],
    var_name: str | None,
    long_name: str,
    coordinates: str = "time lat lon depth",
) -> tuple[np.ndarray, dict]:
    """
    Roll up multiple QARTOD flag arrays into a single summary flag.

    Takes the maximum flag value at each datum, treating 9 (missing) as
    non-competitive. Returns 9 only where every test returned 9.

    Parameters
    ----------
    flag_arrays  : list of equal-shape int8 arrays
    test_names   : QARTOD test names (used in metadata only)
    var_name     : CF standard_name base; pass None to omit standard_name
    long_name    : descriptive variable name
    coordinates  : space-separated coordinate names

    Returns
    -------
    combined : int8 array, same shape as inputs
    attrs    : IOOS QARTOD metadata dict
    """
    stack  = np.stack(flag_arrays)
    masked = np.where(stack == 9, 0, stack)
    result = masked.max(axis=0)
    result = np.where((stack == 9).all(axis=0), 9, result).astype(np.int8)

    attrs: dict = {
        "flag_values"   : QARTOD_FLAG_VALUES,
        "flag_meanings" : QARTOD_FLAG_MEANINGS,
        "long_name"     : f"{long_name} QARTOD Summary Flag",
        "references"    : QARTOD_REFERENCES,
        "comment"       : (
            "Summary QARTOD test flags. For each datum, the flag is set to the most "
            "significant result of all QARTOD tests run for that datum."
        ),
        "coordinates"   : coordinates,
    }
    if var_name is not None:
        attrs["standard_name"] = f"{var_name} status_flag"

    return result, attrs


# ── TRDI instrument QC ────────────────────────────────────────────────────────

def correlation_magnitude_qc(
    ds: xr.Dataset,
    good_threshold: int = 115,
    suspect_threshold: int = 63,
) -> np.ndarray:
    """TRDI correlation magnitude QC — wraps process_adcp."""
    return process_adcp.correlation_magnitude_qc(ds, good_threshold, suspect_threshold)


def error_velocity_qc(
    ds: xr.Dataset,
    suspect_threshold: float = 0.036,
    fail_threshold: float = 0.072,
) -> np.ndarray:
    """TRDI error velocity QC — wraps process_adcp."""
    return process_adcp.error_velocity_qc(ds, suspect_threshold, fail_threshold)


def percent_good_qc(
    ds: xr.Dataset,
    good_threshold: float = 100,
    suspect_threshold: float = 50,
) -> np.ndarray:
    """TRDI percent-good QC — wraps process_adcp."""
    return process_adcp.percent_good_qc(ds, good_threshold, suspect_threshold)


def sidelobe_qc(ds: xr.Dataset) -> np.ndarray:
    """TRDI sidelobe contamination QC — wraps process_adcp."""
    result = process_adcp.sidelobe_qc(ds)
    return result["bin_depths_qc_summary_flag"].values


def merge_qc(flag_list: list[np.ndarray]) -> np.ndarray:
    """Merge TRDI QC flag arrays — wraps process_adcp."""
    return process_adcp.merge_qc(flag_list)


# ── Sensor-engineering summary QC ─────────────────────────────────────────────

def compute_sensor_engineering_qc(
    ds: xr.Dataset,
    config: dict,
) -> tuple[np.ndarray, dict]:
    """
    Compute and combine all TRDI instrument-level QC flags (correlation
    magnitude, error velocity, percent good, roll, pitch, sidelobe) into
    a single summary QC flag array with IOOS-compatible metadata.

    Parameters
    ----------
    ds     : ADCP Dataset (pre-regrid, with bin dimension)
    config : instrument config dict (``config['qc']`` block used)

    Returns
    -------
    combined_qc    : int8 array (time, bin)
    combined_attrs : metadata dict
    """
    qc_cfg = config.get("qc", {})

    # TRDI instrument QC
    cor_mag  = correlation_magnitude_qc(ds, qc_cfg.get("correlation_good", 115),         qc_cfg.get("correlation_suspect", 63))
    err_vel  = error_velocity_qc(ds,        qc_cfg.get("error_velocity_suspect", 0.036), qc_cfg.get("error_velocity_fail", 0.072))
    per_good = percent_good_qc(ds,          qc_cfg.get("percent_good_good", 100),        qc_cfg.get("percent_good_suspect", 100))
    side     = sidelobe_qc(ds)

    # Roll / pitch — broadcast 1-D time flag to (time, bin)
    roll_fail    = tuple(qc_cfg.get("roll_fail_range",    [-2000, 2000]))
    roll_sus     = tuple(qc_cfg.get("roll_suspect_range", [-1500, 1500]))
    pitch_fail   = tuple(qc_cfg.get("pitch_fail_range",   [-2000, 2000]))
    pitch_sus    = tuple(qc_cfg.get("pitch_suspect_range",[-1500, 1500]))

    roll_qc  = qartod_range_test(ds["roll"].values,  roll_fail,  roll_sus)
    pitch_qc = qartod_range_test(ds["pitch"].values, pitch_fail, pitch_sus)
    roll_qc  = np.broadcast_to(roll_qc[:,  np.newaxis], per_good.shape).copy()
    pitch_qc = np.broadcast_to(pitch_qc[:, np.newaxis], per_good.shape).copy()

    test_names = [
        "correlation_magnitude", "error_velocity", "percent_good",
        "roll", "pitch", "sidelobe",
    ]
    combined_qc, combined_attrs = combine_flags(
        flag_arrays = [cor_mag, err_vel, per_good, roll_qc, pitch_qc, side],
        test_names  = test_names,
        var_name    = None,
        long_name   = "sensor engineering data",
    )

    # Augment metadata to reflect mixed QARTOD + TRDI provenance
    combined_attrs["tests"]      = " ".join(test_names)
    combined_attrs["references"] = ", ".join([
        QARTOD_REFERENCES,
        "TRDI ADCP Data QA-QC Model rev12-1",
    ])
    # Replace "QARTOD" with "QC" in all string attribute values
    for k, v in combined_attrs.items():
        if isinstance(v, str) and "QARTOD" in v:
            combined_attrs[k] = v.replace("QARTOD", "QC")

    return combined_qc, combined_attrs


# ── Scalar variable QARTOD ────────────────────────────────────────────────────

def run_scalar_qartod(
    ds: xr.Dataset,
    var: str,
    fail_range: tuple[float, float],
    clim_mean: np.ndarray,
    clim_std: np.ndarray,
    suspect_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Run gross range + climatology tests for a scalar (time,) variable and
    return both individual and summary flags with metadata.

    Parameters
    ----------
    ds           : Dataset containing ``var`` and ``time``
    var          : variable name
    fail_range   : (min, max) absolute fail bounds
    clim_mean    : (12,) monthly means
    clim_std     : (12,) monthly std devs
    suspect_scale: climatology suspect threshold multiplier

    Returns
    -------
    executed_flags  : per-test string array
    summary_flags   : int8 summary flag array
    executed_attrs  : metadata for the executed flags variable
    summary_attrs   : metadata for the summary flag variable
    """
    values    = ds[var].values
    times     = ds["time"].values
    long_name = ds[var].attrs.get("long_name", var)

    range_flags = qartod_range_test(
        values,
        fail_range,
        fail_range,   # use fail range as suspect range for scalars (override if needed)
    )
    clim_flags = qartod_climatology_test(
        values, times, clim_mean, clim_std, fail_range, suspect_scale
    )

    test_names = ["gross_range_test", "climatology_test"]
    executed_flags, executed_attrs = zip_flags(
        [range_flags, clim_flags], test_names, var, long_name
    )
    summary_flags, summary_attrs = combine_flags(
        [range_flags, clim_flags], test_names, var, long_name
    )
    return executed_flags, summary_flags, executed_attrs, summary_attrs


# ── 2-D velocity QARTOD ───────────────────────────────────────────────────────

def run_velocity_qartod(
    ds: xr.Dataset,
    var: str,
    fail_range: tuple[float, float],
    woa_depths: np.ndarray,
    suspect_std: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Run WOA-bin gross range test for a 2-D (time, bin_depths) velocity variable.

    Parameters
    ----------
    ds          : regridded Dataset
    var         : variable name (e.g. 'eastward_sea_water_velocity')
    fail_range  : (min, max) absolute fail bounds in m/s
    woa_depths  : WOA standard depths to use as bin edges
    suspect_std : number of std devs for data-driven suspect threshold

    Returns
    -------
    executed_flags, summary_flags, executed_attrs, summary_attrs
    """
    values     = ds[var].values
    bin_depths = np.tile(ds["bin_depths"].values, (len(ds["time"]), 1))
    long_name  = ds[var].attrs.get("long_name", var)

    flags, _ = compute_gross_range_from_woa_bins(
        values, bin_depths, woa_depths, fail_range, suspect_std
    )

    test_names = ["gross_range_test"]
    executed_flags, executed_attrs = zip_flags([flags], test_names, var, long_name)
    summary_flags,  summary_attrs  = combine_flags([flags], test_names, var, long_name)

    return executed_flags, summary_flags, executed_attrs, summary_attrs