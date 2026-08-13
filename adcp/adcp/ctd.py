"""
adcp.ctd
========
Merge CTD data with ADCP, apply drift calibration against ship bottle
data, gap-fill, and recalculate sound speed at the transducer.

This module is a wrapper over ``ctd_calibration`` and
``adcp.physics``; the calibration mathematics live in ``ctd_calibration.py``.
"""

from __future__ import annotations

import re
import numpy as np
import xarray as xr

from .physics import chen_millero

# Import ctd_calibration — expected to be on sys.path (e.g. in utils/)
try:
    import ctd_calibration
except ImportError as e:
    raise ImportError(
        "ctd_calibration.py must be importable. "
        "Add its directory to sys.path or install it as a package."
    ) from e


def calibrate_ctd(
    ctd: xr.Dataset,
    bottle_path: str,
    output_dir: str = "results/",
    mooring_pressure: float = 500.0,
    mooring_pressure_window: float = 100.0,
    mooring_pattern: str = "SUMO",
    verbose: bool = True,
) -> tuple[xr.Dataset, dict, object]:
    """
    Apply drift calibration to co-located CTD data against ship bottle samples.

    Parameters
    ----------
    ctd                     : merged CTD Dataset
    bottle_path             : path to the cleaned bottle CSV
    output_dir              : directory for validation figures and CSV
    mooring_pressure        : nominal mooring depth (dbar) for bottle matching
    mooring_pressure_window : ± tolerance (dbar)
    mooring_pattern         : regex pattern matched (case-insensitive) against
                              the 'Target Asset' column of the bottle CSV to
                              identify samples from this mooring. Loaded from
                              config['mooring_pattern'] by the pipeline.
    verbose                 : print calibration report

    Returns
    -------
    cal_ctd     : Dataset with calibrated temperature and salinity added
    corrections : dict {deployment_id: DeploymentCorrection}
    stats       : per-comparison-point validation DataFrame
    """
    # Override the module-level pattern in ctd_calibration before calling it.
    # This avoids modifying ctd_calibration.py itself while still supporting
    # arbitrary mooring identifiers from the config.
    ctd_calibration.SUMO_PATTERN = re.compile(mooring_pattern, re.IGNORECASE)

    cal_ctd, corrections, stats = ctd_calibration.calibrate_insitu_ctd_ds(
        ds                      = ctd,
        water_samp_path         = bottle_path,
        output_dir              = output_dir,
        mooring_pressure        = mooring_pressure,
        mooring_pressure_window = mooring_pressure_window,
        verbose                 = verbose,
    )
    return cal_ctd, corrections, stats


def merge_ctd_into_adcp(
    adcp: xr.Dataset,
    ctd: xr.Dataset,
) -> xr.Dataset:
    """
    Interpolate CTD temperature and salinity onto the ADCP time axis,
    add them to the ADCP Dataset with CF-compliant attributes, update
    sea_water_pressure from the ADCP's internal sensor, and recalculate
    sound speed.

    Parameters
    ----------
    adcp : ADCP Dataset (any stage of processing)
    ctd  : calibrated, gap-filled CTD Dataset; must contain
           ``sea_water_temperature_calibrated`` and
           ``sea_water_practical_salinity_calibrated``

    Returns
    -------
    adcp : copy of input with CTD variables and recalculated sound_speed added
    """
    adcp = adcp.copy()

    # ── Interpolate CTD onto ADCP time axis ───────────────────────────────────
    temperature = ctd["sea_water_temperature_calibrated"].interp_like(adcp)
    salinity    = ctd["sea_water_practical_salinity_calibrated"].interp_like(adcp)

    # ── Build CTD provenance metadata ─────────────────────────────────────────
    ctd_refdes = "-".join(filter(None, [
        ctd.attrs.get("subsite"),
        ctd.attrs.get("node"),
        ctd.attrs.get("sensor"),
    ])) or ctd.attrs.get("source", "unknown")

    base_attrs = {
        "source":      ctd_refdes,
        "description": ctd.attrs.get("Description", ""),
        "manufacturer": ctd.attrs.get("Manufacturer", ""),
        "model":        ctd.attrs.get("ModelNumber", ""),
    }

    # ── Add temperature ───────────────────────────────────────────────────────
    adcp["sea_water_temperature"] = temperature
    adcp["sea_water_temperature"].attrs = {
        "standard_name":  "sea_water_temperature",
        "long_name":      "in-situ temperature",
        "units":          "degree_Celsius",
        "units_metadata": "temperature: on_scale",
        "comment":        "Offset/drift-corrected against CTD rosette temperature",
        **base_attrs,
    }

    # ── Add salinity ──────────────────────────────────────────────────────────
    adcp["sea_water_practical_salinity"] = salinity
    adcp["sea_water_practical_salinity"].attrs = {
        "standard_name": "sea_water_practical_salinity",
        "long_name":     "practical salinity",
        "units":         "1",
        "comment":       "Offset/drift-corrected against discrete water-sample bottles",
        **base_attrs,
    }

    # ── Update pressure from ADCP internal sensor ────────────────────────────
    adcp_refdes = "-".join(filter(None, [
        adcp.attrs.get("subsite"),
        adcp.attrs.get("node"),
        adcp.attrs.get("sensor"),
    ])) or adcp.attrs.get("source", "unknown")

    adcp["sea_water_pressure"] = (["time"], adcp["non_zero_pressure"].values)
    adcp["sea_water_pressure"].attrs = {
        "standard_name": "sea_water_pressure",
        "long_name":     "sea water pressure",
        "units":         "dbar",
        "positive":      "down",
        "source":        adcp_refdes,
        "comment":       "Sea water pressure as measured by the ADCP at the transducer face.",
    }

    # ── Recalculate sound speed ───────────────────────────────────────────────
    c = np.round(
        chen_millero(
            adcp["sea_water_temperature"].values,
            adcp["sea_water_practical_salinity"].values,
            adcp["non_zero_pressure"].values,
        ),
        2,
    )
    adcp["sound_speed"] = (["time"], c)
    adcp["sound_speed"].attrs = {
        "standard_name": "speed_of_sound_in_sea_water",
        "long_name":     "speed of sound in sea water",
        "units":         "m s-1",
        "comment": (
            "Calculated from in-situ temperature, practical salinity, and pressure "
            "using the Chen & Millero (1977) formulation (UNESCO Technical Papers "
            "in Marine Science No. 44, 1983)."
        ),
    }

    return adcp