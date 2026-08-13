"""
adcp.gap_fill
=============
Thin wrapper of ocean_gap_fill so the rest of the package imports from
a consistent namespace (``from adcp.gap_fill import ...``).

The gap-filling mathematics live in ``ocean_gap_fill.py``; this module
just makes them available under the ``adcp`` package and adds a small
convenience wrapper that accepts a config dict.
"""

from __future__ import annotations

import xarray as xr
import pandas as pd

# Re-export everything from the standalone module
try:
    from ocean_gap_fill import (          # noqa: F401
        fill_dataset_gaps,
        fill_variable_gaps,
        interpolate_to_hourly,
        plot_diagnostics,
        plot_all_diagnostics,
        find_gaps,
        to_decimal_year,
        VariableFitResult,
        VARIABLE_REGISTRY,
    )
except ImportError as e:
    raise ImportError(
        "ocean_gap_fill.py must be importable. "
        "Add its directory to sys.path."
    ) from e


def fill_ctd_gaps(
    ctd: xr.Dataset,
    variables: list[str] | None = None,
    verbose: bool = True,
) -> tuple[xr.Dataset, dict]:
    """
    Convenience wrapper: gap-fill CTD temperature and salinity.

    Parameters
    ----------
    ctd       : calibrated CTD Dataset
    variables : variables to fill; defaults to calibrated T and S
    verbose   : print fit summaries

    Returns
    -------
    ctd_filled : gap-filled Dataset
    results    : dict {var: VariableFitResult}
    """
    if variables is None:
        variables = [
            "sea_water_temperature_calibrated",
            "sea_water_practical_salinity_calibrated",
        ]
    # Filter to variables that actually exist in the dataset
    variables = [v for v in variables if v in ctd.data_vars]

    ctd_filled, results = fill_dataset_gaps(
        ds        = ctd,
        variables = variables,
        verbose   = verbose,
    )
    return ctd_filled, results