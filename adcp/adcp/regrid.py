"""
adcp.regrid
===========
Interpolate ADCP data from native (time, bin) coordinates onto a
common (time, bin_depths) depth grid.

Exact bin_depths vary between different deployments, so
interpolation is performed independently per timestep using np.interp.

Float variables  : linear interpolation, no extrapolation (NaN outside range)
Integer variables: max of the two bracketing bins (e.g. QC flags)
Non-(time,bin) variables: passed through unchanged
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def regrid_adcp(
    ds: xr.Dataset,
    common_depths: "np.ndarray | list",
) -> xr.Dataset:
    """
    Interpolate all (time, bin) variables onto a common depth grid.

    Parameters
    ----------
    ds            : Dataset with a ``bin_depths`` variable of shape (time, bin)
                    and data variables of shape (time, bin)
    common_depths : 1-D target depth array (metres, positive down).
                    The output is sorted in *descending* order.

    Returns
    -------
    xr.Dataset on dimensions (time, bin_depths), with:
      - float variables linearly interpolated
      - integer variables filled with max of the two bracketing bins
      - 1-D (time,) variables passed through unchanged
      - all variable attributes preserved
      - dataset-level attributes preserved
    """
    common_depths = np.asarray(common_depths, dtype=float)
    n_time  = ds.sizes["time"]
    n_depth = len(common_depths)

    bin_depth = ds["bin_depths"].values  # (time, bin)

    # ── Classify 2-D variables ────────────────────────────────────────────────
    two_dim_vars = [
        v for v in ds.data_vars
        if ds[v].dims == ("time", "bin") and v != "bin_depths"
    ]
    int_vars   = [v for v in two_dim_vars if np.issubdtype(ds[v].dtype, np.integer)]
    float_vars = [v for v in two_dim_vars if not np.issubdtype(ds[v].dtype, np.integer)]

    out       = {}
    out_attrs = {}

    # ── Float variables: linear interpolation ─────────────────────────────────
    for var in float_vars:
        values = ds[var].values
        result = np.full((n_time, n_depth), np.nan, dtype=float)

        for i in range(n_time):
            depths_i = bin_depth[i]
            vals_i   = values[i]

            valid = np.isfinite(depths_i) & np.isfinite(vals_i)
            if valid.sum() < 2:
                continue

            d, v  = depths_i[valid], vals_i[valid]
            order = np.argsort(d)
            d, v  = d[order], v[order]

            result[i] = np.interp(common_depths, d, v, left=np.nan, right=np.nan)

        out[var]       = (["time", "bin_depths"], result)
        out_attrs[var] = ds[var].attrs

    # ── Integer variables: max of two bracketing bins ─────────────────────────
    for var in int_vars:
        values = ds[var].values.astype(float)   # float so NaN is representable
        result = np.full((n_time, n_depth), np.nan, dtype=float)

        for i in range(n_time):
            depths_i = bin_depth[i]
            vals_i   = values[i]

            valid = np.isfinite(depths_i) & np.isfinite(vals_i)
            if valid.sum() < 2:
                continue

            d, v  = depths_i[valid], vals_i[valid]
            order = np.argsort(d)
            d, v  = d[order], v[order]

            idx = np.searchsorted(d, common_depths)
            for j, target_depth in enumerate(common_depths):
                if target_depth < d[0] or target_depth > d[-1]:
                    continue  # outside range — leave as NaN
                lo, hi = idx[j] - 1, idx[j]
                if lo < 0:
                    result[i, j] = v[hi]
                elif hi >= len(v):
                    result[i, j] = v[lo]
                else:
                    result[i, j] = max(v[lo], v[hi])

        out[var]       = (["time", "bin_depths"], result)
        out_attrs[var] = ds[var].attrs

    # ── Sort output depths descending ─────────────────────────────────────────
    sort_idx      = np.argsort(common_depths)[::-1]
    common_depths = common_depths[sort_idx]
    for var in out:
        dims, arr = out[var]
        out[var]  = (dims, arr[:, sort_idx])

    # ── Build Dataset ─────────────────────────────────────────────────────────
    ds_out = xr.Dataset(
        out,
        coords={
            "time"      : ds["time"],
            "bin_depths": common_depths,
        },
    )

    # Restore variable attributes
    for var, attrs in out_attrs.items():
        ds_out[var].attrs = attrs

    # Pass through non-(time, bin) variables unchanged
    non_interp_vars = [
        v for v in ds.data_vars
        if ds[v].dims != ("time", "bin") and v != "bin_depths"
    ]
    for var in non_interp_vars:
        ds_out[var] = ds[var]

    # Restore dataset attributes
    ds_out.attrs = ds.attrs

    return ds_out


def build_depth_grid(config: dict) -> np.ndarray:
    """
    Build the common depth grid from the ``regrid`` block of a config dict.

    Parameters
    ----------
    config : instrument config dict

    Returns
    -------
    np.ndarray of depths in metres
    """
    rg = config.get("regrid", {})
    return np.arange(
        rg.get("depth_min",  0),
        rg.get("depth_max",  500) + rg.get("depth_step", 10),
        rg.get("depth_step", 10),
        dtype=float,
    )
