"""
Ocean Time-Series Gap-Filling: Piecewise Linear Trend + Seasonal Harmonics
===========================================================================
Fits an independent model to each requested variable in an xarray Dataset:

    V(t) = β₀ + β₁·t + β₂·max(0, t − t_break)
         + A₁·sin(2πt) + B₁·cos(2πt)
         + A₂·sin(4πt) + B₂·cos(4πt)

where t is decimal year and t_break is found via grid search independently
for each variable (temperature and salinity inflections need not coincide).

Gaps are identified as time steps exceeding a threshold; both large gaps and
isolated NaN values are filled by evaluating the model at those times.

Input:  xarray.Dataset with a shared time dimension and one or more variables
Output: xarray.Dataset with all variables gap-filled + per-variable source flags
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.linalg import lstsq
from dataclasses import dataclass, field


# ── 1. Configuration ──────────────────────────────────────────────────────────

# Default variable definitions: maps variable name → display metadata.
# Extend or override this dict for your own dataset.
VARIABLE_REGISTRY: dict[str, dict] = {
    "sea_water_practical_salinity": {
        "label":     "Salinity",
        "units":     "PSU",
        "color":     "#3a78b5",
        "slope_fmt": "{:.3f} m-PSU/yr",   # multiply slope by 1000
        "slope_scale": 1000,
    },
    "sea_water_temperature": {
        "label":     "Temperature",
        "units":     "°C",
        "color":     "#c0392b",
        "slope_fmt": "{:.4f} °C/yr",
        "slope_scale": 1,
    },
}

TIME_DIM              = "time"
GAP_THRESHOLD         = pd.Timedelta("1D")
FILL_FREQ             = "7.5min"
BREAKPOINT_STEP       = 0.25   # decimal years
BREAKPOINT_MIN_OFFSET = 1.0    # years from record start before first candidate
BREAKPOINT_MAX_OFFSET = 1.0    # years from record end  after  last  candidate


# ── 2. Result container ───────────────────────────────────────────────────────

@dataclass
class VariableFitResult:
    """Stores all model output for a single fitted variable."""
    var_name:               str
    breakpoint_decimal_year: float
    coefficients:           np.ndarray   # [β₀, β₁, β₂, A₁, B₁, A₂, B₂]
    rmse:                   float
    slope1:                 float        # trend before breakpoint  (units/yr)
    slope2:                 float        # trend after  breakpoint  (units/yr)
    residuals:              np.ndarray
    gaps:                   list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]]
    rss_candidates:         np.ndarray
    rss_values:             np.ndarray
    t0_decimal:             float        # t[0] used as reference in design matrix
    n_nan_filled:           int = 0
    n_gap_filled:           int = 0

    def summary(self, registry: dict = VARIABLE_REGISTRY) -> str:
        meta   = registry.get(self.var_name, {})
        label  = meta.get("label", self.var_name)
        units  = meta.get("units", "")
        scale  = meta.get("slope_scale", 1)
        fmt    = meta.get("slope_fmt", "{:.5f} /yr")
        bp_dt  = _decimal_year_to_timestamp(self.breakpoint_decimal_year)
        lines  = [
            f"  [{label}]",
            f"    Breakpoint  : {self.breakpoint_decimal_year:.3f}  (~{bp_dt.strftime('%Y-%m')})",
            f"    Slope 1     : {fmt.format(self.slope1 * scale)}",
            f"    Slope 2     : {fmt.format(self.slope2 * scale)}",
            f"    RMSE        : {self.rmse:.6f} {units}",
            f"    NaNs filled : {self.n_nan_filled}",
            f"    Gap points  : {self.n_gap_filled}",
        ]
        return "\n".join(lines)


# ── 3. Core mathematical helpers ──────────────────────────────────────────────

def to_decimal_year(dt_index: pd.DatetimeIndex) -> np.ndarray:
    """Convert a DatetimeIndex to an array of decimal years."""
    years  = dt_index.year
    starts = pd.DatetimeIndex([pd.Timestamp(y, 1, 1) for y in years])
    ends   = pd.DatetimeIndex([pd.Timestamp(y + 1, 1, 1) for y in years])
    frac   = (dt_index - starts).total_seconds() / (ends - starts).total_seconds()
    return years.values + frac.values


def _decimal_year_to_timestamp(t: float) -> pd.Timestamp:
    """Approximate inverse of to_decimal_year (used for display only)."""
    return pd.Timestamp("2000-01-01") + pd.to_timedelta((t - 2000) * 365.25, unit="D")


def build_design_matrix(t: np.ndarray, t_break: float) -> np.ndarray:
    """
    Build the 7-column regression design matrix.

    Columns
    -------
    0  intercept
    1  linear trend      (t - t[0])
    2  hockey-stick term max(0, t - t_break)
    3  sin(2πt)   annual harmonic
    4  cos(2πt)
    5  sin(4πt)   semi-annual harmonic
    6  cos(4πt)
    """
    return np.column_stack([
        np.ones_like(t),
        t - t[0],
        np.maximum(0.0, t - t_break),
        np.sin(2 * np.pi * t),
        np.cos(2 * np.pi * t),
        np.sin(4 * np.pi * t),
        np.cos(4 * np.pi * t),
    ])


def find_optimal_breakpoint(
    t: np.ndarray,
    y: np.ndarray,
    step:       float = BREAKPOINT_STEP,
    min_offset: float = BREAKPOINT_MIN_OFFSET,
    max_offset: float = BREAKPOINT_MAX_OFFSET,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Grid-search for the breakpoint that minimises RSS.

    Returns
    -------
    best_tb    : optimal breakpoint (decimal year)
    candidates : all candidate values tested
    rss_vals   : RSS at each candidate
    """
    candidates = np.arange(t[0] + min_offset, t[-1] - max_offset, step)
    rss_vals   = np.empty(len(candidates))
    for i, tb in enumerate(candidates):
        X = build_design_matrix(t, tb)
        c, _, _, _ = lstsq(X, y)
        rss_vals[i] = np.sum((y - X @ c) ** 2)
    best_tb = candidates[np.argmin(rss_vals)]
    return best_tb, candidates, rss_vals


def fit_model(
    t: np.ndarray,
    y: np.ndarray,
    t_break: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit the model for a fixed breakpoint via ordinary least squares.

    Returns
    -------
    coeffs    : (7,) coefficient array
    residuals : observed − predicted
    """
    X = build_design_matrix(t, t_break)
    coeffs, _, _, _ = lstsq(X, y)
    return coeffs, y - X @ coeffs


def predict(
    t:       np.ndarray,
    t_break: float,
    coeffs:  np.ndarray,
    seasonal: bool = True,
) -> np.ndarray:
    """
    Evaluate the fitted model at arbitrary times t.

    Parameters
    ----------
    seasonal : include harmonic terms (True) or return trend-only (False)
    """
    X = build_design_matrix(t, t_break)
    if not seasonal:
        X[:, 3:] = 0.0
    return X @ coeffs


# ── 4. Gap detection ──────────────────────────────────────────────────────────

def find_gaps(
    time_index: pd.DatetimeIndex,
    threshold:  pd.Timedelta = GAP_THRESHOLD,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]]:
    """
    Identify time gaps larger than `threshold`.

    Returns list of (gap_start, gap_end, gap_duration).
    """
    diffs = time_index[1:] - time_index[:-1]
    return [
        (time_index[i], time_index[i + 1], d)
        for i, d in enumerate(diffs)
        if d > threshold
    ]


# ── 5. Single-variable gap-filling ───────────────────────────────────────────

def fill_variable_gaps(
    time_idx:        pd.DatetimeIndex,
    values:          np.ndarray,
    var_name:        str,
    gaps:            list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]],
    fill_freq:       str   = FILL_FREQ,
    breakpoint_step: float = BREAKPOINT_STEP,
    verbose:         bool  = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, VariableFitResult]:
    """
    Fit the model and fill gaps for a single 1-D variable array.

    The gap list is shared across all variables so synthetic timestamps
    are consistent; each variable gets its own independent breakpoint.

    Parameters
    ----------
    time_idx   : DatetimeIndex of the observed time axis
    values     : 1-D array of observed values (NaNs allowed)
    var_name   : variable name string (used for printing)
    gaps       : pre-computed list of large time gaps (from find_gaps)
    fill_freq  : pandas frequency string for synthetic gap timestamps
    breakpoint_step : grid search resolution in decimal years
    verbose    : print fit summary

    Returns
    -------
    new_time   : merged + sorted DatetimeIndex (obs + gap-fill points)
    new_vals   : corresponding filled values
    new_flags  : int8 source flag (0=observed, 1=model-filled)
    result     : VariableFitResult with all model metadata
    """
    t_all = to_decimal_year(time_idx)
    vals  = values.copy().astype(float)

    # --- Fit on valid data ---
    valid   = np.isfinite(vals)
    t_valid = t_all[valid]
    y_valid = vals[valid]

    best_tb, candidates, rss_vals = find_optimal_breakpoint(
        t_valid, y_valid, step=breakpoint_step
    )
    coeffs, residuals = fit_model(t_valid, y_valid, best_tb)
    rmse   = np.sqrt(np.mean(residuals ** 2))
    slope1 = coeffs[1]
    slope2 = coeffs[1] + coeffs[2]

    # --- Patch isolated NaNs in-place ---
    nan_mask = ~valid
    if nan_mask.any():
        vals[nan_mask] = predict(t_all[nan_mask], best_tb, coeffs)

    source_flag = np.zeros(len(time_idx), dtype=np.int8)
    source_flag[nan_mask] = 1

    # --- Build synthetic gap segments ---
    new_time_chunks  = [time_idx]
    new_vals_chunks  = [vals]
    new_flags_chunks = [source_flag]
    n_gap_filled     = 0

    for g0, g1, _ in gaps:
        t_fill_dt = pd.date_range(
            g0 + pd.tseries.frequencies.to_offset(fill_freq),
            g1 - pd.tseries.frequencies.to_offset(fill_freq),
            freq=fill_freq,
        )
        if len(t_fill_dt) == 0:
            continue
        t_fill_dec = to_decimal_year(t_fill_dt)
        y_fill     = predict(t_fill_dec, best_tb, coeffs)
        new_time_chunks.append(t_fill_dt)
        new_vals_chunks.append(y_fill)
        new_flags_chunks.append(np.ones(len(t_fill_dt), dtype=np.int8))
        n_gap_filled += len(t_fill_dt)

    # --- Concatenate and sort ---
    new_time  = pd.DatetimeIndex(np.concatenate(new_time_chunks))
    new_vals  = np.concatenate(new_vals_chunks)
    new_flags = np.concatenate(new_flags_chunks)
    sort_idx  = np.argsort(new_time)

    result = VariableFitResult(
        var_name               = var_name,
        breakpoint_decimal_year= best_tb,
        coefficients           = coeffs,
        rmse                   = rmse,
        slope1                 = slope1,
        slope2                 = slope2,
        residuals              = residuals,
        gaps                   = gaps,
        rss_candidates         = candidates,
        rss_values             = rss_vals,
        t0_decimal             = t_valid[0],
        n_nan_filled           = int(nan_mask.sum()),
        n_gap_filled           = n_gap_filled,
    )

    if verbose:
        print(result.summary())

    return (
        new_time[sort_idx],
        new_vals[sort_idx],
        new_flags[sort_idx],
        result,
    )


# ── 6. Multi-variable orchestrator ───────────────────────────────────────────

def fill_dataset_gaps(
    ds:              xr.Dataset,
    variables:       list[str] | None = None,
    time_dim:        str               = TIME_DIM,
    gap_threshold:   pd.Timedelta      = GAP_THRESHOLD,
    fill_freq:       str               = FILL_FREQ,
    breakpoint_step: float             = BREAKPOINT_STEP,
    verbose:         bool              = True,
) -> tuple[xr.Dataset, dict[str, VariableFitResult]]:
    """
    Fill gaps in multiple co-located variables within an xarray Dataset.

    Each variable receives an independent breakpoint search, but all
    variables share the same synthetic time grid within each gap so the
    output Dataset remains consistently aligned.

    Parameters
    ----------
    ds            : input xarray Dataset
    variables     : list of variable names to process; defaults to all
                    data variables that share `time_dim`
    time_dim      : name of the time coordinate
    gap_threshold : gaps larger than this are filled
    fill_freq     : pandas offset string for synthetic timestamps
    breakpoint_step : decimal-year grid spacing for breakpoint search
    verbose       : print per-variable fit summaries

    Returns
    -------
    ds_filled : Dataset with all requested variables gap-filled.
                Each variable gets a companion "<var>_source" flag
                (0 = observed, 1 = model-filled).
    results   : dict mapping variable name → VariableFitResult
    """
    time_idx = pd.DatetimeIndex(ds[time_dim].values)

    # Default: process all 1-D variables sharing the time dimension
    if variables is None:
        variables = [
            v for v in ds.data_vars
            if ds[v].dims == (time_dim,)
        ]

    if verbose:
        print(f"Processing {len(variables)} variable(s) over "
              f"{len(time_idx):,} time steps "
              f"({time_idx[0].date()} → {time_idx[-1].date()})")

    # Detect gaps once — shared across all variables
    gaps = find_gaps(time_idx, threshold=gap_threshold)
    if verbose:
        print(f"  Large gaps detected: {len(gaps)}")
        for g0, g1, gd in gaps:
            print(f"    {g0.date()} → {g1.date()} ({gd.days} days)")
        print()

    results: dict[str, VariableFitResult] = {}
    data_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # var → (vals, flags)
    shared_new_time: pd.DatetimeIndex | None = None

    for var in variables:
        if verbose:
            print(f"Fitting: {var}")
        values = ds[var].values

        new_time, new_vals, new_flags, result = fill_variable_gaps(
            time_idx        = time_idx,
            values          = values,
            var_name        = var,
            gaps            = gaps,
            fill_freq       = fill_freq,
            breakpoint_step = breakpoint_step,
            verbose         = verbose,
        )

        results[var]     = result
        data_arrays[var] = (new_vals, new_flags)

        # All variables yield the same new_time (gaps are shared), so
        # we only need to store it once.
        if shared_new_time is None:
            shared_new_time = new_time

        if verbose:
            print()

    # --- Assemble output Dataset ---
    xr_vars: dict[str, xr.DataArray] = {}
    for var, (vals, flags) in data_arrays.items():
        xr_vars[var] = xr.DataArray(
            vals, dims=[time_dim], attrs=ds[var].attrs
        )
        xr_vars[f"{var}_source"] = xr.DataArray(
            flags, dims=[time_dim],
            attrs={"description": "0=observed, 1=model-filled"},
        )

    ds_filled = xr.Dataset(
        xr_vars,
        coords={time_dim: shared_new_time},
        attrs=ds.attrs,
    )

    return ds_filled, results


# ── 7. Diagnostic plotting ────────────────────────────────────────────────────

def plot_diagnostics(
    ds_filled:   xr.Dataset,
    result:      VariableFitResult,
    time_dim:    str                = TIME_DIM,
    registry:    dict               = VARIABLE_REGISTRY,
    save_path:   str | None         = None,
) -> plt.Figure:
    """
    Four-panel diagnostic figure for a single variable:
      Row 0 : full time series + piecewise trend + filled gaps
      Row 1 : per-gap zoom panels
      Row 2 : breakpoint RSS curve  |  residual histogram

    Parameters
    ----------
    ds_filled  : output from fill_dataset_gaps
    result     : VariableFitResult for the variable to plot
    time_dim   : time coordinate name
    registry   : dict mapping variable names to display metadata
    save_path  : file path to save figure; None → plt.show()
    """
    var      = result.var_name
    meta     = registry.get(var, {})
    label    = meta.get("label", var)
    units    = meta.get("units", "")
    color    = meta.get("color", "#3a78b5")
    scale    = meta.get("slope_scale", 1)
    slope_fmt= meta.get("slope_fmt", "{:.5f} /yr")

    time_idx = pd.DatetimeIndex(ds_filled[time_dim].values)
    vals     = ds_filled[var].values
    flags    = ds_filled[f"{var}_source"].values
    obs_mask = flags == 0
    fil_mask = flags == 1

    t_dec    = to_decimal_year(time_idx)
    tb       = result.breakpoint_decimal_year
    coeffs   = result.coefficients
    gaps     = result.gaps

    # Trend-only line for overlay
    t_line  = np.linspace(t_dec[0], t_dec[-1], 5000)
    S_trend = predict(t_line, tb, coeffs, seasonal=False)
    dt_line = np.array([_decimal_year_to_timestamp(tt) for tt in t_line],
                       dtype="datetime64[ns]")
    bp_dt   = _decimal_year_to_timestamp(tb)

    obs_ser = (pd.Series(vals[obs_mask], index=time_idx[obs_mask])
                 .resample("D").mean())
    resids  = result.residuals

    # ── Layout ──
    n_gaps = max(len(gaps), 1)
    fig    = plt.figure(figsize=(15, 12))
    gs0    = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55,
                               top=0.93, bottom=0.07, left=0.08, right=0.97)

    # ── Row 0: Full series ──
    ax1 = fig.add_subplot(gs0[0])
    ax1.plot(obs_ser.index, obs_ser.values,
             color=color, lw=0.6, alpha=0.75, label="Observed (daily mean)")
    if fil_mask.any():
        ax1.plot(time_idx[fil_mask], vals[fil_mask],
                 color="tomato", lw=1.3, zorder=5)
        ax1.plot([], [], color="tomato", lw=2, label="Model-filled gaps")
    ax1.plot(dt_line, S_trend, "k--", lw=1.6, label="Piecewise trend", zorder=6)
    for g0, g1, _ in gaps:
        ax1.axvspan(g0, g1, alpha=0.10, color="tomato")
    ax1.axvline(bp_dt, color="darkorange", lw=1.8, ls=":",
                label=f"Breakpoint ({tb:.2f})", zorder=7)
    ax1.set_ylabel(f"{label} ({units})", fontsize=10)
    ax1.set_title(f"{label}: Full Record with Model Gap-Filling",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="best", framealpha=0.85)
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.text(0.02, 0.05,
             f"Slope₁ = {slope_fmt.format(result.slope1 * scale)}",
             transform=ax1.transAxes, fontsize=8, style="italic")
    ax1.text(0.30, 0.05,
             f"Slope₂ = {slope_fmt.format(result.slope2 * scale)}",
             transform=ax1.transAxes, fontsize=8, style="italic")

    # ── Row 1: Gap zooms ──
    gap_colors = ["#d94f3d", "#e8833a", "#c2395c", "#8e44ad", "#27ae60"]
    gs_gaps    = gridspec.GridSpecFromSubplotSpec(
        1, n_gaps, subplot_spec=gs0[1], wspace=0.4
    )
    obs_df = pd.DataFrame({"val": vals[obs_mask]}, index=time_idx[obs_mask])

    for i, (g0, g1, gd) in enumerate(gaps):
        ax  = fig.add_subplot(gs_gaps[i])
        pad = pd.Timedelta(days=min(gd.days // 2, 60))
        win_o  = obs_df[(obs_df.index >= g0 - pad) & (obs_df.index <= g1 + pad)]
        win_od = win_o["val"].resample("D").mean()
        win_f  = pd.Series(vals[fil_mask], index=time_idx[fil_mask])
        win_f  = win_f[(win_f.index >= g0) & (win_f.index <= g1)]
        gc     = gap_colors[i % len(gap_colors)]
        ax.plot(win_od.index, win_od.values, color=color, lw=1.0, label="Observed")
        ax.plot(win_f.index,  win_f.values,  color=gc,    lw=1.4, label="Filled")
        ax.axvspan(g0, g1, alpha=0.12, color=gc)
        ax.set_title(f"Gap {i+1}: {gd.days}d\n{g0.date()} – {g1.date()}", fontsize=8.5)
        ax.set_ylabel(f"{label} ({units})", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5)

    # ── Row 2: RSS curve + residual histogram ──
    gs_bot  = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[2], wspace=0.38)
    ax_rss  = fig.add_subplot(gs_bot[0])
    ax_hist = fig.add_subplot(gs_bot[1])

    ax_rss.plot(result.rss_candidates, result.rss_values, color=color, lw=1.3)
    ax_rss.axvline(tb, color="darkorange", lw=2, ls="--",
                   label=f"Optimal: {tb:.2f}")
    ax_rss.set_xlabel("Candidate breakpoint (decimal year)", fontsize=9)
    ax_rss.set_ylabel("RSS", fontsize=9)
    ax_rss.set_title("Breakpoint Grid Search", fontsize=10, fontweight="bold")
    ax_rss.legend(fontsize=9)
    ax_rss.grid(True, alpha=0.25)

    ax_hist.hist(resids, bins=100, color=color, alpha=0.8, edgecolor="none")
    ax_hist.axvline(0, color="k", lw=1)
    ax_hist.set_xlabel(f"Residual ({units})", fontsize=9)
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.set_title(f"Model Residuals  (RMSE = {result.rmse:.5f} {units})",
                      fontsize=10, fontweight="bold")
    ax_hist.grid(True, alpha=0.25)

    fig.suptitle(
        f"{label} Gap-Filling: Piecewise Linear Trend + Annual & Semi-annual Harmonics",
        fontsize=13, fontweight="bold",
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")
    else:
        plt.show()

    return fig


def plot_all_diagnostics(
    ds_filled: xr.Dataset,
    results:   dict[str, VariableFitResult],
    time_dim:  str         = TIME_DIM,
    registry:  dict        = VARIABLE_REGISTRY,
    save_dir:  str | None  = None,
) -> dict[str, plt.Figure]:
    """
    Convenience wrapper: call plot_diagnostics for every variable in results.

    Parameters
    ----------
    save_dir : directory to write figures into; filenames are auto-generated.
               None → plt.show() for each figure.
    """
    import os
    figs = {}
    for var, result in results.items():
        save_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            safe_name = var.replace(" ", "_")
            save_path = os.path.join(save_dir, f"{safe_name}_diagnostics.png")
        figs[var] = plot_diagnostics(
            ds_filled  = ds_filled,
            result     = result,
            time_dim   = time_dim,
            registry   = registry,
            save_path  = save_path,
        )
    return figs


# ── 8. Hourly interpolation ───────────────────────────────────────────────────

def interpolate_to_hourly(
    ds:         xr.Dataset,
    time_dim:   str  = TIME_DIM,
    method:     str  = "linear",
    keep_source_flags: bool = True,
) -> xr.Dataset:
    """
    Interpolate a gap-filled dataset onto an exact hourly time grid.

    Every data variable is interpolated using `method` (default: linear).
    Companion ``*_source`` flag variables are handled separately with
    nearest-neighbour so their integer values are never blended:

        0 → observed value bracketed this hour
        1 → model-filled value bracketed this hour

    Parameters
    ----------
    ds         : gap-filled xarray Dataset (e.g. output of fill_dataset_gaps)
    time_dim   : name of the time coordinate
    method     : interpolation method passed to xarray/scipy for data variables.
                 Any method accepted by ``xr.Dataset.interp`` is valid:
                 "linear", "nearest", "cubic", "quadratic", etc.
                 "linear" is recommended — the 7.5-min native resolution
                 means each hourly target is always well-bracketed and
                 higher-order methods add no meaningful benefit.
    keep_source_flags : if True (default), carry the *_source flags through
                        using nearest-neighbour; set False to drop them.

    Returns
    -------
    ds_hourly : Dataset on an exact hourly grid with the same variables
                and attributes as the input.

    Notes
    -----
    * The hourly grid spans from the first full hour at or after the record
      start to the last full hour at or before the record end, so no
      extrapolation ever occurs.
    * xarray's ``interp`` uses scipy under the hood; "linear" maps to
      ``scipy.interpolate.interp1d`` with ``kind="linear"``.
    """
    time_idx = pd.DatetimeIndex(ds[time_dim].values)

    # Build the target hourly grid — strictly inside the observed range
    # so we never extrapolate beyond the data.
    t_start = time_idx.min().ceil("h")
    t_end   = time_idx.max().floor("h")
    hourly_index = pd.date_range(t_start, t_end, freq="h")

    if len(hourly_index) == 0:
        raise ValueError(
            f"No full hours found between {time_idx.min()} and {time_idx.max()}."
        )

    # Separate data variables from *_source flag variables
    source_vars = [v for v in ds.data_vars if str(v).endswith("_source")]
    data_vars   = [v for v in ds.data_vars if v not in source_vars]

    # ── Interpolate data variables ────────────────────────────────────────────
    ds_data = ds[data_vars].interp(
        {time_dim: hourly_index},
        method=method,
        kwargs={"fill_value": "extrapolate"} if method == "linear" else {},
    )

    # ── Nearest-neighbour for source flags ────────────────────────────────────
    if keep_source_flags and source_vars:
        ds_flags = ds[source_vars].interp(
            {time_dim: hourly_index},
            method="nearest",
        )
        # Round and cast back to int8 in case floating-point crept in
        for v in source_vars:
            ds_flags[v] = ds_flags[v].round().astype(np.int8)
        ds_hourly = xr.merge([ds_data, ds_flags])
    else:
        ds_hourly = ds_data

    # Preserve dataset-level attributes
    ds_hourly.attrs = ds.attrs

    return ds_hourly


# ── 9. Example usage ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Option A: load directly from NetCDF ──────────────────────────────────
    # ds = xr.open_dataset("example.nc")

    # ── Option B: build a test Dataset from CSV (salinity only) ──────────────
    import numpy as np

    df = pd.read_csv("example.csv", parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Synthesise a co-located temperature record so we can demonstrate
    # the multi-variable workflow even without a real temperature file.
    # Replace this block with your actual temperature variable.
    np.random.seed(42)
    t_dec  = df["time"].apply(
        lambda dt: dt.year + (dt - pd.Timestamp(dt.year, 1, 1)).total_seconds()
                   / (pd.Timestamp(dt.year + 1, 1, 1) - pd.Timestamp(dt.year, 1, 1)).total_seconds()
    ).values
    T_synthetic = (
        10.0
        - 0.05  * (t_dec - t_dec[0])                      # slow cooling trend
        + 0.08  * np.maximum(0, t_dec - 2021.0)           # inflection in 2021
        + 4.5   * np.sin(2 * np.pi * t_dec)               # strong annual cycle
        + 0.8   * np.sin(4 * np.pi * t_dec)               # semi-annual
        + np.random.normal(0, 0.05, len(t_dec))           # noise
    )
    # Mirror the same NaN / gap structure as salinity
    T_synthetic[df["sea_water_practical_salinity"].isna()] = np.nan

    ds = xr.Dataset(
        {
            "sea_water_practical_salinity": (
                "time", df["sea_water_practical_salinity"].values
            ),
            "sea_water_temperature": (
                "time", T_synthetic
            ),
        },
        coords={"time": df["time"].values},
    )

    # ── Run gap-filling on both variables ────────────────────────────────────
    print("=" * 60)
    print("Ocean Time-Series Gap-Filling")
    print("=" * 60)

    ds_filled, results = fill_dataset_gaps(
        ds        = ds,
        variables = [
            "sea_water_practical_salinity",
            "sea_water_temperature",
        ],
        verbose   = True,
    )

    # ── Save output ──────────────────────────────────────────────────────────
    ds_filled.to_netcdf("ocean_gap_filled.nc")
    print(f"Filled dataset saved → ocean_gap_filled.nc")
    print(f"  Original points : {len(ds['time']):,}")
    print(f"  Output points   : {len(ds_filled['time']):,}")

    # ── Diagnostic figures ───────────────────────────────────────────────────
    plot_all_diagnostics(
        ds_filled = ds_filled,
        results   = results,
        save_dir  = "diagnostics",
    )

    # ── Interpolate to the top of each hour ──────────────────────────────────
    print("=" * 60)
    print("Interpolating to hourly grid")
    print("=" * 60)

    ds_hourly = interpolate_to_hourly(ds_filled, method="linear")

    time_hourly = pd.DatetimeIndex(ds_hourly["time"].values)
    print(f"  Hourly grid     : {time_hourly[0]}  →  {time_hourly[-1]}")
    print(f"  Total hours     : {len(time_hourly):,}")

    # Spot-check: all timestamps should be on exact hours
    assert (time_hourly.minute == 0).all() and (time_hourly.second == 0).all(), \
        "Non-hourly timestamps found!"
    print("  Timestamp check : all on the hour ✓")

    ds_hourly.to_netcdf("ocean_hourly.nc")
    print(f"  Saved           → ocean_hourly.nc")
