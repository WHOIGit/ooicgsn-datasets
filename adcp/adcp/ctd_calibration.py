"""
CTD In-Situ Calibration, Drift Correction & Validation
=======================================================
Compares moored in-situ CTD data (salinity, temperature) against
co-located discrete water samples collected during research cruises
at deployment and recovery/turnaround events.

Workflow
--------
1. Load in-situ CTD and water-sampling datasets
2. Identify all SUMO water samples (fuzzy "SUMO" match on Target Asset)
3. For each deployment, assign the best reference cast(s) to the
   deployment START and END events
4. Compare in-situ medians (±2 h window) to discrete bottle values at
   the mooring's nominal depth
5. Fit a linear drift model:  offset(t) = offset_start + slope × (t − t_start)
   - Deployments with only one reference: constant-offset correction
   - Deployments with two references:     drift-corrected
6. Apply the correction to produce calibrated variables
7. Flag each point:
      0 = observed (uncorrected), 1 = offset-only corrected, 2 = drift corrected
8. Emit a comprehensive validation report and diagnostic figures

Author:  auto-generated
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── 1.  Configuration ─────────────────────────────────────────────────────────

# SUMO target asset pattern (case-insensitive)
SUMO_PATTERN = re.compile(r"SUMO", re.IGNORECASE)

# Nominal mooring pressure and tolerance window (dbar) used to pick the
# "mooring-depth" bottle from each CTD cast profile
MOORING_PRESSURE_NOMINAL = 500.0   # dbar
MOORING_PRESSURE_WINDOW  = 100.0   # ± dbar

# Time window around a cast used to extract the in-situ median
COMPARISON_WINDOW = pd.Timedelta("2h")

# Minimum samples required inside the window to accept a comparison
MIN_INSITU_POINTS = 5

# Column names in the in-situ file
INSITU_TIME_COL    = "time"
INSITU_SAL_COL     = "sea_water_practical_salinity"
INSITU_TEMP_COL    = "sea_water_temperature"
INSITU_PRES_COL    = "sea_water_pressure"
INSITU_DEP_COL     = "deployment"

# Column names in the water-sampling file
WS_TARGET_COL      = "Target Asset"
WS_CRUISE_COL      = "Cruise"
WS_BOTTLE_TIME_COL = "CTD Bottle Closure Time [UTC]"
WS_PRESSURE_COL    = "CTD Pressure [db]"
WS_SAL_REF_COL     = "Discrete Salinity [psu]"
WS_SAL_FLAG_COL    = "Discrete Salinity Flag"
WS_CTD_SAL1_COL    = "CTD Salinity 1 [psu]"   # fallback if no discrete
WS_TEMP_REF_COL    = "CTD Temperature 1 [deg C]"

# Acceptable discrete salinity flag values (1 = good in OOI/GO-SHIP convention)
GOOD_SAL_FLAGS = {1.0, 2.0}

# Correction quality-flag values written to output
FLAG_UNCORRECTED      = 0   # raw, no reference available
FLAG_OFFSET_ONLY      = 1   # constant offset applied (one-sided reference)
FLAG_DRIFT_CORRECTED  = 2   # linear drift + offset applied


# ── 2.  Data structures ───────────────────────────────────────────────────────

@dataclass
class WaterSample:
    """One row from the water-sampling dataset at or near mooring depth."""
    cruise:       str
    bottle_time:  pd.Timestamp
    pressure:     float
    ref_salinity: float         # discrete bottle value (preferred) or CTD
    ref_temp:     float         # CTD rosette temperature
    sal_source:   str           # "discrete" or "ctd"


@dataclass
class ComparisonPoint:
    """
    Result of comparing in-situ median to a single WaterSample.
    """
    role:         str           # "start" or "end"
    sample:       WaterSample
    insitu_sal:   float         # median over comparison window
    insitu_temp:  float
    delta_sal:    float         # insitu − reference  (positive → sensor reads high)
    delta_temp:   float
    n_points:     int           # in-situ points in window


@dataclass
class DeploymentCorrection:
    """
    Drift model for one deployment, derived from ≤2 ComparisonPoints.
    """
    deployment:      int
    t_start:         pd.Timestamp
    t_end:           pd.Timestamp
    comparisons:     list[ComparisonPoint]
    # Salinity model: offset_sal(t) = sal_offset_start + sal_slope × (t − t_start)
    sal_offset_start: float = 0.0
    sal_slope:        float = 0.0   # PSU/day
    sal_has_drift:    bool  = False
    # Temperature model
    temp_offset_start: float = 0.0
    temp_slope:        float = 0.0   # °C/day
    temp_has_drift:    bool  = False

    def sal_offset(self, t: pd.Series) -> pd.Series:
        """Salinity correction at times t (removes sensor bias from raw values)."""
        days = (t - self.t_start).dt.total_seconds() / 86400
        return -(self.sal_offset_start + self.sal_slope * days)

    def temp_offset(self, t: pd.Series) -> pd.Series:
        """Temperature correction at times t."""
        days = (t - self.t_start).dt.total_seconds() / 86400
        return -(self.temp_offset_start + self.temp_slope * days)

    def correction_flag(self) -> int:
        """Return the quality flag value to assign corrected points."""
        if self.sal_has_drift or self.temp_has_drift:
            return FLAG_DRIFT_CORRECTED
        return FLAG_OFFSET_ONLY

    def summary(self) -> str:
        lines = [
            f"Deployment {self.deployment}  "
            f"({self.t_start.date()} → {self.t_end.date()})",
            f"  Salinity:",
            f"    Initial offset : {self.sal_offset_start:+.5f} PSU "
            f"(insitu − reference; correction = {-self.sal_offset_start:+.5f})",
        ]
        if self.sal_has_drift:
            lines.append(
                f"    Drift rate     : {self.sal_slope*1000:+.4f} m-PSU/day "
                f"({self.sal_slope*1000*365:.3f} m-PSU/yr)"
            )
        else:
            lines.append("    Drift          : not estimated (single reference)")
        lines += [
            f"  Temperature:",
            f"    Initial offset : {self.temp_offset_start:+.5f} °C",
        ]
        if self.temp_has_drift:
            lines.append(
                f"    Drift rate     : {self.temp_slope*1000:+.4f} m°C/day"
            )
        else:
            lines.append("    Drift          : not estimated (single reference)")
        lines.append(f"  Correction flag : {self.correction_flag()}")
        return "\n".join(lines)


# ── 3.  Water-sample utilities ────────────────────────────────────────────────

def load_and_filter_sumo_samples(ws_path: str) -> pd.DataFrame:
    """
    Load the water-sampling CSV and return only rows where Target Asset
    contains 'SUMO' (case-insensitive).  Parse timestamps.
    """
    ws = pd.read_csv(ws_path)
    mask = ws[WS_TARGET_COL].astype(str).str.contains(SUMO_PATTERN.pattern,
                                                        case=False, na=False)
    ws_sumo = ws[mask].copy()
    ws_sumo["_bottle_time"] = pd.to_datetime(
        ws_sumo[WS_BOTTLE_TIME_COL], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    ws_sumo = ws_sumo.dropna(subset=["_bottle_time"])
    ws_sumo = ws_sumo.sort_values("_bottle_time").reset_index(drop=True)
    return ws_sumo


def _best_salinity(row: pd.Series) -> tuple[float, str]:
    """
    Return (salinity_value, source_string) for one water-sampling row.
    Prefer discrete bottle value with good flag; fall back to CTD rosette.
    """
    flag = row.get(WS_SAL_FLAG_COL, np.nan)
    disc = row.get(WS_SAL_REF_COL, np.nan)
    if pd.notna(disc) and (pd.isna(flag) or float(flag) in GOOD_SAL_FLAGS):
        return float(disc), "discrete"
    ctd_sal = row.get(WS_CTD_SAL1_COL, np.nan)
    if pd.notna(ctd_sal):
        return float(ctd_sal), "ctd"
    return np.nan, "missing"


def extract_mooring_depth_samples(ws_sumo: pd.DataFrame,
                                   nominal_p: float = MOORING_PRESSURE_NOMINAL,
                                   window_p:  float = MOORING_PRESSURE_WINDOW,
                                   ) -> list[WaterSample]:
    """
    From all SUMO water-sampling rows, retain only those bottles whose
    pressure is within [nominal_p − window_p, nominal_p + window_p] dbar.
    If a cast has multiple bottles in that window, select the one nearest
    nominal_p.  Returns a list of WaterSample objects sorted by time.
    """
    ws_near = ws_sumo[
        ws_sumo[WS_PRESSURE_COL].between(nominal_p - window_p,
                                          nominal_p + window_p)
    ].copy()

    # Pick the nearest-to-nominal bottle per (cruise, cast-date-hour)
    ws_near["_p_diff"] = (ws_near[WS_PRESSURE_COL] - nominal_p).abs()
    # Group by cruise + cast date
    ws_near["_cast_id"] = (
        ws_near[WS_CRUISE_COL].astype(str)
        + "_"
        + ws_near["_bottle_time"].dt.strftime("%Y%m%d")
    )

    samples: list[WaterSample] = []
    for _, grp in ws_near.groupby("_cast_id"):
        row = grp.sort_values("_p_diff").iloc[0]
        sal, source = _best_salinity(row)
        if np.isnan(sal):
            continue
        samples.append(WaterSample(
            cruise       = str(row[WS_CRUISE_COL]),
            bottle_time  = row["_bottle_time"],
            pressure     = float(row[WS_PRESSURE_COL]),
            ref_salinity = sal,
            ref_temp     = float(row.get(WS_TEMP_REF_COL, np.nan)),
            sal_source   = source,
        ))

    samples.sort(key=lambda s: s.bottle_time)
    return samples


# ── 4.  In-situ comparison ────────────────────────────────────────────────────

def compare_insitu_to_sample(
        insitu: pd.DataFrame,
        sample: WaterSample,
        window: pd.Timedelta = COMPARISON_WINDOW,
        min_pts: int = MIN_INSITU_POINTS,
        role: str = "start",
) -> Optional[ComparisonPoint]:
    """
    Extract in-situ data in [sample.bottle_time ± window] and compute
    median salinity & temperature.  Returns None if insufficient data.
    """
    t0 = sample.bottle_time - window
    t1 = sample.bottle_time + window
    sub = insitu[(insitu[INSITU_TIME_COL] >= t0) &
                 (insitu[INSITU_TIME_COL] <= t1)]

    # Relax window if too few points
    if len(sub) < min_pts:
        sub = insitu[(insitu[INSITU_TIME_COL] >= sample.bottle_time - 2 * window) &
                     (insitu[INSITU_TIME_COL] <= sample.bottle_time + 2 * window)]

    if len(sub) < 1:
        return None

    insitu_sal  = float(sub[INSITU_SAL_COL].median())
    insitu_temp = float(sub[INSITU_TEMP_COL].median())

    return ComparisonPoint(
        role        = role,
        sample      = sample,
        insitu_sal  = insitu_sal,
        insitu_temp = insitu_temp,
        delta_sal   = insitu_sal  - sample.ref_salinity,
        delta_temp  = insitu_temp - sample.ref_temp,
        n_points    = len(sub),
    )


# ── 5.  Deployment-level correction fitting ───────────────────────────────────

def _assign_samples_to_deployments(
        dep_bounds: dict[int, tuple[pd.Timestamp, pd.Timestamp]],
        samples:    list[WaterSample],
        max_start_delta: pd.Timedelta = pd.Timedelta("30D"),
        max_end_delta:   pd.Timedelta = pd.Timedelta("30D"),
) -> dict[int, dict[str, list[WaterSample]]]:
    """
    For each deployment, find water samples closest to the start (first data
    point) and end (last data point) of the deployment.

    A sample qualifies as a "start" reference if it falls within
    [dep_start − max_start_delta, dep_start + max_start_delta] AND is earlier
    than dep_end.  Similarly for "end" references.  Multiple qualifying samples
    are kept and later averaged.

    Returns {deployment_id: {"start": [WaterSample,...], "end": [WaterSample,...]}}.
    """
    assignment: dict[int, dict[str, list[WaterSample]]] = {}
    for dep, (t_start, t_end) in dep_bounds.items():
        start_samples = [
            s for s in samples
            if abs((s.bottle_time - t_start).total_seconds()) <=
               max_start_delta.total_seconds()
            and s.bottle_time <= t_end
        ]
        end_samples = [
            s for s in samples
            if abs((s.bottle_time - t_end).total_seconds()) <=
               max_end_delta.total_seconds()
            and s.bottle_time >= t_start
        ]
        assignment[dep] = {"start": start_samples, "end": end_samples}
    return assignment


def fit_deployment_correction(
        dep:           int,
        dep_data:      pd.DataFrame,
        start_comps:   list[ComparisonPoint],
        end_comps:     list[ComparisonPoint],
) -> DeploymentCorrection:
    """
    Fit a linear drift model (or constant offset if only one reference).

    offset(t) = insitu(t) − reference(t)
    Corrected value = raw − offset(t)  i.e.  raw + |offset| when sensor reads low
    """
    t_dep_start = dep_data[INSITU_TIME_COL].min()
    t_dep_end   = dep_data[INSITU_TIME_COL].max()

    dc = DeploymentCorrection(
        deployment = dep,
        t_start    = t_dep_start,
        t_end      = t_dep_end,
        comparisons = start_comps + end_comps,
    )

    def _mean_delta(comps: list[ComparisonPoint], var: str) -> Optional[float]:
        vals = [getattr(c, var) for c in comps if pd.notna(getattr(c, var))]
        return float(np.mean(vals)) if vals else None

    sal_start  = _mean_delta(start_comps, "delta_sal")
    sal_end    = _mean_delta(end_comps,   "delta_sal")
    temp_start = _mean_delta(start_comps, "delta_temp")
    temp_end   = _mean_delta(end_comps,   "delta_temp")

    # Reference times for slope calculation
    t_ref_start = np.mean([c.sample.bottle_time.timestamp()
                           for c in start_comps]) if start_comps else None
    t_ref_end   = np.mean([c.sample.bottle_time.timestamp()
                           for c in end_comps])   if end_comps else None

    # ── Salinity ──────────────────────────────────────────────────────────────
    if sal_start is not None and sal_end is not None and t_ref_start and t_ref_end:
        dt_days = (t_ref_end - t_ref_start) / 86400
        if abs(dt_days) > 1:
            dc.sal_offset_start = sal_start
            dc.sal_slope        = (sal_end - sal_start) / dt_days
            dc.sal_has_drift    = True
        else:
            dc.sal_offset_start = np.mean([sal_start, sal_end])
    elif sal_start is not None:
        dc.sal_offset_start = sal_start
    elif sal_end is not None:
        dc.sal_offset_start = sal_end

    # ── Temperature ───────────────────────────────────────────────────────────
    if temp_start is not None and temp_end is not None and t_ref_start and t_ref_end:
        dt_days = (t_ref_end - t_ref_start) / 86400
        if abs(dt_days) > 1:
            dc.temp_offset_start = temp_start
            dc.temp_slope        = (temp_end - temp_start) / dt_days
            dc.temp_has_drift    = True
        else:
            dc.temp_offset_start = np.mean([temp_start, temp_end])
    elif temp_start is not None:
        dc.temp_offset_start = temp_start
    elif temp_end is not None:
        dc.temp_offset_start = temp_end

    return dc


# ── 6.  Apply corrections ─────────────────────────────────────────────────────

def apply_corrections(
        insitu: pd.DataFrame,
        corrections: dict[int, DeploymentCorrection],
) -> pd.DataFrame:
    """
    Apply drift corrections per deployment.  Adds columns:
      - sea_water_practical_salinity_calibrated
      - sea_water_temperature_calibrated
      - calibration_flag  (0=uncorrected, 1=offset, 2=drift)
    """
    out = insitu.copy()
    out["sea_water_practical_salinity_calibrated"] = np.nan
    out["sea_water_temperature_calibrated"]         = np.nan
    out["calibration_flag"]                         = FLAG_UNCORRECTED

    for dep, dc in corrections.items():
        mask = out[INSITU_DEP_COL] == float(dep)
        t    = out.loc[mask, INSITU_TIME_COL]

        sal_corr  = out.loc[mask, INSITU_SAL_COL]  + dc.sal_offset(t)
        temp_corr = out.loc[mask, INSITU_TEMP_COL] + dc.temp_offset(t)

        out.loc[mask, "sea_water_practical_salinity_calibrated"] = sal_corr
        out.loc[mask, "sea_water_temperature_calibrated"]         = temp_corr
        out.loc[mask, "calibration_flag"]                         = dc.correction_flag()

    return out


# ── 7.  Validation statistics ─────────────────────────────────────────────────

def validation_stats(
        calibrated:  pd.DataFrame,
        corrections: dict[int, DeploymentCorrection],
) -> pd.DataFrame:
    """
    For each comparison point, compute residual = (calibrated in-situ) − reference.
    Returns a DataFrame summarising pre- and post-correction bias and RMSE.
    """
    rows = []
    for dep, dc in corrections.items():
        dep_data = calibrated[calibrated[INSITU_DEP_COL] == float(dep)]
        for cp in dc.comparisons:
            t0 = cp.sample.bottle_time - COMPARISON_WINDOW
            t1 = cp.sample.bottle_time + COMPARISON_WINDOW
            sub = dep_data[(dep_data[INSITU_TIME_COL] >= t0) &
                           (dep_data[INSITU_TIME_COL] <= t1)]
            if len(sub) == 0:
                continue
            cal_sal  = float(sub["sea_water_practical_salinity_calibrated"].median())
            cal_temp = float(sub["sea_water_temperature_calibrated"].median())
            rows.append({
                "deployment":      dep,
                "role":            cp.role,
                "cruise":          cp.sample.cruise,
                "time":            cp.sample.bottle_time,
                "pressure_db":     cp.sample.pressure,
                "ref_sal":         cp.sample.ref_salinity,
                "ref_temp":        cp.sample.ref_temp,
                "raw_sal":         cp.insitu_sal,
                "raw_temp":        cp.insitu_temp,
                "cal_sal":         cal_sal,
                "cal_temp":        cal_temp,
                "raw_delta_sal":   cp.delta_sal,
                "raw_delta_temp":  cp.delta_temp,
                "cal_delta_sal":   cal_sal  - cp.sample.ref_salinity,
                "cal_delta_temp":  cal_temp - cp.sample.ref_temp,
                "n_insitu_pts":    cp.n_points,
                "sal_source":      cp.sample.sal_source,
                "correction_type": "drift" if dc.sal_has_drift else "offset",
            })
    return pd.DataFrame(rows)


def print_validation_report(
        corrections: dict[int, DeploymentCorrection],
        stats:       pd.DataFrame,
) -> None:
    print("\n" + "=" * 70)
    print("CTD CALIBRATION & VALIDATION REPORT")
    print("=" * 70)

    for dep, dc in corrections.items():
        print(f"\n{'─'*70}")
        print(dc.summary())

    print(f"\n{'─'*70}")
    print("COMPARISON POINT VALIDATION\n")
    hdr = ("{dep:>4}  {role:>5}  {cruise:>10}  {time:>12}  "
           "{ref_sal:>8}  {raw_d:>8}  {cal_d:>8}  "
           "{ref_t:>7}  {raw_dt:>7}  {cal_dt:>7}  {flag}")
    fmt = hdr  # same format - values already pre-formatted as strings
    print(hdr.format(dep="Dep", role="Role", cruise="Cruise", time="Date",
                     ref_sal="RefSal", raw_d="RawΔSal", cal_d="CalΔSal",
                     ref_t="RefT", raw_dt="RawΔT", cal_dt="CalΔT",
                     flag="Correction"))
    print("─" * 100)
    for _, row in stats.iterrows():
        print(fmt.format(
            dep    = int(row["deployment"]),
            role   = row["role"],
            cruise = row["cruise"],
            time   = str(row["time"].date()),
            ref_sal= f"{row['ref_sal']:.4f}",
            raw_d  = f"{row['raw_delta_sal']:+.4f}",
            cal_d  = f"{row['cal_delta_sal']:+.4f}",
            ref_t  = f"{row['ref_temp']:.4f}",
            raw_dt = f"{row['raw_delta_temp']:+.4f}",
            cal_dt = f"{row['cal_delta_temp']:+.4f}",
            flag   = row["correction_type"],
        ))

    # Summary statistics
    print(f"\n{'─'*70}")
    print("SUMMARY STATISTICS")
    for dep in sorted(stats["deployment"].unique()):
        sub = stats[stats["deployment"] == dep]
        print(f"\n  Deployment {int(dep)} ({len(sub)} comparison points):")
        print(f"    Salinity  — raw  bias={sub['raw_delta_sal'].mean():+.5f} PSU  "
              f"RMSE={np.sqrt((sub['raw_delta_sal']**2).mean()):.5f}")
        print(f"    Salinity  — cal  bias={sub['cal_delta_sal'].mean():+.5f} PSU  "
              f"RMSE={np.sqrt((sub['cal_delta_sal']**2).mean()):.5f}")
        print(f"    Temp      — raw  bias={sub['raw_delta_temp'].mean():+.5f} °C  "
              f"RMSE={np.sqrt((sub['raw_delta_temp']**2).mean()):.5f}")
        print(f"    Temp      — cal  bias={sub['cal_delta_temp'].mean():+.5f} °C  "
              f"RMSE={np.sqrt((sub['cal_delta_temp']**2).mean()):.5f}")


# ── 8.  Diagnostic figures ────────────────────────────────────────────────────

def _dep_colormap(deployments: list) -> dict:
    """
    Return a {deployment_id: RGBA colour} dict that works for any number of
    deployments.

    Strategy
    --------
    ≤10 deps  → qualitative ``tab10``  (maximally distinct, familiar palette)
    11–20     → ``tab20``
    21+       → ``turbo`` (perceptually-uniform rainbow sampled uniformly)
    """
    n = len(deployments)
    if n == 0:
        return {}
    if n <= 10:
        cmap   = plt.get_cmap("tab10")
        colors = [cmap(i / 10) for i in range(n)]
    elif n <= 20:
        cmap   = plt.get_cmap("tab20")
        colors = [cmap(i / 20) for i in range(n)]
    else:
        cmap   = plt.get_cmap("turbo")
        colors = [cmap(i / (n - 1)) for i in range(n)]
    return {dep: colors[i] for i, dep in enumerate(sorted(deployments))}


def _outside_legend(fig: plt.Figure, ax: plt.Axes, handles, ncol_max: int = 6) -> None:
    """Place a compact legend below *ax*, spanning the full figure width."""
    ncol = min(len(handles), ncol_max)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=ncol,
        fontsize=7,
        frameon=True,
        borderaxespad=0,
    )


def _date_axis(ax: plt.Axes, n_years: float) -> None:
    """Apply appropriate date tick density based on total time span."""
    if n_years <= 3:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    elif n_years <= 10:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")


def _scatter_ax(ax, raw, cal, ref, label, unit, dep_colors, dep_ids):
    """
    In-situ vs reference scatter coloured by deployment.
    Circles = raw; triangles = calibrated.
    """
    raw_arr = np.asarray(raw)
    cal_arr = np.asarray(cal)
    ref_arr = np.asarray(ref)
    dep_arr = np.asarray(dep_ids)

    for dep in sorted(np.unique(dep_arr)):
        mask  = dep_arr == dep
        color = dep_colors.get(int(dep), "gray")
        ax.scatter(ref_arr[mask], raw_arr[mask], s=30, color=color, alpha=0.75, zorder=3, marker="o")
        ax.scatter(ref_arr[mask], cal_arr[mask], s=30, color=color, alpha=0.90, zorder=4, marker="^")

    lo = min(ref_arr.min(), raw_arr.min(), cal_arr.min()) - 0.01
    hi = max(ref_arr.max(), raw_arr.max(), cal_arr.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=2)

    raw_bias = float(np.mean(raw_arr - ref_arr))
    cal_bias = float(np.mean(cal_arr - ref_arr))
    ax.plot([], [], "o", color="gray", label=f"Raw   (bias={raw_bias:+.4f})")
    ax.plot([], [], "^", color="gray", label=f"Cal   (bias={cal_bias:+.4f})")
    ax.plot([], [], "k--", lw=0.8,    label="1:1 line")
    ax.legend(fontsize=7)
    ax.set_xlabel(f"Reference {label} ({unit})")
    ax.set_ylabel(f"In-situ {label} ({unit})")
    ax.set_title(f"{label}: In-situ vs Reference")


def plot_timeseries(
        calibrated:  pd.DataFrame,
        corrections: dict,
        stats:       pd.DataFrame,
        save_path:   Optional[str] = None,
) -> plt.Figure:
    """
    Three-row figure:
      Row 0 (full width) : salinity time series  (raw faint + calibrated solid)
      Row 1 (full width) : temperature time series
      Row 2 left/right   : salinity / temperature scatter vs reference

    Deployment colours are generated dynamically — works for any N.
    Legends are placed below each time-series panel (up to 6 columns).
    """
    deps       = sorted(corrections.keys())
    dep_colors = _dep_colormap(deps)

    t_all   = calibrated[INSITU_TIME_COL]
    n_years = (t_all.max() - t_all.min()).days / 365.25

    legend_rows = max(1, int(np.ceil(len(deps) / 6)))
    fig = plt.figure(figsize=(17, 14 + legend_rows * 0.5))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.55, wspace=0.30,
                            top=0.94, bottom=0.04)
    ax_sal       = fig.add_subplot(gs[0, 0:2])
    ax_temp      = fig.add_subplot(gs[1, 0:2])
    ax_scat_sal  = fig.add_subplot(gs[2, 0])
    ax_scat_temp = fig.add_subplot(gs[2, 1])

    sal_handles  = []
    temp_handles = []

    for dep in deps:
        color = dep_colors[dep]
        sub   = calibrated[calibrated[INSITU_DEP_COL] == float(dep)]
        if sub.empty:
            continue
        dc    = corrections[dep]
        ltype = "drift" if dc.sal_has_drift else "offset"

        ax_sal.plot(sub[INSITU_TIME_COL], sub[INSITU_SAL_COL],
                    color=color, lw=0.4, alpha=0.35)
        line_s, = ax_sal.plot(sub[INSITU_TIME_COL],
                               sub["sea_water_practical_salinity_calibrated"],
                               color=color, lw=0.8, label=f"Dep {dep} ({ltype})")
        sal_handles.append(line_s)

        ax_temp.plot(sub[INSITU_TIME_COL], sub[INSITU_TEMP_COL],
                     color=color, lw=0.4, alpha=0.35)
        line_t, = ax_temp.plot(sub[INSITU_TIME_COL],
                                sub["sea_water_temperature_calibrated"],
                                color=color, lw=0.8, label=f"Dep {dep} ({ltype})")
        temp_handles.append(line_t)

    # Reference cast markers
    for _, row in stats.iterrows():
        color  = dep_colors.get(int(row["deployment"]), "gray")
        marker = "v" if row["role"] == "start" else "^"
        ax_sal.scatter(row["time"], row["ref_sal"],
                       marker=marker, s=55, color=color, zorder=5, edgecolors="k", lw=0.5)
        ax_temp.scatter(row["time"], row["ref_temp"],
                        marker=marker, s=55, color=color, zorder=5, edgecolors="k", lw=0.5)

    ax_sal.set_ylabel("Salinity (PSU)")
    ax_sal.set_title(
        "Salinity  (faint = raw,  solid = calibrated,  ▼ start ref,  △ end ref)",
        fontsize=10)
    _date_axis(ax_sal, n_years)
    _outside_legend(fig, ax_sal, sal_handles, ncol_max=6)

    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.set_title("Temperature  (faint = raw,  solid = calibrated)", fontsize=10)
    _date_axis(ax_temp, n_years)
    _outside_legend(fig, ax_temp, temp_handles, ncol_max=6)

    if not stats.empty:
        dep_ids = stats["deployment"].values
        _scatter_ax(ax_scat_sal,
                    raw=stats["raw_sal"].values, cal=stats["cal_sal"].values,
                    ref=stats["ref_sal"].values,
                    label="Salinity", unit="PSU",
                    dep_colors=dep_colors, dep_ids=dep_ids)
        _scatter_ax(ax_scat_temp,
                    raw=stats["raw_temp"].values, cal=stats["cal_temp"].values,
                    ref=stats["ref_temp"].values,
                    label="Temperature", unit="°C",
                    dep_colors=dep_colors, dep_ids=dep_ids)

    fig.suptitle("CTD In-Situ Calibration & Drift Correction", fontsize=13)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig


def plot_residuals(
        stats:     pd.DataFrame,
        save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel residual figure (salinity top, temperature bottom).

    Each comparison point is plotted as a circle (raw) and triangle
    (calibrated), coloured by deployment.  Cruise labels annotate each
    raw point with a small vertical jitter when multiple casts share the
    same cruise.  Legend placed below each panel; works for any N.
    """
    if stats.empty:
        fig, _ = plt.subplots(2, 1, figsize=(13, 8))
        return fig

    deps       = sorted(stats["deployment"].unique().astype(int))
    dep_colors = _dep_colormap(deps)

    t_all   = stats["time"]
    n_years = (t_all.max() - t_all.min()).days / 365.25

    legend_rows = max(1, int(np.ceil(len(deps) / 6)))
    fig, (ax_sal, ax_temp) = plt.subplots(
        2, 1, figsize=(13, 10 + legend_rows * 0.6),
        gridspec_kw={"hspace": 0.55},
    )

    # Small jitter to separate annotations from the same cruise
    cruise_counter: dict = {}
    jitter_step = 0.00015

    for dep in deps:
        sub   = stats[stats["deployment"] == dep]
        color = dep_colors[dep]
        first_sal  = True
        first_temp = True

        for _, row in sub.iterrows():
            marker = "v" if row["role"] == "start" else "^"
            cruise = str(row["cruise"])
            idx    = cruise_counter.get(cruise, 0)
            cruise_counter[cruise] = idx + 1
            sign   = 1 if idx % 2 == 0 else -1
            jitter = jitter_step * (idx // 2) * sign

            ax_sal.scatter(row["time"], row["raw_delta_sal"] + jitter,
                           marker=marker, s=60, color="gray", alpha=0.65, zorder=3,
                           label="Raw (all deps)" if first_sal else "_")
            ax_sal.scatter(row["time"], row["cal_delta_sal"] + jitter,
                           marker=marker, s=60, color=color, zorder=4,
                           label=f"Dep {dep}" if first_sal else "_")
            ax_sal.annotate(cruise,
                            (row["time"], row["raw_delta_sal"] + jitter),
                            textcoords="offset points",
                            xytext=(3, 5), fontsize=5.5, color="dimgray",
                            rotation=40, ha="left")

            ax_temp.scatter(row["time"], row["raw_delta_temp"] + jitter,
                            marker=marker, s=60, color="gray", alpha=0.65, zorder=3,
                            label="Raw (all deps)" if first_temp else "_")
            ax_temp.scatter(row["time"], row["cal_delta_temp"] + jitter,
                            marker=marker, s=60, color=color, zorder=4,
                            label=f"Dep {dep}" if first_temp else "_")
            first_sal  = False
            first_temp = False

    for ax in (ax_sal, ax_temp):
        ax.axhline(0, color="k", lw=0.8, ls="--")
        _date_axis(ax, n_years)
        handles, labels = ax.get_legend_handles_labels()
        seen, dedup_h = set(), []
        for h, l in zip(handles, labels):
            if l not in seen and not l.startswith("_"):
                seen.add(l)
                dedup_h.append(h)
        _outside_legend(fig, ax, dedup_h, ncol_max=6)

    ax_sal.set_ylabel("Δ Salinity (PSU)  [in-situ − reference]")
    ax_sal.set_title(
        "Salinity residuals  (gray = raw,  coloured = calibrated,  "
        "▼ = start ref,  △ = end ref)", fontsize=10)
    ax_temp.set_ylabel("Δ Temperature (°C)")
    ax_temp.set_title("Temperature residuals", fontsize=10)

    fig.suptitle("Calibration Residuals — Before (gray) and After (colour)", fontsize=12)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig


def plot_drift_models(
        calibrated:  pd.DataFrame,
        corrections: dict,
        save_path:   Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel figure showing the applied correction (offset or drift line)
    for every deployment, with reference cast markers overlaid.

    Legend placed below each panel; tick density adapts to record length.
    Works for any number of deployments.
    """
    deps       = sorted(corrections.keys())
    dep_colors = _dep_colormap(deps)

    t_all   = calibrated[INSITU_TIME_COL]
    n_years = (t_all.max() - t_all.min()).days / 365.25

    legend_rows = max(1, int(np.ceil(len(deps) / 6)))
    fig, (ax_sal, ax_temp) = plt.subplots(
        2, 1, figsize=(14, 9 + legend_rows * 0.6),
        gridspec_kw={"hspace": 0.55},
    )

    sal_handles  = []
    temp_handles = []

    for dep in deps:
        dc    = corrections[dep]
        sub   = calibrated[calibrated[INSITU_DEP_COL] == float(dep)]
        if sub.empty:
            continue
        color = dep_colors[dep]
        t     = sub[INSITU_TIME_COL]
        ltype = "drift" if dc.sal_has_drift else "offset"

        line_s, = ax_sal.plot(t,  -dc.sal_offset(t),
                               color=color, lw=1.4, label=f"Dep {dep} ({ltype})")
        line_t, = ax_temp.plot(t, -dc.temp_offset(t),
                                color=color, lw=1.4, label=f"Dep {dep} ({ltype})")
        sal_handles.append(line_s)
        temp_handles.append(line_t)

        for cp in dc.comparisons:
            mk = "v" if cp.role == "start" else "^"
            ax_sal.scatter(cp.sample.bottle_time,  cp.delta_sal,
                           marker=mk, s=70, color=color, edgecolors="k", lw=0.5, zorder=5)
            ax_temp.scatter(cp.sample.bottle_time, cp.delta_temp,
                            marker=mk, s=70, color=color, edgecolors="k", lw=0.5, zorder=5)

    for ax in (ax_sal, ax_temp):
        ax.axhline(0, color="k", lw=0.8, ls="--")
        _date_axis(ax, n_years)

    ax_sal.set_ylabel("Salinity correction (PSU)")
    ax_sal.set_title(
        "Applied salinity correction  (▼ = start ref,  △ = end ref)", fontsize=10)
    _outside_legend(fig, ax_sal, sal_handles, ncol_max=6)

    ax_temp.set_ylabel("Temperature correction (°C)")
    ax_temp.set_title("Applied temperature correction", fontsize=10)
    _outside_legend(fig, ax_temp, temp_handles, ncol_max=6)

    fig.suptitle("Drift / Offset Correction Models by Deployment", fontsize=12)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig

# ── 9.  Main entry point ──────────────────────────────────────────────────────

def calibrate_insitu_ctd(
        insitu_path:    str,
        water_samp_path: str,
        output_csv:     str = "ctd_calibrated.csv",
        output_dir:     str = ".",
        mooring_pressure:      float = MOORING_PRESSURE_NOMINAL,
        mooring_pressure_window: float = MOORING_PRESSURE_WINDOW,
        verbose:        bool = True,
) -> tuple[pd.DataFrame, dict[int, DeploymentCorrection], pd.DataFrame]:
    """
    Full pipeline: load → match samples → fit corrections → apply → validate.

    Returns
    -------
    calibrated   : pd.DataFrame with raw + calibrated columns + flag
    corrections  : dict {deployment_id: DeploymentCorrection}
    stats        : pd.DataFrame with per-comparison-point validation stats
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    if verbose: print("Loading in-situ CTD data …")
    insitu = pd.read_csv(insitu_path, parse_dates=[INSITU_TIME_COL])
    insitu = insitu.sort_values(INSITU_TIME_COL).reset_index(drop=True)

    if verbose: print("Loading water-sampling data …")
    ws_raw  = pd.read_csv(water_samp_path)
    ws_sumo = load_and_filter_sumo_samples(water_samp_path)
    if verbose:
        print(f"  SUMO water samples found: {len(ws_sumo)}")

    # ── Extract mooring-depth bottles ─────────────────────────────────────────
    samples = extract_mooring_depth_samples(
        ws_sumo,
        nominal_p = mooring_pressure,
        window_p  = mooring_pressure_window,
    )
    if verbose:
        print(f"  Mooring-depth samples (±{mooring_pressure_window} db "
              f"around {mooring_pressure} db): {len(samples)}")
        for s in samples:
            print(f"    {s.cruise:10s}  {str(s.bottle_time.date()):12s}  "
                  f"p={s.pressure:.0f} db  sal={s.ref_salinity:.4f} PSU ({s.sal_source})")

    # ── Deployment boundaries ─────────────────────────────────────────────────
    dep_bounds: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for dep, grp in insitu.groupby(INSITU_DEP_COL):
        dep_bounds[int(dep)] = (grp[INSITU_TIME_COL].min(),
                                grp[INSITU_TIME_COL].max())

    # ── Assign samples → deployments ──────────────────────────────────────────
    assigned = _assign_samples_to_deployments(dep_bounds, samples)

    # ── Build comparison points ───────────────────────────────────────────────
    corrections: dict[int, DeploymentCorrection] = {}
    for dep, (t_start, t_end) in dep_bounds.items():
        dep_data = insitu[insitu[INSITU_DEP_COL] == float(dep)].copy()
        start_comps, end_comps = [], []
        for s in assigned[dep]["start"]:
            cp = compare_insitu_to_sample(dep_data, s, role="start")
            if cp: start_comps.append(cp)
        for s in assigned[dep]["end"]:
            cp = compare_insitu_to_sample(dep_data, s, role="end")
            if cp: end_comps.append(cp)

        if verbose:
            n_s, n_e = len(start_comps), len(end_comps)
            corr_type = "drift" if (n_s and n_e) else "offset" if (n_s or n_e) else "none"
            print(f"  Dep {dep}: {n_s} start ref(s), {n_e} end ref(s) → {corr_type}")

        dc = fit_deployment_correction(dep, dep_data, start_comps, end_comps)
        corrections[dep] = dc

    # ── Apply ─────────────────────────────────────────────────────────────────
    calibrated = apply_corrections(insitu, corrections)

    # ── Validate ──────────────────────────────────────────────────────────────
    stats = validation_stats(calibrated, corrections)
    if verbose:
        print_validation_report(corrections, stats)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    import os
    out_path = os.path.join(output_dir, output_csv)
    calibrated.to_csv(out_path, index=False)
    if verbose: print(f"\nCalibrated data saved → {out_path}")

    # ── Save validation table ─────────────────────────────────────────────────
    val_path = os.path.join(output_dir, "calibration_validation.csv")
    stats.to_csv(val_path, index=False)
    if verbose: print(f"Validation table saved → {val_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_timeseries(calibrated, corrections, stats,
                    save_path=os.path.join(output_dir, "ctd_calibration_timeseries.png"))
    plot_residuals(stats,
                   save_path=os.path.join(output_dir, "ctd_calibration_residuals.png"))
    plot_drift_models(calibrated, corrections,
                      save_path=os.path.join(output_dir, "ctd_drift_models.png"))

    return calibrated, corrections, stats


# ── 10.  xarray entry point ───────────────────────────────────────────────────

def calibrate_insitu_ctd_ds(
        ds:                      "xr.Dataset",
        water_samp_path:         str,
        output_dir:              str   = ".",
        time_dim:                str   = "time",
        sal_var:                 str   = INSITU_SAL_COL,
        temp_var:                str   = INSITU_TEMP_COL,
        pres_var:                str   = INSITU_PRES_COL,
        deployment_var:          str   = INSITU_DEP_COL,
        mooring_pressure:        float = MOORING_PRESSURE_NOMINAL,
        mooring_pressure_window: float = MOORING_PRESSURE_WINDOW,
        comparison_window:       str   = "2h",          # ← new
        verbose:                 bool  = True,
) -> tuple["xr.Dataset", dict[int, DeploymentCorrection], pd.DataFrame]:
    """
    xarray-native entry point.  Accepts the Dataset produced by
    fill_dataset_gaps / interpolate_to_hourly and returns a new Dataset
    with three additional variables:

        sea_water_practical_salinity_calibrated  – drift-corrected salinity
        sea_water_temperature_calibrated         – drift-corrected temperature
        calibration_flag                         – int8 per-point quality flag
                                                   0 = uncorrected (no reference)
                                                   1 = constant offset applied
                                                   2 = linear drift corrected

    The Dataset must contain a ``deployment`` variable (or the name supplied
    via ``deployment_var``) aligned on the time dimension.  All other
    processing is identical to the CSV-based pipeline.

    Parameters
    ----------
    ds                      : xarray Dataset with in-situ CTD data
    water_samp_path         : path to the water-sampling CSV
    output_dir              : directory for diagnostic figures and validation CSV
    time_dim                : name of the time coordinate in ds
    sal_var                 : salinity variable name in ds
    temp_var                : temperature variable name in ds
    pres_var                : pressure variable name in ds
    deployment_var          : deployment-ID variable name in ds
    mooring_pressure        : nominal mooring pressure (dbar) for bottle matching
    mooring_pressure_window : ± tolerance around nominal pressure (dbar)
    verbose                 : print progress and validation report

    Returns
    -------
    ds_calibrated  : copy of ds with calibrated variables added
    corrections    : dict {deployment_id: DeploymentCorrection}
    stats          : per-comparison-point validation DataFrame
    """
    import os
    import xarray as xr
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Extract working DataFrame from Dataset ─────────────────────────────
    if verbose:
        print("Extracting in-situ data from xarray Dataset …")

    time_vals = pd.DatetimeIndex(ds[time_dim].values)

    insitu = pd.DataFrame({
        INSITU_TIME_COL: time_vals,
        INSITU_SAL_COL:  ds[sal_var].values,
        INSITU_TEMP_COL: ds[temp_var].values,
        INSITU_PRES_COL: ds[pres_var].values if pres_var in ds else np.nan,
        INSITU_DEP_COL:  ds[deployment_var].values.astype(float),
    }).sort_values(INSITU_TIME_COL).reset_index(drop=True)

    if verbose:
        deps = sorted(insitu[INSITU_DEP_COL].dropna().unique())
        print(f"  {len(insitu):,} time steps  |  deployments: {[int(d) for d in deps]}")

    # ── 2. Run the shared calibration pipeline ────────────────────────────────
    if verbose: print("Loading water-sampling data …")
    ws_sumo = load_and_filter_sumo_samples(water_samp_path)
    if verbose:
        print(f"  SUMO water samples found: {len(ws_sumo)}")

    samples = extract_mooring_depth_samples(
        ws_sumo,
        nominal_p = mooring_pressure,
        window_p  = mooring_pressure_window,
    )
    if verbose:
        print(f"  Mooring-depth samples (±{mooring_pressure_window} db "
              f"around {mooring_pressure} db): {len(samples)}")
        for s in samples:
            print(f"    {s.cruise:10s}  {str(s.bottle_time.date()):12s}  "
                  f"p={s.pressure:.0f} db  sal={s.ref_salinity:.4f} PSU ({s.sal_source})")

    dep_bounds: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for dep, grp in insitu.groupby(INSITU_DEP_COL):
        dep_bounds[int(dep)] = (grp[INSITU_TIME_COL].min(),
                                grp[INSITU_TIME_COL].max())

    assigned = _assign_samples_to_deployments(dep_bounds, samples)

    corrections: dict[int, DeploymentCorrection] = {}
    
    # Convert string to Timedelta so callers can pass e.g. "4h" or "30min"
    _window = pd.Timedelta(comparison_window)

    for dep, (t_start, t_end) in dep_bounds.items():
        dep_data = insitu[insitu[INSITU_DEP_COL] == float(dep)].copy()
        start_comps, end_comps = [], []
        for s in assigned[dep]["start"]:
            cp = compare_insitu_to_sample(dep_data, s, window=_window, role="start")
            if cp: start_comps.append(cp)
        for s in assigned[dep]["end"]:
            cp = compare_insitu_to_sample(dep_data, s, window=_window, role="end")
            if cp: end_comps.append(cp)

        if verbose:
            n_s, n_e = len(start_comps), len(end_comps)
            corr_type = "drift" if (n_s and n_e) else "offset" if (n_s or n_e) else "none"
            print(f"  Dep {dep}: {n_s} start ref(s), {n_e} end ref(s) → {corr_type}")

        corrections[dep] = fit_deployment_correction(dep, dep_data,
                                                      start_comps, end_comps)

    calibrated = apply_corrections(insitu, corrections)
    stats      = validation_stats(calibrated, corrections)

    if verbose:
        print_validation_report(corrections, stats)

    # ── 3. Write calibrated variables back into the Dataset ───────────────────
    # Align on the Dataset's time coordinate (handles any ordering differences)
    cal_indexed = calibrated.set_index(INSITU_TIME_COL)
    sal_cal  = cal_indexed["sea_water_practical_salinity_calibrated"].reindex(time_vals).values
    temp_cal = cal_indexed["sea_water_temperature_calibrated"].reindex(time_vals).values
    flags    = cal_indexed["calibration_flag"].reindex(time_vals).fillna(0).values.astype(np.int8)

    ds_out = ds.copy()
    ds_out["sea_water_practical_salinity_calibrated"] = xr.DataArray(
        sal_cal, dims=[time_dim],
        attrs={
            "long_name":     "Calibrated sea water practical salinity",
            "units":         "1",
            "comment":       "Offset/drift-corrected against discrete water-sample bottles",
            "flag_values":   "0 1 2",
            "flag_meanings": "uncorrected offset_only drift_corrected",
        },
    )
    ds_out["sea_water_temperature_calibrated"] = xr.DataArray(
        temp_cal, dims=[time_dim],
        attrs={
            "long_name":     "Calibrated sea water temperature",
            "units":         "degree_Celsius",
            "comment":       "Offset/drift-corrected against CTD rosette temperature",
            "flag_values":   "0 1 2",
            "flag_meanings": "uncorrected offset_only drift_corrected",
        },
    )
    ds_out["calibration_flag"] = xr.DataArray(
        flags, dims=[time_dim],
        attrs={
            "long_name":     "Calibration correction type",
            "flag_values":   "0 1 2",
            "flag_meanings": "uncorrected offset_only drift_corrected",
        },
    )

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    val_path = os.path.join(output_dir, "calibration_validation.csv")
    stats.to_csv(val_path, index=False)
    if verbose: print(f"\nValidation table saved → {val_path}")

    plot_timeseries(calibrated, corrections, stats,
                    save_path=os.path.join(output_dir, "ctd_calibration_timeseries.png"))
    plot_residuals(stats,
                   save_path=os.path.join(output_dir, "ctd_calibration_residuals.png"))
    plot_drift_models(calibrated, corrections,
                      save_path=os.path.join(output_dir, "ctd_drift_models.png"))

    return ds_out, corrections, stats


# ── 11.  CLI / script entry ───────────────────────────────────────────────────

if __name__ == "__main__":
    import xarray as xr

    # ── CSV-based usage (original) ────────────────────────────────────────────
    calibrated, corrections, stats = calibrate_insitu_ctd(
        insitu_path      = "example_insitu_ctd_data.csv",
        water_samp_path  = "water_sampling.csv",
        output_csv       = "ctd_calibrated.csv",
        output_dir       = ".",
        verbose          = True,
    )

    # ── xarray-based usage ────────────────────────────────────────────────────
    # ds = xr.open_dataset("ocean_gap_filled.nc")
    # ds_cal, corrections, stats = calibrate_insitu_ctd_ds(
    #     ds               = ds,
    #     water_samp_path  = "water_sampling.csv",
    #     output_dir       = "diagnostics",
    # )
    # ds_cal.to_netcdf("ocean_calibrated.nc")
