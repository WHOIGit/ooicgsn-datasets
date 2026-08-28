"""
nitrate.resample
=================
Burst-average SUNA/NUTNR data to a fixed 15-minute interval using a median
average (plus the median absolute deviation of the corrected nitrate
concentration).

Functions
---------
mad                -- Calculates median absolute standard deviation
burst_resample     -- Resample nitrate dataset to define time interval using
                      a median average. Same as previous version, but vectorized
                      to speed up processing.
met_burst_resample -- Burst resamples co-located METCT data to the nitrate
                      dataset. Only applicable to Global Surface Buoys
"""

from __future__ import annotations

import pandas as pd
import xarray as xr
from scipy.stats import median_abs_deviation


def mad(array):
    """Calculate the median absolute standard deviation"""
    return median_abs_deviation(array, axis=0, nan_policy='omit')


def burst_resample(ds):
    """Resample the data to a defined time interval using a median average.

    See module docstring for the rationale behind this vectorized
    implementation.
    """
    ds = ds.load()

    spectral_var = 'raw_spectral_measurements'
    wl_dim = ds[spectral_var].dims[-1]
    scalar_vars = [v for v in ds.data_vars if ds[v].dims == ('time',)]

    # equivalent to the original resample(time='900s', base=3150, loffset='450s')
    resample_kwargs = dict(origin='start_day', offset=pd.Timedelta(seconds=3150))
    loffset = pd.Timedelta(seconds=450)

    # ---- All "time"-only (scalar) variables: a single grouped-median pass
    # across every one of them at once, rather than xarray looping over each
    # variable (and re-deriving the bin edges) separately.
    scalar_df = ds[scalar_vars].to_dataframe()
    scalar_med = scalar_df.resample('900s', **resample_kwargs).median()

    # ---- Median absolute deviation of corrected_nitrate_concentration, via
    # two vectorized passes (broadcast the group median back to every row,
    # then take a second grouped median of the absolute deviations) instead
    # of a per-bin scipy callback, which is what made the original ~10x
    # slower than it needed to be.
    cnc = scalar_df['corrected_nitrate_concentration']
    cnc_group_median = cnc.resample('900s', **resample_kwargs).transform('median')
    cnc_mad = (cnc - cnc_group_median).abs().resample('900s', **resample_kwargs).median()
    del scalar_df, cnc, cnc_group_median

    # ---- The 2-D spectral variable is resampled on its own (rather than
    # folded into one giant wide DataFrame with the scalar variables), since
    # it's the bulk of the data volume -- this keeps peak memory down while
    # still getting a single vectorized grouped-median pass across all 256
    # wavelength channels at once.
    spec_df = pd.DataFrame(ds[spectral_var].values, index=ds['time'].to_index())
    spec_med = spec_df.resample('900s', **resample_kwargs).median()
    del spec_df

    # shift bin labels to match the original loffset-based center-of-burst labeling
    scalar_med.index = scalar_med.index + loffset
    cnc_mad.index = cnc_mad.index + loffset
    spec_med.index = spec_med.index + loffset

    # drop empty bins (matches the original's `.where(~np.isnan(burst.deployment), drop=True)`)
    keep = ~scalar_med['deployment'].isna()
    scalar_med = scalar_med[keep]
    cnc_mad = cnc_mad.reindex(scalar_med.index)
    spec_med = spec_med.reindex(scalar_med.index)

    # rebuild the xarray Dataset, restoring the original per-variable attrs
    burst = xr.Dataset(coords={'time': scalar_med.index, wl_dim: ds[wl_dim].values})
    for v in scalar_vars:
        burst[v] = ('time', scalar_med[v].values)
        burst[v].attrs = ds[v].attrs
    burst[spectral_var] = (('time', wl_dim), spec_med.values)
    burst[spectral_var].attrs = ds[spectral_var].attrs
    burst['corrected_nitrate_concentration_mad'] = ('time', cnc_mad.values)
    burst['corrected_nitrate_concentration_mad'].attrs = {
        'comment': 'The median absolute standard deviation.'}
    burst.attrs = ds.attrs

    # and reset some data types
    data_types = ['deployment', 'spectrum_average', 'serial_number', 'dark_value_used_for_fit',
                  'raw_spectral_measurements']
    for v in data_types:
        burst[v] = burst[v].astype('int32')

    return burst


def met_burst_resample(ds):
    """Burst-average co-located METBK (met/CT) data to the same 15-minute
    interval as ``burst_resample`` -- used for SUMO buoy NUTNR deployments
    that derive sea-surface temperature/salinity from the METBK instead of
    a co-located CTD.
    """
    ds = ds.load()

    scalar_vars = [v for v in ds.data_vars if ds[v].dims == ('time',)]

    # equivalent to the original resample(time='900s', base=3150, loffset='450s')
    resample_kwargs = dict(origin='start_day', offset=pd.Timedelta(seconds=3150))
    loffset = pd.Timedelta(seconds=450)

    # ---- All "time"-only (scalar) variables: a single grouped-median pass
    # across every one of them at once, rather than xarray looping over each
    # variable (and re-deriving the bin edges) separately.
    scalar_df = ds[scalar_vars].to_dataframe()
    scalar_med = scalar_df.resample('900s', **resample_kwargs).median()

    # shift bin labels to match the original loffset-based center-of-burst labeling
    scalar_med.index = scalar_med.index + loffset

    # drop empty bins (matches the original's `.where(~np.isnan(burst.deployment), drop=True)`)
    keep = ~scalar_med['deployment'].isna()
    scalar_med = scalar_med[keep]

    # rebuild the xarray Dataset, restoring the original per-variable attrs
    burst = xr.Dataset(coords={'time': scalar_med.index})
    for v in scalar_vars:
        burst[v] = ('time', scalar_med[v].values)
        burst[v].attrs = ds[v].attrs
    burst.attrs = ds.attrs

    return burst
