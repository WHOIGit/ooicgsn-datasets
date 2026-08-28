"""
nitrate.qc
==========
Instrument-level quality assessment for SUNA/NUTNR nitrate data, using a
subset of the QARTOD flag conventions.

Functions
---------
quality_checks         -- per-record QARTOD-style flag (RMSE, absorbance,
                           dark value, spectrum average, range checks)
add_not_evaluated_flags -- mark NaN-valued records with QARTOD flag 9
                           ("missing/not evaluated") on a paired *_qc_flag
                           variable
"""

from __future__ import annotations

import numpy as np


def quality_checks(ds, param):
    """
    Quality assessment of the raw and calculated nitrate concentration data
    using a susbset of the QARTOD flags to indicate the quality. QARTOD
    flags used are:

        1 = Pass
        3 = Suspect or of High Interest
        4 = Fail

    The final flag value represents the worst case assessment of the data quality.

    :param ds: xarray dataset with the raw signal data and the calculated
               seawater pH
    :param param: the name of the nitrate variable to check the range of
    :return qc_flag: array of flag values indicating seawater pH quality
    """
    qc_flag = ds['time'].astype('int32') * 0 + 1   # default flag values, no errors

    # "RMSE: The root-mean-square error parameter from the SUNA V2 can be used to make
    # an estimate of how well the nitrate spectral fit is. This should usually be less than 1E-3. If
    # it is higher, there is spectral shape (likely due to CDOM) that adversely impacts the nitrate
    # estimate." SUNA V2 vendor documentation (Sea-Bird Scientific Document# SUNA180725)
    m = ds.fit_rmse > 0.001  # per the vendor documentation
    qc_flag[m] = 3
    m = ds.fit_rmse > 0.100  # based on experience with the instrument data sets
    qc_flag[m] = 4

    # "Absorption: The data output of the SUNA V2 is the absorption at 350 nm and 254 nm
    # (A350 and A254). These wavelengths are outside the nitrate absorption range and can be
    # used to make an estimate of the impact of CDOM. If absorption is high (>1.3 AU), the
    # SUNA will not be able to collect adequate light to make a measurement." SUNA V2 vendor
    # documentation (Sea-Bird Scientific Document# SUNA180725)
    m254 = ds.absorbance_at_254_nm > 1.3
    qc_flag[m254] = 4
    m350 = ds.absorbance_at_350_nm > 1.3
    qc_flag[m350] = 4

    # test for failed dark value measurements (can't be less than 0)
    m = ds.dark_value_used_for_fit <= 0
    qc_flag[m] = 4

    # test for a blocked absorption channel (or a failed lamp)
    m = ds.spectrum_average < 10000
    qc_flag[m] = 4

    # test for out of range corrected dissolved nitrate readings
    m = (ds[param].values < -2.0) | (ds[param].values > 3000)
    qc_flag[m] = 4

    return qc_flag


def add_not_evaluated_flags(ds, var):
    """
    Mark records where ``var`` is NaN with QARTOD flag 9 ("missing / not
    evaluated") on the paired ``<var>_qc_flag`` variable. Intended to be
    called during the finalize stage, after drift/bottle correction may
    have introduced NaNs (e.g. deployments with no bottle match).
    """
    flag_var = '_'.join((var, 'qc_flag'))
    nan_mask = np.isnan(ds[var].values)
    flags = ds[flag_var].values.copy()
    flags[nan_mask] = 9
    attrs = ds[flag_var].attrs.copy()
    ds[flag_var] = (ds[flag_var].dims, flags.astype(ds[flag_var].dtype))
    ds[flag_var].attrs = attrs
    return ds
