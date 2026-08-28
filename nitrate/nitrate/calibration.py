"""
nitrate.calibration
=====================
Deployment calibration coefficient lookup, and the Plant et al. (2023)
recalculation / drift correction of ``corrected_nitrate_concentration``.

Functions
---------
add_plant_correction    -- recalculate corrected_nitrate_concentration for
                           a single deployment using the pre-deployment
                           calibration (no drift correction)
plant_drift_correction  -- additionally correct for calibration drift
                           across the deployment, using the post-cruise
                           calibration
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Needed to look up the deployment-specific calibration coefficients 
from ooi_data_explorations.common import get_deployment_dates, get_calibrations_by_refdes, get_calibrations_by_uid

from .ts_correction import plant2023_tsp_correction
from .qc import quality_checks


def _get_pre_post_calibrations(ds, site, node, sensor):
    """
    Identify the pre- and post-deployment calibration coefficients for a
    given SUNA dataset. This mirrors the calibration lookup logic inside
    process_nutnr.drift_correction(), reproduced here (rather than imported)
    so process_nutnr.py itself doesn't need to be modified. Shared by
    add_plant_correction() and plant_drift_correction() below.

    :param ds: A SUNA dataset reprocessed using the suna_datalogger
               or suna_instrument functions
    :param site: The site of the associated SUNA dataset
    :param node: The node of the associated SUNA dataset
    :param sensor: The sensor of the associated SUNA dataset
    :return: (pre_deploy_cal, post_deploy_cal) pandas DataFrames of calibration
             coefficients, respectively for the calibration in effect during
             the deployment and for the next calibration performed afterwards
             (e.g., the post-cruise calibration).
    """
    # grab the deployment number of the dataset
    depNum = int(np.unique(ds["deployment"])[0])

    # get the deployment start and end times
    deployStart, deployEnd = get_deployment_dates(site, node, sensor, depNum)

    # use the deployment start and end times to get the deployment specific calibrations
    calInfo = get_calibrations_by_refdes(site, node, sensor, deployStart, deployEnd, to_dataframe=True)
    pre_deploy_cal = calInfo[calInfo["deploymentNumber"] == depNum]

    # get the UID of the instrument
    uid = pre_deploy_cal["uid"].unique()[0]

    # now get all of the deployments for the given instrument
    calInfo = get_calibrations_by_uid(uid, to_dataframe=True)

    # find the next calibration AFTER the one used for the deployment
    pre_date = pre_deploy_cal["calDate"].unique()[0]
    delta_t = calInfo["calDate"].apply(lambda x: x - pre_date)
    delta_t = delta_t[delta_t.dt.days > 0]
    delta_t = delta_t[delta_t == np.min(delta_t)]

    # use the delta_t indices to get the post-cruise calibration
    post_deploy_cal = calInfo.loc[delta_t.index]

    return pre_deploy_cal, post_deploy_cal


def _cal_coefficient(cal_df, cal_coef, dtype='float64'):
    """
    Small helper to pull a single named calibration coefficient array out
    of a calibration DataFrame as returned by _get_pre_post_calibrations().
    """
    return np.array(cal_df[cal_df["calCoef"] == cal_coef]["value"].values[0], dtype=dtype)


def add_plant_correction(ds, site, node, sensor):
    """
    Recalculates the temperature-, salinity-, and pressure-corrected
    dissolved nitrate concentration using the Plant et al. (2023) update
    to the Sakamoto et al. (2009) algorithm (which also incorporates the
    Sakamoto et al. (2017) pressure correction), and uses the result to
    replace the 'corrected_nitrate_concentration' variable that ships with
    the OOI NUTNR data sets (which was calculated with the original
    Sakamoto et al. (2009) algorithm).

    The original, OOI-delivered Sakamoto (2009) values are preserved in a
    new 'corrected_nitrate_concentration_sakamoto2009' variable so the two
    can still be compared.

    Uses the calibration coefficients (CC_cal_temp, CC_wl, CC_di, CC_eno3,
    CC_eswa) in effect during the deployment -- i.e. no drift correction is
    applied here. Use plant_drift_correction() afterwards (or let it call
    this function for you) to additionally correct for calibration drift
    across the deployment.

    :param ds: A SUNA dataset reprocessed using process_nutnr's
               suna_datalogger or suna_instrument functions
    :param site: The site of the associated SUNA dataset
    :param node: The node of the associated SUNA dataset
    :param sensor: The sensor of the associated SUNA dataset
    :return ds: dataset with 'corrected_nitrate_concentration' recalculated
               using the Plant et al. (2023) algorithm
    """
    # get the calibration coefficients in effect during the deployment
    pre_deploy_cal, _ = _get_pre_post_calibrations(ds, site, node, sensor)

    cal_temp = float(_cal_coefficient(pre_deploy_cal, "CC_cal_temp"))
    wl = _cal_coefficient(pre_deploy_cal, "CC_wl")
    di = _cal_coefficient(pre_deploy_cal, "CC_di")
    eno3 = _cal_coefficient(pre_deploy_cal, "CC_eno3")
    eswa = _cal_coefficient(pre_deploy_cal, "CC_eswa")

    # get the spectral input values -- by this point the dark frames have
    # already been dropped by suna_datalogger/suna_instrument, so treat
    # every record as a light frame
    dark_counts = ds['dark_value_used_for_fit'].values
    data_in = ds['raw_spectral_measurements'].values
    frame_type = np.array(len(dark_counts) * ['Light'], dtype='str')

    # get the CTD temperature, salinity, and (if available) pressure values.
    # NUTNR data sets do not always carry a co-located pressure record; when
    # absent, fall back to no pressure correction (matches Sakamoto 2009
    # behavior) rather than fail outright.
    ctd_t = ds['sea_water_temperature'].values
    ctd_sp = ds['sea_water_practical_salinity'].values
    ctd_p = ds['sea_water_pressure'].values if 'sea_water_pressure' in ds.variables else None

    plant_nitrate = plant2023_tsp_correction(cal_temp, wl, eno3, eswa, di, dark_counts,
                                              ctd_t, ctd_sp, ctd_p, data_in, frame_type)

    # preserve the original, OOI-delivered Sakamoto (2009) value for comparison,
    # but only the first time this is applied to a given dataset
    if 'corrected_nitrate_concentration_sakamoto2009' not in ds.variables:
        ds['corrected_nitrate_concentration_sakamoto2009'] = ds['corrected_nitrate_concentration'].copy()
        ds['corrected_nitrate_concentration_sakamoto2009'].attrs = {
            'long_name': 'Corrected Dissolved Nitrate Concentration (Sakamoto 2009)',
            'units': 'umol L-1',
            'comment': ('Temperature and salinity corrected dissolved nitrate concentration as originally '
                        'delivered by OOI, calculated with the Sakamoto et al. (2009) algorithm. Retained here '
                        'for comparison; superseded by the Plant et al. (2023) corrected value now found in '
                        'corrected_nitrate_concentration.')
        }

    # overwrite corrected_nitrate_concentration with the Plant (2023) results
    ds['corrected_nitrate_concentration'] = (['time'], plant_nitrate)
    ds['corrected_nitrate_concentration'].attrs = {
        'long_name': 'Corrected Dissolved Nitrate Concentration',
        'standard_name': 'mole_concentration_of_nitrate_in_sea_water',
        'comment': ('Temperature, salinity, and (if available) pressure corrected dissolved nitrate '
                    'concentration, calculated using the Plant et al. (2023) update to the Sakamoto et al. '
                    '(2009) algorithm, with the pressure correction from Sakamoto et al. (2017). Supersedes '
                    'the Sakamoto (2009) value originally delivered by OOI, which is preserved in '
                    'corrected_nitrate_concentration_sakamoto2009.'),
        'units': 'umol L-1',
        'data_product_identifier': 'NITRTSC_L2',
        'ancillary_variables': ('sea_water_temperature sea_water_practical_salinity raw_spectral_measurements '
                                'dark_value_used_for_fit')
    }

    # the quality checks include a range test on corrected_nitrate_concentration, so re-run them
    ds['nitrate_sensor_quality_flag'] = quality_checks(ds, 'corrected_nitrate_concentration')

    return ds


def plant_drift_correction(ds, site, node, sensor):
    """
    Apply a drift correction to a processed SUNA dataset using the Plant
    et al. (2023) T-S(-P) algorithm throughout.

    This is a drop-in replacement for process_nutnr.drift_correction() that
    uses plant2023_tsp_correction (instead of the Sakamoto et al. 2009
    ts_corrected_nitrate) both for the baseline 'corrected_nitrate_concentration'
    (via add_plant_correction(), called automatically if not already applied)
    and for the post-cruise-calibration nitrate concentration used to estimate
    the drift -- so drift is estimated consistently within a single T-S(-P)
    correction scheme, rather than mixing Sakamoto (2009) and Plant (2023)
    results.

    :param ds: A SUNA dataset reprocessed using process_nutnr's
               suna_datalogger or suna_instrument functions
    :param site: The site of the associated SUNA dataset
    :param node: The node of the associated SUNA dataset
    :param sensor: The sensor of the associated SUNA dataset
    """
    # make sure corrected_nitrate_concentration has already been recalculated
    # with the Plant (2023) algorithm; if not, do so now so the drift estimate
    # below compares apples-to-apples
    if 'corrected_nitrate_concentration_sakamoto2009' not in ds.variables:
        ds = add_plant_correction(ds, site, node, sensor)

    # ----------------------------------------
    # Get the appropriate calibration files and
    # coefficients: the pre-deployment calibration (used for the baseline
    # corrected_nitrate_concentration above) and the next calibration
    # performed after it (e.g., the post-cruise calibration)
    pre_deploy_cal, post_deploy_cal = _get_pre_post_calibrations(ds, site, node, sensor)

    # ---------------------------------------------------
    # Use the post-cruise DI values to recalculate the
    # Plant (2023) T-S(-P) corrected nitrate
    cal_temp = float(_cal_coefficient(pre_deploy_cal, "CC_cal_temp"))
    wl = _cal_coefficient(pre_deploy_cal, "CC_wl")
    di = _cal_coefficient(post_deploy_cal, "CC_di")
    eno3 = _cal_coefficient(pre_deploy_cal, "CC_eno3")
    eswa = _cal_coefficient(pre_deploy_cal, "CC_eswa")

    # Get the spectral input values
    dark_counts = ds['dark_value_used_for_fit'].values
    data_in = ds['raw_spectral_measurements'].values
    frame_type = np.array(len(dark_counts) * ['Light'], dtype='str')

    # Get the CTD temperature, salinity, and (if available) pressure values
    ctd_t = ds['sea_water_temperature'].values
    ctd_sp = ds['sea_water_practical_salinity'].values
    ctd_p = ds['sea_water_pressure'].values if 'sea_water_pressure' in ds.variables else None

    # Now run with the post-cal di, using the Plant (2023) algorithm
    post_cal_nitrate = plant2023_tsp_correction(cal_temp, wl, eno3, eswa, di, dark_counts,
                                                 ctd_t, ctd_sp, ctd_p, data_in, frame_type)
    ds["post_cal_nitrate"] = (["time"], post_cal_nitrate)
    ds["post_cal_nitrate"].attrs = {
        "long_name": "T-S(-P) Corrected Dissolved Nitrate Concentration",
        "units": "umol L-1",
        "comment": ("Temperature, salinity, and (if available) pressure corrected dissolved nitrate "
                    "concentration recalculated using the post-cruise DI-water calibration and the "
                    "Plant et al. (2023) algorithm.")
    }

    # Get the calibration file names
    pre_deploy_file = pre_deploy_cal["calFile"].unique()[0]
    post_deploy_file = post_deploy_cal["calFile"].unique()[0]

    # Now, calculate a drift rate based on the offset between the
    # initial T-S(-P) corrected nitrogen from the pre-deploy calibration
    # and the post-cruise calibration. This operates on the assumption of
    # a linear drift, thus is LTI
    dno3 = (post_cal_nitrate - ds['corrected_nitrate_concentration']).sel(
        time=slice(ds["time"].min(), ds["time"].min() + pd.Timedelta('1D'))).mean()
    pre_date = pre_deploy_file.split("__")[-1].split("_")[0]
    post_date = post_deploy_file.split("__")[-1].split("_")[0]
    dt = (pd.to_datetime(post_date) - pd.to_datetime(pre_date)).total_seconds() * 1E9
    dno3_dt = dno3 / int(dt)
    delta_t = (ds.time - ds.time.min()).astype('int')

    # Apply the drift correction
    drift_correction = dno3_dt.values * delta_t
    drift_corrected_nitrate = ds["corrected_nitrate_concentration"] + drift_correction
    ds["drift_corrected_nitrate"] = drift_corrected_nitrate
    ds["drift_corrected_nitrate"].attrs = {
        "long_name": "Drift and T-S(-P) Corrected Dissolved Nitrate Concentration",
        "units": "umol L-1",
        "comment": ("Temperature, salinity, and (if available) pressure corrected dissolved nitrate "
                    "concentration (Plant et al. 2023), with linear drift, estimated from the difference "
                    "between pre-and-post cruise DI-water calibrations, removed."),
        "pre_cruise_calibration": pre_deploy_file,
        "post_cruise_calibration": post_deploy_file,
        "dNO3": str(dno3),
        "dt": str(dt)
    }

    return ds
