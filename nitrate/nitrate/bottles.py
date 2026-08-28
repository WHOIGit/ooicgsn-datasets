"""
nitrate.bottles
=================
Shipboard discrete (bottle) nitrate sample handling: matching deployment
and recovery cruises to a reference designator, and computing/applying the
bottle offset correction to the drift-corrected sensor time series.

Functions
---------
remove_last_letter   -- strip a trailing 'A'/'B' cruise-ID suffix so buoy
                        (SBD) and NSIF sensor cruise IDs match the shared
                        ship discrete sample cruise ID
get_deployment_info  -- look up deployment/recovery cruise IDs and start/end
                        times for a list of deployments via M2M
bottle_correction    -- smooth the drift-corrected time series and apply a
                        deployment/recovery bottle-derived offset (and,
                        where a recovery bottle match exists, an
                        additional linear correction), following the
                        approach in Palevsky et al. (2023)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ooi_data_explorations.common import get_sensor_information

from .time_utils import convert_time


def remove_last_letter(x):
    """Match deployment and recovery numbers for each buoy with the datasets.

    Ship discrete sample cruise IDs sometimes carry a trailing 'A'/'B'
    (e.g. distinguishing two legs of the same cruise) that isn't present
    in the sensor-reported deployment/recovery cruise ID -- strip it so the
    two can be matched.
    """
    if x.endswith(('A', 'B')):
        x = x[0:-1]
    else:
        pass
    return x


def get_deployment_info(site, node, sensor, deployments):
    """
    Look up the deployment/recovery cruise IDs and deployment start/end
    times for each deployment number in ``deployments``, via the M2M
    sensor-information endpoint.

    :param site: OOI site (e.g. 'CP01CNSM')
    :param node: OOI node (e.g. 'RID26')
    :param sensor: OOI sensor (e.g. '07-NUTNRB000')
    :param deployments: iterable of deployment numbers
    :return: dict with keys deploymentNumber, uid, deployStart, deployEnd,
             deployCruise, recoverCruise -- suitable for
             ``pd.DataFrame(deployInfo).set_index('deploymentNumber')``
    """
    keys = ['deploymentNumber', 'uid', 'deployStart', 'deployEnd', 'deployCruise', 'recoverCruise']
    deployInfo = {x: [] for x in keys}
    for dN in deployments:
        # Get the sensor info
        sensorInfo = get_sensor_information(site, node, sensor, dN)

        # With the sensor info for a given deployment, get relevant data
        assetUid = sensorInfo[0]['sensor']['uid']

        # Get deployment info
        deployCruise = sensorInfo[0]['deployCruiseInfo']['uniqueCruiseIdentifier']
        deployStart = sensorInfo[0]['eventStartTime']
        deployStart = pd.to_datetime(convert_time(deployStart))

        # Get recovery info
        recoverCruise = sensorInfo[0]['recoverCruiseInfo']['uniqueCruiseIdentifier']
        deployEnd = sensorInfo[0]['eventStopTime']
        deployEnd = pd.to_datetime(convert_time(deployEnd))

        # Save results
        deployInfo['deploymentNumber'].append(int(dN))
        deployInfo['uid'].append(assetUid)
        deployInfo['deployStart'].append(deployStart)
        deployInfo['deployEnd'].append(deployEnd)
        deployInfo['deployCruise'].append(deployCruise)
        deployInfo['recoverCruise'].append(recoverCruise)

    return deployInfo


def bottle_correction(ds, deployments, bottle_data):
    """Apply the bottle correction to a dataset.

    :param ds: a single-deployment, drift-corrected xarray Dataset (must
               carry a 'drift_corrected_nitrate' variable and a 'deployment'
               variable)
    :param deployments: DataFrame as returned by
               ``pd.DataFrame(get_deployment_info(...))``, indexed (or
               indexable) by 'deploymentNumber'
    :param bottle_data: DataFrame of cleaned shipboard discrete nitrate
               samples, with 'Cruise', 'Start Time [UTC]', and
               'Discrete Nitrate [uM]' columns
    :return: (ds, smoothed_data, deployBottles, recoverBottles)
    """
    # First, check that the deployments dataset index is set to the deploymentNumber
    if not deployments.index.name == 'deploymentNumber':
        deployments.set_index(keys='deploymentNumber', drop=True, inplace=True)

    # Next, get the unique deployment number of the dataset
    deployNum = np.unique(ds['deployment'])

    # Get the deployment and recovery cruises
    deployCruise = deployments.loc[deployNum]['deployCruise'].values[0]
    recoverCruise = deployments.loc[deployNum]['recoverCruise'].values[0]

    # Select the associated nitrate data
    deployBottles = bottle_data[bottle_data['Cruise'] == deployCruise].drop(columns='Cruise').groupby('Start Time [UTC]').mean()
    recoverBottles = bottle_data[bottle_data['Cruise'] == recoverCruise].drop(columns='Cruise').groupby('Start Time [UTC]').mean()

    # Next, filter the NUTNR data using a 6H rolling window
    smoothed_data = ds['drift_corrected_nitrate'].to_dataframe().rolling('6H', center=True, closed='both').mean()
    smoothed_data = xr.Dataset(smoothed_data)

    # Find the closest data point in the smoothed data to get the
    deploy_NO3 = deployBottles.reset_index().mean()
    suna_NO3 = smoothed_data.sel(time=deploy_NO3['Start Time [UTC]'], method='nearest')

    # Calculate the bottle offset
    bottle_offset = deploy_NO3['Discrete Nitrate [uM]'] - suna_NO3['drift_corrected_nitrate'].data

    # Now add the offset to the smoothed suna data and full suna drift-corrected
    smoothed_data = smoothed_data + bottle_offset

    # With the data adjusted for the starting offset, calculate the difference at the end
    recover_NO3 = recoverBottles.reset_index().mean()
    suna_NO3 = smoothed_data.sel(time=recover_NO3['Start Time [UTC]'], method='nearest')

    # Check if the time difference between the bottle sample and the identified nearest
    # SUNA measurement exceeds 3 days, in which case ONLY apply the initial offset
    if len(recoverBottles) == 0:
        delta_NO3 = xr.DataArray(
            data=np.zeros(np.shape(smoothed_data['time'])),
            dims='time',
            coords=dict(
                time=smoothed_data.time))
    elif np.abs(suna_NO3['time'].values - recover_NO3['Start Time [UTC]']).to_timedelta64() > pd.Timedelta('3 days'):
        delta_NO3 = xr.DataArray(
            data=np.zeros(np.shape(smoothed_data['time'])),
            dims='time',
            coords=dict(
                time=smoothed_data.time))
    else:
        # Calculate the bottle-derived drift
        dNO3 = recover_NO3['Discrete Nitrate [uM]'] - suna_NO3['drift_corrected_nitrate'].data
        dt = recover_NO3['Start Time [UTC]'] - deploy_NO3['Start Time [UTC]']
        dNO3_dt = dNO3 / dt.to_timedelta64().astype('int')
        delta_NO3 = (smoothed_data.time - np.datetime64(deploy_NO3['Start Time [UTC]'])).astype('int') * dNO3_dt

    # Add in the bottle correction to both the smoothed data and the drift-corrected-data
    smoothed_data = smoothed_data + delta_NO3
    smoothed_data['deployment'] = ds['deployment']
    ds['bottle_corrected_nitrate'] = ds['drift_corrected_nitrate'] + bottle_offset + delta_NO3

    return ds, smoothed_data, deployBottles, recoverBottles
