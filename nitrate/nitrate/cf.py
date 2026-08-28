"""
nitrate.cf
==========
CF-1.11 compliance and final metadata/attribute cleanup for a finished,
ready-to-publish nitrate dataset.

Functions
---------
finalize_cf_compliance -- fix issues flagged by the IOOS Compliance
                           Checker (CF-1.11): monotonic time, Conventions
                           attribute, units_metadata, missing/invalid
                           standard_name and long_name, stale
                           ancillary_variables, flag_values dtype,
                           featureType
clean_netcdf            -- trim global attributes down to a curated set,
                           rebuild the title from the OOI vocabulary, and
                           append a processing history entry
fix_serial_numbers       -- patch mis-reported instrument serial numbers
                           for specific deployments
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ooi_data_explorations.common import get_vocabulary


def finalize_cf_compliance(ds):
    """
    Clean up a final, ready-to-publish nitrate dataset so it passes the IOOS
    Compliance Checker (CF-1.11) -- call this right before the final
    ds.to_netcdf(...) call.

    Addresses, in order:
      Sec 1.2   non-monotonic / duplicate time values
      Sec 2.6   Conventions global attribute
      Sec 3.1.2 units_metadata on temperature-like variables
      Sec 3.3   missing long_name/standard_name, invalid standard_name values
      Sec 3.4   ancillary_variables pointing at variables that were dropped
      Sec 3.5   flag_values dtype mismatch with its own variable
      Sec 5 /   missing lat/lon coordinate variables referenced by 'coordinates'
      Sec 5.1
      Sec 9.1   featureType so timeseries variables aren't detected as 'point'
    """
    ds = ds.copy()

    # ---- Sec 1.2: time must be strictly monotonic, with no duplicates ----
    ds = ds.sortby('time')
    _, index = np.unique(ds['time'].values, return_index=True)
    ds = ds.isel(time=index)

    # ---- Sec 2.6: Conventions global attribute ----
    ds.attrs['Conventions'] = 'CF-1.11'

    # ---- Sec 5 / Sec 5.1: add lat/lon as proper coordinate variables ----
    # (OOI moorings are fixed-position, so these are scalars pulled from the
    # existing global lat/lon attributes rather than a real per-record track)
    if 'lat' not in ds.coords and 'lat' in ds.attrs:
        ds = ds.assign_coords(lat=float(ds.attrs['lat']))
        ds['lat'].attrs = {'standard_name': 'latitude', 'long_name': 'Latitude', 'units': 'degrees_north'}
    if 'lon' not in ds.coords and 'lon' in ds.attrs:
        ds = ds.assign_coords(lon=float(ds.attrs['lon']))
        ds['lon'].attrs = {'standard_name': 'longitude', 'long_name': 'Longitude', 'units': 'degrees_east'}

    # ---- Sec 3.3 / Sec 5.1: time coordinate needs standard_name/long_name ----
    ds['time'].attrs.setdefault('standard_name', 'time')
    ds['time'].attrs.setdefault('long_name', 'Time')
    ds['time'].attrs.setdefault('units_metadata', 'leap_seconds: none')

    # ---- Sec 3.3: long_name for variables that were missing one ----
    default_long_names = {
        'nitrate_concentration_qc_flag': 'Nitrate Concentration QC Flag',
        'burst_median_absolute_deviation': 'Burst Median Absolute Deviation',
        'deployment': 'Deployment Number',
    }
    for v, long_name in default_long_names.items():
        if v in ds.variables:
            ds[v].attrs.setdefault('long_name', long_name)

    # ---- Sec 3.3: drift_corrected_nitrate / bottle_corrected_nitrate carry an
    # invalid standard_name (their own variable name isn't a real CF term).
    # They're still the same physical quantity as corrected_nitrate_concentration,
    # just further corrected -- CF standard names describe the quantity, not
    # the correction applied, so re-use the valid one. Scan by the *attribute
    # value*, not the variable name -- a derived/sibling variable (e.g. a MAD
    # or QC variable that inherited attrs via keep_attrs during a resample
    # step) can carry this same invalid standard_name under a different
    # variable name.
    invalid_standard_names = {'drift_corrected_nitrate', 'bottle_corrected_nitrate'}
    for v in ds.data_vars:
        if ds[v].attrs.get('standard_name') in invalid_standard_names:
            ds[v].attrs['standard_name'] = 'mole_concentration_of_nitrate_in_sea_water'

    # ---- Sec 3.1.2: units_metadata recommended for temperature-like variables ----
    if 'sea_water_temperature' in ds.variables:
        ds['sea_water_temperature'].attrs.setdefault('units_metadata', 'temperature: on_scale')

    # ---- Sec 3.4: drop any ancillary_variables references to variables that
    # aren't actually present in this (trimmed-down) final dataset ----
    for v in ds.data_vars:
        anc = ds[v].attrs.get('ancillary_variables')
        if not anc:
            continue
        remaining = [a for a in anc.split() if a in ds.variables]
        if remaining:
            ds[v].attrs['ancillary_variables'] = ' '.join(remaining)
        else:
            del ds[v].attrs['ancillary_variables']

    # ---- Sec 3.5: flag_values dtype must match the variable's own dtype ----
    for v in ds.data_vars:
        if 'flag_values' in ds[v].attrs:
            # flags should always be integers; cast the variable itself if it
            # somehow ended up float (e.g. after a resample/median step)
            if not np.issubdtype(ds[v].dtype, np.integer):
                ds[v] = ds[v].astype('int32')
            ds[v].attrs['flag_values'] = np.asarray(ds[v].attrs['flag_values'], dtype=ds[v].dtype)

    # ---- Sec 9.1: declare the featureType so timeseries variables aren't
    # mis-detected as isolated 'point' features ----
    ds.attrs['featureType'] = 'timeSeries'

    return ds


def _make_title(ds: xr.Dataset) -> str:
    """Build a dataset title from OOI vocabulary if available."""
    subsite = ds.attrs.get("subsite", "")
    node = ds.attrs.get("node", "")
    sensor = ds.attrs.get("sensor", "")
    try:
        vocab = get_vocabulary(subsite, node, sensor)[0]
        return " : ".join([vocab["tocL1"], vocab["tocL2"], vocab["tocL3"], vocab["instrument"]])
    except Exception:
        return f"{subsite}-{node}-{sensor}"


def _make_history(new_attrs):
    """Append a processing history entry to whatever history already exists."""
    existing = new_attrs.get("history", "")
    entry = f"{pd.Timestamp.now(tz='UTC').isoformat()} Plant et al. 2023 correction applied, data corrected for drift and discrete bottle offsets."
    new_history = f"{entry}\n{existing}".strip()
    return new_history


def clean_netcdf(ds):
    """
    Trim a final dataset's global attributes down to a curated set, refresh
    the corrected_nitrate_concentration_qc_flag documentation, rebuild the
    title from the OOI vocabulary, and append a processing history entry.

    Call this before finalize_cf_compliance() in the finalize stage.
    """
    ds['corrected_nitrate_concentration_qc_flag'].attrs['long_name'] = 'Corrected Dissolved Nitrate Concentration Quality Flag'
    ds['corrected_nitrate_concentration_qc_flag'].attrs['comment'] = (
        'This quality flag represents an assessment of the nitrate concentration that is corrected for '
        'temperature, salinity following Plant et al (2023). Checks include assessment of RMSE of the '
        'spectral measurements, absorptions at 254 nm and 350 nm wavelengths, dark values, spectral '
        'averages, and a range check based on instrument calibration.'
    )
    old_attrs = ds.attrs
    tstart = str(ds['time'].min().values) + "Z"
    tend = str(ds['time'].max().values) + "Z"
    new_attrs = {
        'title': old_attrs['title'],
        'history': old_attrs['history'],
        'comment': old_attrs['comment'],
        'sourceUrl': old_attrs['sourceUrl'],
        'featureType': old_attrs['featureType'],
        'publisher_name': old_attrs['publisher_name'],
        'references': old_attrs['references'],
        'Metadata_Conventions': old_attrs['Metadata_Conventions'],
        'nodc_template_version': old_attrs['nodc_template_version'],
        'creator_name': old_attrs['creator_name'],
        'standard_name_vocabulary': old_attrs['standard_name_vocabulary'],
        'acknowledgement': old_attrs['acknowledgement'],
        'project': old_attrs['project'],
        'source': "-".join(old_attrs['source'].split("-")[0:4]),
        'subsite': old_attrs['subsite'],
        'node': old_attrs['node'],
        'sensor': old_attrs['sensor'],
        'Manufacturer': old_attrs['Manufacturer'],
        'Model': old_attrs['ModelNumber'],
        'Description': old_attrs['Description'],
        'time_coverage_start': tstart,
        'time_coverage_end': tend,
        'geospatial_lat_min': old_attrs['geospatial_lat_min'],
        'geospatial_lat_max': old_attrs['geospatial_lat_max'],
        'geospatial_lat_units': old_attrs['geospatial_lat_units'],
        'geospatial_lat_resolution': old_attrs['geospatial_lat_resolution'],
        'geospatial_lon_min': old_attrs['geospatial_lon_min'],
        'geospatial_lon_max': old_attrs['geospatial_lon_max'],
        'geospatial_lon_units': old_attrs['geospatial_lon_units'],
        'geospatial_lon_resolution': old_attrs['geospatial_lon_resolution'],
        'geospatial_vertical_units': old_attrs['geospatial_vertical_units'],
        'geospatial_vertical_resolution': old_attrs['geospatial_vertical_resolution'],
        'geospatial_vertical_positive': old_attrs['geospatial_vertical_positive'],
        'lat': old_attrs['lat'],
        'lon': old_attrs['lon'],
    }
    # Make the title
    new_attrs['title'] = _make_title(ds)
    # Make a new history
    new_attrs['history'] = _make_history(new_attrs)
    ds.attrs = new_attrs
    return ds


def fix_serial_numbers(ds, deployment, serial_number):
    """
    Patch the reported instrument serial_number for a specific deployment
    (e.g. when the sensor metadata is known to be wrong for that
    deployment). Pass (deployment, serial_number) pairs looked up from the
    asset management record for the refdes in question.
    """
    ind, = np.where(ds['deployment'] == deployment)
    serials = ds['serial_number'].values
    serials[ind] = serial_number
    attrs = ds['serial_number'].attrs.copy()
    ds['serial_number'] = (ds['serial_number'].dims, serials.astype(ds['serial_number'].dtype))
    ds['serial_number'].attrs = attrs
    return ds
