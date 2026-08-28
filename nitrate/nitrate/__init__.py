"""
nitrate
=======
OOI CGSN/Pioneer-NES and Irminger Sea SUNA/NUTNR nitrate processing pipeline.

Modules
-------
time_utils    : OOI timestamp conversions
resample      : burst-averaging (median + MAD) to a fixed time interval
ts_correction : Plant (2023) and Sakamoto (2009) T-S(-P) nitrate algorithms
calibration   : calibration coefficient lookup, Plant correction, drift correction
qc            : SUNA instrument quality checks (QARTOD-style flags)
bottles       : deployment info lookup + shipboard bottle offset correction
cf            : CF-1.11 compliance, final metadata cleanup
"""

from . import time_utils, resample, ts_correction, calibration, qc, bottles, cf

__version__ = "0.1.0"
__all__ = [
    "time_utils", "resample", "ts_correction", "calibration",
    "qc", "bottles", "cf",
]
