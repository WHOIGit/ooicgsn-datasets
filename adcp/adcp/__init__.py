"""
adcp
====
OOI CGSN moored ADCP processing pipeline.
 
Modules
-------
download  : M2M data retrieval
merge     : multi-stream merging and variable crosswalk
ctd       : CTD calibration against ship bottle data
gap_fill  : piecewise-linear + harmonic gap filling
qc        : QARTOD and TRDI instrument QC
regrid    : interpolation to common depth grid
cf        : CF-1.11 compliance, beam stacking, metadata
physics   : oceanographic formulae (sound speed, etc.)
"""
 
from . import download, merge, ctd, gap_fill, qc, regrid, cf, physics
 
__version__ = "0.1.0"
__all__ = ["cf", "ctd_calibration", "ctd", "download", "gap_fill",
           "merge", "ocean_gap_fill", "physics", "qc", "regrid"]