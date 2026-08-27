# Background

The OOI deploys Acoustic Doppler Current Profilers (ADCPs) as a core physical oceanographic instrument across all arrays to measure water column velocity. ADCPs use acoustics to measure three-dimensional water-current velocity profiles above or below the sensor. Sound waves ranging from 75 kHz to 1 MHz, depending on model of ADCP, emitted by the profiler scatter off suspended particles and return to the sensor, which calculates velocity by measuring the Doppler shift of the returning signal. Higher frequencies provide finer vertical resolution at shallower depths, while lower frequencies penetrate deeper into the water column.

# Methods

## Datasets

ADCPs deployed on the Global Arrays are shown in [Table 1](#table-1). All Global ADCPs are deployed at a nominal depth of 500 m, oriented upward, and configured to sample beyond the expected ocean surface. ADCPs deployed on the Global Flanking Moorings (FLMA/FLMB) sample once per hour with an ensemble of 27 pings spaced 8 seconds apart. Surface Mooring (SUMO) ADCPs sample once every 3 hours with an ensemble of 80 pings spaced 2.15 seconds apart. ADCPs deployed on the Coastal Pioneer Arrays ([Table 2](#table-2)) are deployed at a nominal depth of 1 m above the seafloor, oriented upward, and similarly configured to sample beyond the expected ocean surface. The 150 kHz ADCPs (ADCPT-F/G) sample once every 30 minutes with an ensemble of 90 pings spaced 2 seconds apart. The 75 kHz ADCPs (ADCPS-J/L) sample once per hour with an ensemble of 72 pings spaced 2.5 seconds apart. These are baseline sampling configurations that may vary based on power availability, deployment duration, and other technical considerations. Blanking distances and cell sizes are adjusted to optimize coverage for the deployed depth and are included in the dataset as parameters.

## Processing

### ADCP

ADCP datasets are downloaded from the OOI Gold Copy THREDDS server and merged on a deployment-by-deployment basis into a single dataset, harmonizing parameter names across data delivery methods. Associated bin depths are calculated from the blanking distance and cell size for each deployment. Practical salinity and in-situ temperature from the co-located CTD are then resampled to the ADCP time basis, and the speed of sound is recalculated at the transducer face following Chen & Millero (1977).

Quality control (QC) of sensor engineering data follows recommendations from Teledyne RDI using their QC model version 12.1. Data with pitch or roll exceeding 20° are flagged as bad; data with pitch or roll exceeding 15° are flagged as suspect. The QC model computes good and suspect thresholds from a weighted combination of correlation magnitude, percent bad beams, and error velocity, parameterized by instrument model (e.g. 75 kHz vs. 150 kHz), number of pings per ensemble, and cell size. Each sea water velocity measurement is assigned a QC flag of 1 (good), 3 (suspect), 4 (bad), or 9 (not evaluated).

Bins affected by sidelobe interference are identified following Lentz et al. (2022):

$$z_{ic} = h_a \left[1 - \cos(\theta)\right] + \frac{3\,\Delta z}{2}$$

where $z_{ic}$ is the depth above which sidelobe interference is expected, $h_a$ is the transducer face depth, $\theta$ is the beam angle, and $\Delta z$ is the cell size. Bins shallower than $z_{ic}$ are flagged as bad.

The data are then regridded to a standardized depth grid based on the nominal cell size for the deployed location using nearest-neighbor interpolation. QC flags are propagated conservatively, where the regridded bin inherits the worst flag among the contributing source bins (e.g. interpolating between bins with flags 1 and 3 yields a flag of 3).

The regridded data are then quality-controlled following IOOS QARTOD (Quality Assurance of Real-Time Oceanographic Data) guidelines. [Table 3](#table-3) summarizes which tests are applied to each parameter. The Gross Range test uses a fail range corresponding to the instrument's sensor limits, with a suspect range of ±3σ of the observed data. The Climatology test fits a two-cycle harmonic with a linear trend to the observed data, computes monthly 3σ bounds, and sets the suspect range as the fitted value ±3σ for each calendar month. Two-dimensional parameters are first grouped into World Ocean Atlas (WOA) standard depth bins before the respective tests are applied. Each datum is assigned a flag for every test executed, and a summary flag equal to the most significant (highest) flag across all tests.

Finally, dataset metadata are aligned with CF-1.11 conventions.

### CTD

Prior to being merged into the ADCP dataset, co-located CTD data are processed separately. On a deployment-by-deployment basis, temperature and salinity observations from the co-located CTD are corrected for sensor drift and validated against discrete observations collected via Niskin bottle and ship CTD during deployment and recovery cruises. Gaps in the CTD record are then filled by fitting the following model independently to temperature and salinity:

$$V(t) = \beta_0 + \beta_1 t + \beta_2 \max(0,\, t - t_{\text{break}}) + A_1\sin(2\pi t) + B_1\cos(2\pi t) + A_2\sin(4\pi t) + B_2\cos(4\pi t)$$

where $t$ is decimal year and $t_{\text{break}}$ is determined via grid search independently for each variable, such that temperature and salinity inflection points need not coincide. Gaps are identified as time steps exceeding a defined threshold; both extended gaps and isolated missing values are filled by evaluating the model at those times. The resulting gap-filled temperature and salinity records are then interpolated to the ADCP time basis.


<a id="table-1"></a>
**Table 1.** OOI Global Array ADCP instruments.
| Reference Designator | Array | Site | Node | Manufacturer | Model | Nominal Depth |
|---|---|---|---|---|---|---|
| GS03FLMB-RIM01-02-ADCPSL007 | Global Southern Ocean | Flanking Subsurface Mooring B | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 500.0 |
| GS03FLMA-RIM01-02-ADCPSL003 | Global Southern Ocean | Flanking Subsurface Mooring A | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 500.0 |
| GS01SUMO-RII11-02-ADCPSN010 | Global Southern Ocean | Apex Surface Mooring | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 500.0 |
| GP03FLMB-RIM01-02-ADCPSL007 | Global Station Papa | Flanking Subsurface Mooring B | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 500.0 |
| GP03FLMA-RIM01-02-ADCPSL003 | Global Station Papa | Flanking Subsurface Mooring A | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 500.0 |
| GI03FLMB-RIM01-02-ADCPSL007 | Global Irminger Sea | Flanking Subsurface Mooring B | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 500.0 |
| GI03FLMA-RIM01-02-ADCPSL003 | Global Irminger Sea | Flanking Subsurface Mooring A | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 500.0 |
| GI01SUMO-RII11-02-ADCPSN010 | Global Irminger Sea | Apex Surface Mooring | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 500.0 |
| GA03FLMB-RIM01-02-ADCPSL007 | Global Argentine Basin | Flanking Subsurface Mooring B | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 500.0 |
| GA03FLMA-RIM01-02-ADCPSL003 | Global Argentine Basin | Flanking Subsurface Mooring A | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 500.0 |
| GA01SUMO-RII11-02-ADCPSN010 | Global Argentine Basin | Apex Surface Mooring | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 500.0 |

<a id="table-2"></a>
**Table 2.** OOI Coastal Pioneer moored ADCP instruments.
| Reference Designator | Array | Site | Node | Manufacturer | Model | Nominal Depth |
|---|---|---|---|---|---|---|
| CP14SEPM-RII01-02-ADCPSL010 | Coastal Pioneer MAB | Southeastern Profiler Mooring | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 281.0 |
| CP14NEPM-RII01-02-ADCPSL010 | Coastal Pioneer MAB | Northeastern Profiler Mooring | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 281.0 |
| CP13SOPM-RII01-02-ADCPTG010 | Coastal Pioneer MAB | Southern Profiler Mooring | Mooring Riser | Teledyne RDI | WorkHorse Sentinel150khz - inductive | 75.0 |
| CP13NOPM-RII01-02-ADCPTG010 | Coastal Pioneer MAB | Northern Profiler Mooring | Mooring Riser | Teledyne RDI | WorkHorse Sentinel150khz - inductive | 79.0 |
| CP13EAPM-RII01-02-ADCPTG010 | Coastal Pioneer MAB | Eastern Profiler Mooring | Mooring Riser | Teledyne RDI | WorkHorse Sentinel150khz - inductive | 79.0 |
| CP11SOSM-MFD37-02-ADCPTF000 | Coastal Pioneer MAB | Southern Surface Mooring | Seafloor Multi-Function Node (MFN) | Teledyne RDI | WorkHorse Sentinel 150khz | 99.0 |
| CP11NOSM-MFD37-02-ADCPTF000 | Coastal Pioneer MAB | Northern Surface Mooring | Seafloor Multi-Function Node (MFN) | Teledyne RDI | WorkHorse Sentinel 150khz | 99.0 |
| CP04OSSM-MFD35-01-ADCPSJ000 | Coastal Pioneer NES | Offshore Surface Mooring | Seafloor Multi-Function Node (MFN) | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz | 450.0 |
| CP04OSPM-RII01-02-ADCPSL010 | Coastal Pioneer NES | Offshore Profiler Mooring | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 425.0 |
| CP03ISSM-MFD35-01-ADCPTF000 | Coastal Pioneer NES | Inshore Surface Mooring | Seafloor Multi-Function Node (MFN) | Teledyne RDI | WorkHorse Sentinel 150khz | 91.5 |
| CP03ISPM-RII01-02-ADCPTG010 | Coastal Pioneer NES | Inshore Profiler Mooring | Mooring Riser | Teledyne RDI | WorkHorse Sentinel150khz - inductive | 70.0 |
| CP02PMUO-RII01-02-ADCPSL010 | Coastal Pioneer NES | Upstream Offshore Profiler Mooring | Mooring Riser | Teledyne RDI | WorkHorse LongRanger Sentinel 75khz - inductive | 425.0 |
| CP02PMUI-RII01-02-ADCPTG010 | Coastal Pioneer NES | Upstream Inshore Profiler Mooring | Mooring Riser | Teledyne RDI | WorkHorse Sentinel150khz - inductive | 70.0 |
| CP02PMCO-RII01-02-ADCPTG010 | Coastal Pioneer NES | Central Offshore Profiler Mooring | Mooring Riser | Teledyne RDI | WorkHorse Sentinel150khz - inductive | 125.0 |
| CP02PMCI-RII01-02-ADCPTG010 | Coastal Pioneer NES | Central Inshore Profiler Mooring | Mooring Riser | Teledyne RDI | WorkHorse Sentinel150khz - inductive | 104.0 |
| CP01CNSM-MFD35-01-ADCPTF000 | Coastal Pioneer NES | Central Surface Mooring | Seafloor Multi-Function Node (MFN) | Teledyne RDI | WorkHorse Sentinel 150khz | 133.0 |
| CP01CNPM-RII01-02-ADCPTG010 | Coastal Pioneer NES | Central Profiler Mooring | Mooring Riser | Teledyne RDI | WorkHorse Sentinel150khz - inductive | 70.0 |

<a id="table-3"></a>
**Table 3.** Executed QARTOD-tests for a given parameter in the ADCP datasets
| Parameter | Gross Range | Climatology |
| --------- | ----------- | ----------- |
| sea_water_temperature | X | X |
| sea_water_practical_salinity | X | X |
| eastward_sea_water_velocity | X | N/A |
| northward_sea_water_velocity | X | N/A |


## References
Chen, C.-T. and Millero, F.J. (1977). Speed of sound in seawater at high pressures. _Journal of the Acoustical Society of America_, 62(5), 1129–1135. [UNESCO Technical Papers in Marine Science No. 44, 1983]

Lentz, S. J., Kirincich, A., and Plueddemann, A. J. (2022) A Note on the Depth of Sidelobe Contamination in Acoustic Doppler Current Profiles. _J. Atmos. Oceanic Technol._, 39, 31–35. https://doi.org/10.1175/JTECH-D-21-0075.1.