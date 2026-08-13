"""
adcp.physics
============
Oceanographic formulae used in ADCP data processing.
"""

from __future__ import annotations

import numpy as np


def chen_millero(
    T: "np.ndarray",
    S: "np.ndarray",
    P: "np.ndarray",
) -> "np.ndarray":
    """
    Speed of sound in seawater — Chen & Millero (1977) / UNESCO formulation.

    Parameters
    ----------
    T : float or array-like
        In-situ temperature (°C)
    S : float or array-like
        Practical salinity (PSU)
    P : float or array-like
        Pressure (dbar)  [1 dbar ≈ 1 m depth]

    Returns
    -------
    c : float or array-like
        Speed of sound (m s⁻¹)

    Valid ranges
    ------------
    T : 0–40 °C  |  S : 0–40 PSU  |  P : 0–10 000 dbar

    Reference
    ---------
    Chen, C.-T. and Millero, F.J. (1977). Speed of sound in seawater at high
    pressures. Journal of the Acoustical Society of America, 62(5), 1129–1135.
    [UNESCO Technical Papers in Marine Science No. 44, 1983]
    """
    # Convert dbar → bar
    P = np.asarray(P, dtype=float) * 0.1
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)

    # ── Pure-water term Cw(T, P) ─────────────────────────────────────────────
    Cw = (
        (1402.388 + T * (5.03830 + T * (-5.81090e-2 + T * (3.3432e-4 + T * (-1.47797e-6 + T * 3.1419e-9)))))
        + (0.153563 + T * (6.8999e-4 + T * (-8.1829e-6 + T * (1.3632e-7 + T * -6.1260e-10)))) * P
        + (3.1260e-5 + T * (-1.7111e-6 + T * (2.5986e-8 + T * (-2.5353e-10 + T * 1.0415e-12)))) * P ** 2
        + (-9.7729e-9 + T * (3.8513e-10 + T * -2.3654e-12)) * P ** 3
    )

    # ── Salinity term A(T, P) ────────────────────────────────────────────────
    A = (
        (1.389 + T * (-1.262e-2 + T * (7.166e-5 + T * (2.008e-6 + T * -3.21e-8))))
        + (9.4742e-5 + T * (-1.2580e-5 + T * (-6.4885e-8 + T * (1.0507e-8 + T * -2.0122e-10)))) * P
        + (-3.9064e-7 + T * (9.1041e-9 + T * (-1.6002e-10 + T * 7.988e-12))) * P ** 2
        + (1.100e-10 + T * (6.649e-12 + T * -3.389e-13)) * P ** 3
    )

    # ── S^(3/2) term B(T, P) ─────────────────────────────────────────────────
    B = (-1.922e-2 + T * -4.42e-5) + (7.3637e-5 + T * 1.7950e-7) * P

    # ── S^2 term D(P) ────────────────────────────────────────────────────────
    D = 1.727e-3 - 7.9836e-6 * P

    return Cw + A * S + B * S ** 1.5 + D * S ** 2