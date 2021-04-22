import os
import pkg_resources
import numpy
import h5py
from scipy.interpolate import CubicSpline


# solar masses in seconds
MTSUN = 4.925491025543576e-06


# we'll cache the splines
_reomega_splines = {}
_imomega_splines = {}


def _create_spline(name, l, m, n):
    """Creates a cubic spline for the specified mode data."""
    # load the data
    lmn = '{}{}{}'.format(l, abs(m), n)
    try:
        dfile = pkg_resources.resource_stream(__name__,
                                              'data/l{}.hdf'.format(l))
    except OSError:
        raise ValueError("unsupported lmn {}{}{}".format(l, m, n))
    with h5py.File(dfile, 'r') as fp:
        x = fp[lmn]['spin'][()]
        y = fp[lmn][name][()]
    return CubicSpline(x, y, axis=0, bc_type='natural', extrapolate=False)


def kerr_freq(mass, spin, l, m, n):
    """Returns the QNM frequency for a Kerr black hole.

    Parameters
    ----------
    mass : float
        Mass of the object (in solar masses).
    spin : float
        Dimensionless spin. Must be in [-0.9999, 0.9999].
    l : int
        The l index.
    m : int
        The m index.
    n : int
        The overtone number (where n=0 is the fundamental mode).

    Returns
    -------
    float :
        The frequency (in Hz) of the requested mode.
    """
    if abs(spin) > 0.9999:
        raise ValueError("|spin| must be < 0.9999")
    try:
        spline = _reomega_splines[l, abs(m), n]
    except KeyError:
        spline = _create_spline('omegaR', l, m, n)
        _reomega_splines[l, abs(m), n] = spline
    # if m is 0, use the absolute value of the spin
    if m == 0:
        spin = abs(spin)
    # negate the frequency if m < 0
    sign = (-1)**int(m < 0)
    return sign * spline(spin) / (2*numpy.pi*mass*MTSUN)


def kerr_tau(mass, spin, l, m, n):
    """"Returns the QNM damping time for a Kerr black hole.

    Parameters
    ----------
    mass : float
        Mass of the object (in solar masses).
    spin : float
        Dimensionless spin. Must be in [-0.9999, 0.9999].
    l : int
        The l index.
    m : int
        The m index.
    n : int
        The overtone number (where n=0 is the fundamental mode).

    Returns
    -------
    float :
        The frequency (in Hz) of the requested mode.
    """
    if abs(spin) > 0.9999:
        raise ValueError("|spin| must be < 0.9999")
    try:
        spline = _imomega_splines[l, abs(m), n]
    except KeyError:
        spline = _create_spline('omegaI', l, m, n)
        _imomega_splines[l, abs(m), n] = spline
    # if m is 0, use the absolute value of the spin
    if m == 0:
        spin = abs(spin)
    return -mass*MTSUN / spline(spin)
