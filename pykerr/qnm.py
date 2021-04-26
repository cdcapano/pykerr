import os
import pkg_resources
import numpy
import h5py
from scipy.interpolate import CubicSpline

# the maximum spin we'll allow; this is based on what is in the data files
MAX_SPIN = 0.9997

# solar masses in seconds
MTSUN = 4.925491025543576e-06


# we'll cache the splines
_reomega_splines = {}
_imomega_splines = {}


def _create_spline(name, reim, l, m, n):
    """Creates a cubic spline for the specified mode data."""
    # load the data
    lmn = '{}{}{}'.format(l, abs(m), n)
    try:
        dfile = pkg_resources.resource_stream(__name__,
                                              'data/l{}.hdf'.format(l))
    except OSError:
        raise ValueError("unsupported lmn {}{}{}".format(l, m, n))
    with h5py.File(dfile, 'r') as fp:
        # load and convert the spin
        x = 1e-4 * fp['spin'][()]
        y = fp[lmn][name][()]
        if reim == 're':
            y = y.real
        elif reim == 'im':
            y = y.imag
    return CubicSpline(x, y, axis=0, bc_type='not-a-knot', extrapolate=False)


def _getspline(name, reim, l, m, n, cache):
    """Gets a spline."""
    try:
        spline = cache[l, abs(m), n]
    except KeyError:
        spline = _create_spline(name, reim, l, m, n)
        cache[l, abs(m), n] = spline
    return spline


def _checkspin(spin):
    """Checks that the spin is in bounds."""
    if abs(spin) > MAX_SPIN:
        raise ValueError("|spin| must be < {}".format(MAX_SPIN))
    return


def _qnmomega(spin, l, m, n):
    """Returns the dimensionless complex angular frequency of a Kerr BH.

    Parmeters
    ---------
    spin : float
        Dimensionless spin. Must be in [-0.9997, 0.9997].
    l : int
        The l index. Up to l=7 is supported.
    m : int
        The m index. All +/-m for the given l are supported.
    n : int
        The overtone number (where n=0 is the fundamental mode). Up to n=7 is
        supported.

    Returns
    -------
    complex :
        The complex angular frequency.
    """
    _checkspin(spin)
    respline = _getspline('omega', 're', l, m, n, _reomega_splines)
    imspline = _getspline('omega', 'im', l, m, n, _imomega_splines)
    omega = respline(spin) + 1j*imspline(spin)
    # omega_{-m} = -omega_{m}.conj()
    if m < 0:
        omega = -omega.conj()
    return omega


# vectorize
_npqnmomega = numpy.frompyfunc(_qnmomega, 4, 1)

def qnmomega(spin, l, m, n):
    out = _npqnmomega(spin, l, m, n)
    if isinstance(out, numpy.ndarray):
        out = out.astype(numpy.complex)
    return out

qnmomega.__doc__ = _qnmomega.__doc__


def _qnmfreq(mass, spin, l, m, n):
    """Returns the QNM frequency for a Kerr black hole.

    Parameters
    ----------
    mass : float
        Mass of the object (in solar masses).
    spin : float
        Dimensionless spin. Must be in [-0.9997, 0.9997].
    l : int
        The l index. Up to l=7 is supported.
    m : int
        The m index. All +/-m for the given l are supported.
    n : int
        The overtone number (where n=0 is the fundamental mode). Up to n=7 is
        supported.

    Returns
    -------
    float :
        The frequency (in Hz) of the requested mode.
    """
    _checkspin(spin)
    spline = _getspline('omega', 're', l, m, n, _reomega_splines)
    reomega = spline(spin)
    # negate the frequency if m < 0
    if m < 0:
        reomega = -reomega
    return reomega / (2*numpy.pi*mass*MTSUN)

# vectorize
_npqnmfreq = numpy.frompyfunc(_qnmfreq, 5, 1)

def qnmfreq(mass, spin, l, m, n):
    out = _npqnmfreq(mass, spin, l, m, n)
    if isinstance(out, numpy.ndarray):
        out = out.astype(numpy.float)
    return out

qnmfreq.__doc__ = _qnmfreq.__doc__


def _qnmtau(mass, spin, l, m, n):
    """"Returns the QNM damping time for a Kerr black hole.

    Parameters
    ----------
    mass : float
        Mass of the object (in solar masses).
    spin : float
        Dimensionless spin. Must be in [-0.9997, 0.9997].
    l : int
        The l index. Up to l=7 is supported.
    m : int
        The m index. All +/-m for the given l are supported.
    n : int
        The overtone number (where n=0 is the fundamental mode). Up to n=7 is
        supported.

    Returns
    -------
    float :
        The frequency (in Hz) of the requested mode.
    """
    _checkspin(spin)
    spline = _getspline('omega', 'im', l, m, n, _imomega_splines)
    # Note: Berti et al. [arXiv:0512160] used the convention
    # h+ + ihx ~ e^{iwt}, (see Eq. 2.4) so we
    # need to negate the spline for tau to have the right sign.
    return -mass*MTSUN / spline(spin)

# vectorize
_npqnmtau = numpy.frompyfunc(_qnmtau, 5, 1)

def qnmtau(mass, spin, l, m, n):
    out = _npqnmtau(mass, spin, l, m, n)
    if isinstance(out, numpy.ndarray):
        out = out.astype(numpy.float)
    return out

qnmtau.__doc__ = _qnmtau.__doc__
