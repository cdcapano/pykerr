import numpy
from lalsimulation import SimBlackHoleRingdownSpheroidalWaveFunction \
    as _spheroidal
from lal import SpinWeightedSphericalHarmonic as _spherical

"""Functions to calculate spheroidal and spherical harmonics.
"""

def slm(inclination, spin, l, m, n=0, s=-2, azimuthal=0.):
    """Calculate the spin-weighted spheroidal harmonic.

    Uses functions from lalsimulation. Currently, only the fundamental
    overtone is supported.

    Parameters
    ----------
    inclination : float or array
        The inclination angle of the observer with respect to the black hole's
        spin.
    spin : float
        The dimensionless spin of the black hole.
    l : int
        The l index.
    m : int
        The m index.
    n : int, optional
        The overtone number. Default is 0. Currently, only 0 is supported.
    s : int, optional
        The spin number. Default is -2.
    azimuthal : float or array, optional
        The azimuthal angle of the observer. Default is 0.

    Returns
    -------
    complex :
        The value of the spheroidal harmonic.
    """
    # for some reason, the spheriodal functions in lalsimulation are off
    # by the following factor, which is obtained by comparing the spherioidal
    # harmonic at zero spin to the spherical harmonic
    fac = (-1)**l * 0.39894228040143276
    # we use the convention that the frequency of the -m mode = -freq of the
    # +m mode, with spins between -1 and 1
    negspin = spin < 0
    spin = abs(spin)
    negm = m < 0
    m = abs(m)
    if negspin:
        m = -m
    if negm:
        # flip the inclination and negate
        inclination = numpy.pi - inclination
        fac *= -1
    fac *= numpy.exp(1j * m * azimuthal)
    try:
        return fac * _spheroidal(inclination, spin, l, m, s)
    except TypeError:
        # will get this if some of the inputs are arrays
        inclination = numpy.atleast_1d(inclination)
        return fac * numpy.array([
            _spheroidal(inc, spin, l, m, s)
            for inc in inclination.ravel()]).reshape(inclination.shape)


def ylm(inclination, l, m, s=-2, azimuthal=0.):
    """Calculate the spin-weighted spherical harmonic.

    Uses functions from lal.

    Parameters
    ----------
    inclination : float or array
        The polar angle.
    l : int
        The l index.
    m : int
        The m index.
    s : int, optional
        The spin number. Default is -2.
    azimuthal : float or array, optional
        The azimuthal angle. Default is 0.

    Returns
    -------
    complex :
        The value of the spherical harmonic.
    """
    try:
        return _spherical(inclination, azimuthal, s, l, m)
    except TypeError:
        # will get this if some of the inputs are arrays
        inclination = numpy.atleast_1d(inclination)
        azimuthal = numpy.atleast_1d(azimuthal)
        incphi = numpy.broadcast(inclination, azimuthal)
        return numpy.array([
            _spherical(inc, phi, s, l, m)
            for inc, phi in incphi]).reshape(incphi.shape)
