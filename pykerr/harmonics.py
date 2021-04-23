import numpy

from .qnm import (_getspline, kerr_omega, _checkspin)

"""Functions to calculate spheroidal and spherical harmonics.
"""


# we'll cache the splines
_realm_splines = {}
_imalm_splines = {}


def kerr_alm(spin, l, m, n):
    """Returns the angular separation constant for a Kerr BH.

    Parmeters
    ---------
    spin : float
        The dimensionless spin. Must be in [-0.9999, 0.9999].
    l : int
        The l index.
    m : int
        The m index.
    n : int
        The overtone number (where n=0 is the fundamental mode).

    Returns
    -------
    complex :
        The complex separation constant.
    """
    _checkspin(spin)
    respline = _getspline('alm', 're', l, m, n, _realm_splines)
    imspline = _getspline('alm', 'im', l, m, n, _imalm_splines)
    # if m is 0, use the absolute value of the spin
    if m == 0:
        spin = abs(spin)
    alm = respline(spin) + 1j * imspline(spin)
    # conjugate if -m (see Eq. 2.7 of Berti et al., arXiv:0512160)
    if m < 0:
        alm = alm.conj()
    return alm 


def _alpha(jj, k1):
    """Leaver's alpha coefficient for the Slm (Eq. 20 of Leaver).

    Note: we use jj in place of Leaver's n so as not to confuse with the
    overtone index.
    """
    return -2 * (jj + 1) * (jj + 2*k1 + 1)

def _beta(jj, k1, k2, aw, almterm):
    """Leaver's beta coefficient for the Slm (Eq. 20 of Leaver).

    Note: we use jj in place of Leaver's n so as not to confuse with the
    overtone index.
    """
    return jj*(jj - 1) + 2*jj*(k1 + k2 + 1 - 2*aw) - almterm

def _gamma(jj, k1, k2, aw, s):
    """Leaver's gamma coefficient for the Slm (Eq. 20 of Leaver).

    Note: we use jj in place of Leaver's n so as not to confuse with the
    overtone index.
    """
    return 2 * aw * (jj + k1 + k2 + s)

def spheroidal(theta, spin, l, m, n, s=-2, phi=0., tol=1e-8,
               max_recursion=1000, debug=False):
    """Calculate the spin-weighted spheroidal harmonic.

    See Eq. 18 of E. W. Leaver (1985)
    [`doi:10.1098/rspa.1985.0119` <https://doi.org/10.1098/rspa.1985.0119>].
    Solutions for the angular separation constants :math:`A_{\ell m}` are
    from Berti, Cardoso, & Starinets CQG26, 163001 (2009) [arXiv:0905.2975].

    Parameters
    ----------
    theta : float or array
        The polar angle of the observer with respect to the black hole's
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
    phi : float or array, optional
        The azimuthal angle of the observer. Default is 0.

    Returns
    -------
    complex :
        The value of the spheroidal harmonic.
    """
    u = numpy.cos(theta)
    alm = kerr_alm(spin, l, m, n)
    # prefactors
    omega = kerr_omega(spin, l, m, n)
    aw = spin * omega
    # Note: Leaver used the convention 2M = 1, meaning that our spin should
    # be halved and omega doubled. However, in everything below, we only use
    # spin*omega, so we don't need to worry about it.
    k1 = 0.5 * abs(m - s)
    k2 = 0.5 * abs(m + s)
    # last two terms in beta (Leaver Eq. 20)
    almterm = 2 * aw * (2 * k1 + s + 1) - (k1 + k2)*(k1 + k2 + 1) \
              + aw**2 + s*(s+1) + alm
    # Note: we use jj for Leaver's n in Eqs. 18-20 so as not to confuse with
    # the overtone index
    jj = 1
    b_n = 1 + u
    # initial a; this sets the norm of the Slm
    a0 = 1.
    # initialize terms for sum, which will start from jj = 1
    slm = a0 + 0j
    # Eq. 19 of Leaver
    alpha0 = _alpha(0, k1)
    beta0 = _beta(0,  k1, k2, aw, almterm)
    a_nm1 = a0
    a_n = -a0 * beta0/alpha0  # a1
    # we sum until the difference in terms is ~ 0
    delta = numpy.inf
    while abs(delta) > tol and jj < max_recursion:
        delta = a_n * b_n**jj
        slm += delta 
        # update for next loop
        alpha_n = _alpha(jj, k1)
        beta_n = _beta(jj,  k1, k2, aw, almterm)
        gamma_n = _gamma(jj, k1, k2, aw, s) 
        # Eq. 19 of Leaver:
        a_np1 = -(beta_n * a_n + gamma_n * a_nm1)/alpha_n
        a_nm1 = a_n
        a_n = a_np1
        jj += 1
    if jj == max_recursion and delta > tol:
        raise ValueError("maximum recursion exceeded; current delta is "
                         "{}".format(delta))
    # norm (Eq. 18 of Leaver)
    slmnorm = numpy.exp(aw*u) * (1 + u)**k1 *  (1 - u)**k2
    slm *= slmnorm
    return slm
        

def lalspheroidal(inclination, spin, l, m, n=0, s=-2, azimuthal=0.):
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
    try:
        from lalsimulation import SimBlackHoleRingdownSpheroidalWaveFunction \
            as _spheroidal
    except ImportError:
        raise ImportError("lalsuite must be installed for spheroidal "
                          "harmonics")
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


def spherical(inclination, l, m, s=-2, azimuthal=0.):
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
        from lal import SpinWeightedSphericalHarmonic as _spherical
    except ImportError:
        raise ImportError("lalsuite must be installed to generate the "
                          "spherical harmonics")
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
