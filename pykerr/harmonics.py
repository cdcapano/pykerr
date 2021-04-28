import logging
import numpy

from .qnm import (_getspline, _qnmomega, _checkspin)

"""Functions to calculate spheroidal and spherical harmonics.
"""

# the default tolerances we will use
TOL = 1e-6
MAXTOL = 0.01
MAX_RECURSION = 100


# to print out additional messages; set to True if you're having convergence
# issues and are trying to figure out why
DEBUG = False


# we'll cache the splines
_realm_splines = {}
_imalm_splines = {}
_norm_splines = {}
_normnm_splines = {}


def _checkspinweight(s):
    """Checks that the spin weight is a supported value."""
    if not s in [-2, -1, 0]:
        raise ValueError("s must be either -2, -1, or 0")
    if s != -2:
        raise NotImplementedError("only s=-2 is currently supported")


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


def _setdefault(arg, default):
    """Convenience function to set tolerance defaults."""
    return default if arg is None else arg


def slmnorm(spin, l, m, n, s=-2, npoints=1000, tol=None, maxtol=None,
            max_recursion=None, use_cache=True):
    r"""Calculate the normalization constant for a spheroidal harmonic.

    The normalization is such that:

    .. math::

        \int_{0}^{2\pi}\int_{0}^{\pi} |{}_{-s}S_{\ell m}(J,\theta,\phi)|^2
            \sin(\theta)\mathrm{d}\theta \mathrm{d}\phi = 1

    where :math:`J` is the spin of the black hole.

    The integral is calulcated using the trapezoidal rule.

    Parameters
    ----------
    spin : float
        The dimensionless spin of the black hole.
    l : int
        The l index. Up to m=7 is supported.
    m : int
        The m index. All +/-m for the given l are supported.
    n : int
        The overtone number (where n=0 is the fundamental mode). Up to n=7 is
        supported.
    s : int, optional
        The spin number. Must be either 0, -1, or -2. Default is -2.
    npoints : int, optional
        The number of points to use in the integral. Default is 1000.
    tol : float, optional
        Tolerance used to determine when to stop the sum over coefficients
        :math:`c_n = a_n (1 + \cos(theta))^n` in the spheroidal harmonics (see
        Eq. 18 of Leaver). The sum stops when :math:`|c_n|` is less than the
        given value. If ``None`` (the default) will use
        :const:`pykerr.harmonics.TOL`.
    maxtol : float, optional
        Maximum allowed error in the sum over coefficients in the spheroidal
        harmonics. If no :math:`c_n` up to :math:`n` = ``max_recursion``
        satisfies :math:`|c_n| <` ``tol``, then the sum will be done up to
        the smallest :math:`|c_n|` such that :math:`|c_n|` < ``maxtol``. If
        no :math:`|c_n|` is < ``maxtol``, then a ValueError will be raised.
        If set to ``None`` (the default), will use
        :const:`pykerr.harmonics.MAXTOL`.
    max_recursion : int, optional
        Maximum number of terms that will be used in the sum over coefficients
        in the spheroidal harmonics (see Eq. 18 of Leaver). If ``None`` (the
        default), will use :const:`pykerr.harmonics.MAX_RECURSION`.
    use_cache : bool, optional
        Use tabulated values in the cached data, if available. A cubic spline
        is used to interpolate to spins that are not in the cache. Default is
        True.
    """
    if use_cache:
        try:
            if m < 0:
                spline = _getspline('s{}nmnorm'.format(abs(s)), None, l, m, n,
                                    _normnm_splines)
            else:
                spline = _getspline('s{}norm'.format(abs(s)), None, l, m, n,
                                    _norm_splines)
            return spline(spin)
        except KeyError:
            pass
    # set the tolerance values
    tol = _setdefault(tol, TOL)
    maxtol = _setdefault(maxtol, MAXTOL)
    max_recursion = _setdefault(max_recursion, MAX_RECURSION)
    # evaluate
    thetas = numpy.linspace(0, numpy.pi, num=npoints)
    slm = spheroidal(thetas, spin, l, m, n, s=s, tol=tol, maxtol=maxtol,
                     max_recursion=max_recursion, normalize=False)
    return (2*numpy.pi*numpy.trapz((slm.conj()*slm).real*numpy.sin(thetas),
                                   dx=thetas[1]))**(-0.5)


def _pyslm(theta, spin, l, m, n, s=-2, phi=0., tol=None, maxtol=None,
           max_recursion=None, normalize=True, use_cache=True):
    r"""Calculate the spin-weighted spheroidal harmonic.

    See Eq. 18 of E. W. Leaver (1985)
    [`doi:10.1098/rspa.1985.0119` <https://doi.org/10.1098/rspa.1985.0119>].
    Solutions for the angular separation constants :math:`A_{\ell m}` are
    from Berti, Cardoso, & Starinets CQG26, 163001 (2009) [arXiv:0905.2975].

    Parameters
    ----------
    theta : float
        The polar angle of the observer with respect to the black hole's
        spin.
    spin : float
        The dimensionless spin of the black hole.
    l : int
        The l index. Up to l=7 is supported.
    m : int
        The m index. All +/-m for the given l are supported.
    n : int
        The overtone number (where n=0 is the fundamental mode). Up to n=7 is
        supported.
    s : int, optional
        The spin number. Must be either 0, -1, or -2. Default is -2.
    phi : float, optional
        The azimuthal angle of the observer. Default is 0.
    tol : float, optional
        Tolerance used to determine when to stop the sum over coefficients
        :math:`c_n = a_n (1 + \cos(theta))^n` in the spheroidal harmonics (see
        Eq. 18 of Leaver). The sum stops when :math:`|c_n|` is less than the
        given value. If ``None`` (the default) will use
        :const:`pykerr.harmonics.TOL`.
    maxtol : float, optional
        Maximum allowed error in the sum over coefficients in the spheroidal
        harmonics. If no :math:`c_n` up to :math:`n` = ``max_recursion``
        satisfies :math:`|c_n| <` ``tol``, then the sum will be done up to
        the smallest :math:`|c_n|` such that :math:`|c_n|` < ``maxtol``. If
        no :math:`|c_n|` is < ``maxtol``, then a ValueError will be raised.
        If set to ``None`` (the default), will use
        :const:`pykerr.harmonics.MAXTOL`.
    max_recursion : int, optional
        Maximum number of terms that will be used in the sum over coefficients
        in the spheroidal harmonics (see Eq. 18 of Leaver). If ``None`` (the
        default), will use :const:`pykerr.harmonics.MAX_RECURSION`.
    normalize : bool, optional
        Normalize the harmonic before returning. The normalization is such
        that the harmonic integrates to 1 over the unit sphere.
    use_cache : bool, optional
        If normalizing, use tabulated normalization constants in the cached
        data, if available. A cubic spline is used to interpolate to spins that
        are not in the cache. Otherwise, the normalization constant will be
        calculated on the fly by numerically integrating over theta. Default is
        True.

    Returns
    -------
    slm : complex
        The value of the spheroidal harmonic.
    """
    # set the tolerance values
    tol = _setdefault(tol, TOL)
    maxtol = _setdefault(maxtol, MAXTOL)
    max_recursion = _setdefault(max_recursion, MAX_RECURSION)
    # check spin weight
    _checkspinweight(s)
    u = numpy.cos(theta)
    alm = kerr_alm(spin, l, m, n)
    # prefactors
    omega = _qnmomega(spin, l, m, n)
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
    slm = numpy.zeros(max_recursion, dtype=complex)
    slm[0] = a0 + 0j
    # Eq. 19 of Leaver
    alpha0 = _alpha(0, k1)
    beta0 = _beta(0,  k1, k2, aw, almterm)
    a_nm1 = a0
    a_n = -a0 * beta0/alpha0  # a1
    # we sum until the difference between consecutive terms is ~ 0
    c_n = complex(numpy.inf, numpy.inf)
    while abs(c_n) > tol and jj < max_recursion:
        c_n = a_n * b_n**jj
        slm[jj] = c_n
        # update for next loop
        alpha_n = _alpha(jj, k1)
        beta_n = _beta(jj,  k1, k2, aw, almterm)
        gamma_n = _gamma(jj, k1, k2, aw, s) 
        # Eq. 19 of Leaver:
        a_np1 = -(beta_n * a_n + gamma_n * a_nm1)/alpha_n
        a_nm1 = a_n
        a_n = a_np1
        jj += 1
    if jj == max_recursion:
        # nothing satisfied the tolerance; try using the minimum
        minidx = abs(slm).argmin()
        c_n = slm[minidx]
        if abs(c_n) < maxtol:
            # ok, use that
            jj = minidx + 1
        if DEBUG:
            logging.debug("Did not get to target tolerance of {} for Slm. "
                          "Actual tolerance is {}. Parameters: theta={}, "
                          "spin={}, l={}, m={}, n={}, s={}, phi={}"
                          .format(tol, abs(c_n), theta, spin, l, m, n, s,
                                  phi))
    if abs(c_n) > maxtol:
        # were not able to satisfy the max condition, raise an error
        msg = ("Max recursion exceeded without satisfying maxtol. Smallest "
              "abs(term): {}. Parameters: theta={}, spin={}, l={}, m={}, "
              "n={}, s={}, phi={}."
              .format(abs(c_n), theta, spin, l, m, n, s, phi))
        if DEBUG:
            # print the slms
            msg += " Terms:\n{}".format('\n'.join(map(str, slm)))
        raise ValueError(msg)
    # sum up
    slm = slm[:jj].sum()
    # norm (Eq. 18 of Leaver)
    norm = numpy.exp(aw*u + 1j*m*phi) * (1 + u)**k1 *  (1 - u)**k2
    if normalize:
        norm *= slmnorm(spin, l, m, n, s=s, tol=tol, maxtol=maxtol,
                        max_recursion=max_recursion, use_cache=use_cache)
    slm *= norm
    # adjust to be consistent with the spherical harmonics... I'm not sure
    # why this works, it's just what I had to do to get it to match.
    slm *= (-1)**l * (-1)**max(0, s-m)
    return slm


# vectorize
_npslm = numpy.frompyfunc(_pyslm, 12, 1)


def spheroidal(theta, spin, l, m, n, s=-2, phi=0., tol=None, maxtol=None,
               max_recursion=None, normalize=True, use_cache=True):
    # done this way to vectorize the function
    out = _npslm(theta, spin, l, m, n, s, phi, tol, maxtol, max_recursion,
                 normalize, use_cache)
    if isinstance(out, numpy.ndarray):
        out = out.astype(numpy.complex)
    return out


spheroidal.__doc__ = _pyslm.__doc__


def spherical(theta, l, m, s=-2, phi=0.):
    """Calculate the spin-weighted spherical harmonic.

    Uses functions from lal.

    Parameters
    ----------
    theta : float or array
        The polar angle.
    l : int
        The l index.
    m : int
        The m index.
    s : int, optional
        The spin number. Default is -2.
    phi : float or array, optional
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
                          "spherical harmonics. Run `pip install lalsuite`.")
    try:
        return _spherical(theta, phi, s, l, m)
    except TypeError:
        # will get this if some of the inputs are arrays
        theta = numpy.atleast_1d(theta)
        phi = numpy.atleast_1d(phi)
        incphi = numpy.broadcast(theta, phi)
        return numpy.array([
            _spherical(inc, phi, s, l, m)
            for inc, phi in incphi]).reshape(incphi.shape)
