import numpy as np
from pykerr import qnm
from pykerr import (qnmfreq, qnmtau, spheroidal)


def test_qnmomega_callable():
    """Tests that qnmomega returns a complex number for the 2,2,0 mode. This
    largely just tests that the HDF resource is correctly located and opened. 
    """
    omega = qnm.qnmomega(0.0, 2, 2, 0)
    # must be a complex number and not NaN
    assert isinstance(omega, complex)
    assert not np.isnan(omega.real)
    assert not np.isnan(omega.imag)


def test_qnmfreq_value():
    """Test that qnmfreq returns the expected frequency for a specific case.
    This test uses the example from the README.md documentation.
    """
    freq = qnmfreq(200., 0.7, 2, 2, 0)
    expected = 86.04823229677822
    np.testing.assert_allclose(freq, expected, rtol=1e-7)


def test_qnmtau_value():
    """Test that qnmtau returns the expected damping time for a specific case.
    This test uses the example from the README.md documentation.
    """
    tau = qnmtau(200., 0.7, 2, 2, 0)
    expected = 0.012192884850631896 
    np.testing.assert_allclose(tau, expected, rtol=1e-7)


def test_spheroidal_value():
    """Test that spheroidal returns the expected complex value for a specific case.
    This test uses the example from the README.md documentation.
    """
    theta = np.pi/3  # polar angle
    a = 0.7         # dimensionless spin
    l, m, n = 2, 2, 0
    phi = 0.0       # azimuthal angle
    
    value = spheroidal(theta, a, l, m, n, phi=phi)
    expected = 0.3378406925922286-0.0007291958007333236j
    np.testing.assert_allclose(value, expected, rtol=1e-7)
