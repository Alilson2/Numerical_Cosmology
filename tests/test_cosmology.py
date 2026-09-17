import numpy as np

from union21_mcmc.Cosmology import (
    H_Z,
    E_Z,
    Cosmic_time,
    conformal_time,
    Comoving_distance,
    Distance_M,
    Distance_Angular,
    Distance_luminosity,
    distance_modulus,
    Volume_element
)

# Fiducial cosmology

H_0 = 70.0
OM_M = 0.3
OM_DE = 0.7

z = np.linspace(0.01, 2.0, 1000)

# Test 1: H(0)
def test_H_at_zero():

    H_at_0 = H_Z(0.0,H_0,OM_M,OM_DE)

    assert np.isclose(H_at_0,H_0)

# Test 2: E(0)
def test_E_at_zero():

    E_at_0 = E_Z(0.0,H_0,OM_M,OM_DE)

    assert np.isclose(E_at_0,1.0)

# Test 3: E(z) = H(z) / H0
def test_E_relation():

    H = H_Z(z,H_0,OM_M,OM_DE)

    E = E_Z(z,H_0,OM_M,OM_DE)
    E_expected = H / H_0

    assert np.allclose(E,E_expected)


#Test 4: D_A = D_M / (1 + z)
def test_angular_distance_relation():

    D_M = Distance_M(z,H_0,OM_M,OM_DE)

    D_A = Distance_Angular(z,H_0,OM_M,OM_DE)

    D_A_expected = (D_M / (1.0 + z) )

    assert np.allclose(D_A,D_A_expected)

# Test 5: D_L = (1 + z)^2 D_A
def test_luminosity_angular_relation():

    D_A = Distance_Angular(z,H_0,OM_M,OM_DE)

    D_L = Distance_luminosity(z,H_0,OM_M,OM_DE)

    D_L_expected = (D_A * (1.0 + z)**2)

    assert np.allclose(D_L,D_L_expected)

# Test 6: D_L = (1 + z) D_M
def test_luminosity_comoving_relation():

    D_M = Distance_M(z,H_0,OM_M,OM_DE)

    D_L = Distance_luminosity(z,H_0,OM_M,OM_DE)
    D_L_expected = (D_M * (1.0 + z))

    assert np.allclose(D_L,D_L_expected)

# Test 7: Distance modulus
def test_distance_modulus():

    D_L = Distance_luminosity(z,H_0,OM_M,OM_DE)

    mu = distance_modulus(z,H_0,OM_M,OM_DE)

    mu_expected = (5.0 * np.log10(D_L)+ 25.0)
    assert np.allclose(mu,mu_expected)

# Test 8: dchi/dz = c/H(z)
def test_comoving_distance_derivative():

    H = H_Z(z,H_0,OM_M,OM_DE)

    chi = Comoving_distance(z,H_0,OM_M,OM_DE)

    dchi_dz_numeric = np.gradient(chi,z)
    dchi_dz_expected = (299792.458 / H)
    relative_error = np.abs((dchi_dz_numeric- dchi_dz_expected)/ dchi_dz_expected)
    max_error = np.max(relative_error[1:-1])
    assert max_error < 0.01

# Test 9: deta/dz
def test_conformal_time_derivative():

    H = H_Z(z,H_0,OM_M,OM_DE)
    eta = conformal_time(z,H_0,OM_M,OM_DE)
    deta_dz_numeric = np.gradient(eta,z)
    deta_dz_expected = (-299792.458 / H)

    relative_error = np.abs(
        (
            deta_dz_numeric
            - deta_dz_expected
        )
        / deta_dz_expected
    )

    max_error = np.max(relative_error[1:-1])

    assert max_error < 0.01

# Test 10: dt/dz
def test_cosmic_time_derivative():

    H = H_Z(z,H_0,OM_M,OM_DE)
    t = Cosmic_time(z,H_0,OM_M,OM_DE)
    dt_dz_numeric = np.gradient(t,z)
    dt_dz_expected = ( -299792.458 / ((1.0 + z) * H))
    relative_error = np.abs((dt_dz_numeric - dt_dz_expected) / dt_dz_expected)
    max_error = np.max(relative_error[1:-1])

    assert max_error < 0.01

# Test 11: dt/deta = a
def test_cosmic_conformal_relation():

    eta = conformal_time(z,H_0,OM_M,OM_DE)
    t = Cosmic_time(z,H_0,OM_M,OM_DE)
    deta_dz = np.gradient(eta,z)
    dt_dz = np.gradient(t,z)
    dt_deta_numeric = (dt_dz / deta_dz)
    a = 1.0 / (1.0 + z)

    assert np.allclose(
        dt_deta_numeric[1:-1],
        a[1:-1],
        rtol=0.01
    )

# Test 12: dV/(dz dOmega) = c D_M^2 / H(z)
def test_volume_element():

    H = H_Z(z,H_0,OM_M,OM_DE)
    D_M = Distance_M(z,H_0,OM_M,OM_DE)
    dV = Volume_element(z,H_0,OM_M,OM_DE)
    dV_expected = (299792.458 * D_M**2 / H)

    assert np.allclose(
        dV,
        dV_expected,
        rtol=1e-10,
        atol=0.0
    )

# Test 13: Distances are positive
def test_distances_are_positive():

    D_M = Distance_M(z,H_0,OM_M,OM_DE)
    D_A = Distance_Angular(z,H_0,OM_M,OM_DE)
    D_L = Distance_luminosity(z,H_0,OM_M,OM_DE)

    assert np.all(D_M > 0)

    assert np.all(D_A > 0)

    assert np.all(D_L > 0)


# Test 14: Distance modulus is finite
def test_distance_modulus_is_finite():

    mu = distance_modulus(z,H_0,OM_M,OM_DE)
    
    assert np.all(np.isfinite(mu))