import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

#Speed of light
c = 299792.458  # km/s


# Curvature density parameter
# Omega_K = 1 - Omega_M - Omega_DE

# No flatness condition is imposed. Omega_K is derived from Omega_M and Omega_DE.

def Omega_K(OM_M, OM_DE):

    return 1.0 - OM_M - OM_DE

# The cosmological constant case w = -1 is assumed.

# Hubble parameter H(z)
def H_Z(z,H_0,OM_M,OM_DE):

    z = np.asarray(z,dtype=float)
    OM_K = Omega_K(OM_M,OM_DE)
    E2 = (OM_M * (1.0 + z)**3 + OM_K * (1.0 + z)**2+ OM_DE)

    # Invalid cosmology
    if (
        np.any(E2 <= 0.0)
        or not np.all(np.isfinite(E2))
    ):
        return np.full_like(
            z,
            np.nan,
            dtype=float
        )

    return H_0 * np.sqrt(E2)

# E^2(z) = Omega_M (1+z)^3 + Omega_K (1+z)^2 + Omega_DE

# H(z) = H_0 E(z)

# Dimensionless Hubble parameter E(z)
def E_Z(z,H_0,OM_M,OM_DE):

    return (H_Z(z,H_0,OM_M,OM_DE)/ H_0)

# Conformal time
# eta(z) = - integral_z^zmax c dz' / H(z')
# Units: Mpc

def conformal_time(z,H_0,OM_M,OM_DE,zmax=1e4,ngrid=20000):

    z = np.atleast_1d(np.asarray(z,dtype=float))

    z_grid = np.logspace(
        np.log10(1.0 + np.min(z)),
        np.log10(1.0 + zmax),
        ngrid) - 1.0
    
    H_grid = H_Z(z_grid,H_0,OM_M,OM_DE)
    
    if not np.all(
        np.isfinite(H_grid)
    ):
        return np.full_like(
            z,
            np.nan
        )

    integrand = c / H_grid
    integral = cumulative_trapezoid(
        integrand[::-1],
        z_grid[::-1],
        initial=0.0
    )[::-1]

    eta_grid = -integral

    interpolation = interp1d(
        z_grid,
        eta_grid,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    return interpolation(z)

# Cosmic time
# t(z) = - integral_z^zmax c dz' / [(1+z') H(z')]
# Units: Mpc/c

def Cosmic_time(z,H_0,OM_M,OM_DE,zmax=1e4,ngrid=20000):

    z = np.atleast_1d(np.asarray(z,dtype=float))
    z_grid = np.logspace(
        np.log10(1.0 + np.min(z)),
        np.log10(1.0 + zmax),
        ngrid
    ) - 1.0

    H_grid = H_Z(z_grid,H_0,OM_M,OM_DE)

    if not np.all(
        np.isfinite(H_grid)
    ):
        return np.full_like(
            z,
            np.nan
        )

    integrand = c / ((1.0 + z_grid) * H_grid)

    integral = cumulative_trapezoid(
        integrand[::-1],
        z_grid[::-1],
        initial=0.0
    )[::-1]

    t_grid = -integral

    interpolation = interp1d(
        z_grid,
        t_grid,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    return interpolation(z)

# Comoving radial distance
# chi(z) = integral_0^z c dz' / H(z')
# Units: Mpc
def Comoving_distance(z,H_0,OM_M,OM_DE,ngrid=20000):

    z = np.atleast_1d(np.asarray(z,dtype=float))

    z_grid = np.linspace(0.0,np.max(z),ngrid)
    H_grid = H_Z(z_grid,H_0,OM_M,OM_DE)

    if not np.all(
        np.isfinite(H_grid)
    ):
        return np.full_like(
            z,
            np.nan
        )

    integrand = c / H_grid

    chi_grid = cumulative_trapezoid(
        integrand,
        z_grid,
        initial=0.0
    )

    interpolation = interp1d(
        z_grid,
        chi_grid,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    return interpolation(z)

# Transverse comoving distance
# Units: Mpc
def Distance_M(z,H_0,OM_M,OM_DE):

    z = np.asarray(z,dtype=float)
    chi = Comoving_distance(z,H_0,OM_M,OM_DE)

    if not np.all(
        np.isfinite(chi)
    ):
        return np.full_like(
            z,
            np.nan
        )

    OM_K = Omega_K(OM_M,OM_DE)
    D_H = c / H_0

    if np.isclose(
        OM_K,
        0.0
    ):

        D_M = chi

    elif OM_K > 0.0:

        sqrt_OK = np.sqrt(OM_K)
        D_M = (D_H/ sqrt_OK * np.sinh(sqrt_OK * chi / D_H))

    else:

        sqrt_abs_OK = np.sqrt(abs(OM_K))
        D_M = (D_H / sqrt_abs_OK * np.sin(sqrt_abs_OK * chi / D_H))

    return D_M

# Angular diameter distance
# D_A = D_M / (1+z)
# Units: Mpc
def Distance_Angular(z,H_0,OM_M,OM_DE):

    z = np.asarray(z,dtype=float)
    D_M = Distance_M(z,H_0,OM_M,OM_DE)

    return D_M / (1.0 + z)

# Luminosity distance
# D_L = (1+z)^2 D_A or equivalently D_L = (1+z) D_M
# Units: Mpc
def Distance_luminosity(z,H_0,OM_M,OM_DE):
    z = np.asarray(z,dtype=float)
    D_M = Distance_M(z,H_0,OM_M,OM_DE)
    return (1.0 + z) * D_M

# Distance modulus
# mu = 5 log10(D_L) + 25
# D_L is expressed in Mpc.

def distance_modulus(z,H_0,OM_M,OM_DE):
    z = np.asarray(z,dtype=float)
    D_L = Distance_luminosity(z,H_0,OM_M,OM_DE)

    if (
        np.any(D_L <= 0.0)
        or not np.all(
            np.isfinite(D_L)
        )
    ):
        return np.full_like(
            z,
            np.nan,
            dtype=float
        )

    return (5.0 * np.log10(D_L)+ 25.0)

# Comoving volume element
# dV / (dz dOmega) = c D_M^2 / H(z)
# Units: Mpc^3 sr^-1
def Volume_element(z,H_0,OM_M,OM_DE):
    z = np.asarray(z,dtype=float)

    D_M = Distance_M(z,H_0,OM_M,OM_DE)
    H = H_Z(z,H_0,OM_M,OM_DE)

    if (
        not np.all(
            np.isfinite(D_M)
        )
        or not np.all(
            np.isfinite(H)
        )
    ):
        return np.full_like(
            z,
            np.nan,
            dtype=float
        )

    return (c * D_M**2 / H)