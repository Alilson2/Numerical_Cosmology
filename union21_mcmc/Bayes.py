import numpy as np
from scipy.linalg import solve_triangular
from .Cosmology import distance_modulus


# Fixed cosmological parameters
H_0 = 70.0

# Log-prior
def log_prior(theta, z):
    """
    Compute the log-prior for the cosmological parameter vector.

    Parameters
    ----------
    theta : array
        Parameter vector [Omega_M, Omega_DE].

    z : array
        Redshift values used to verify the validity of the
        cosmological model.

    Returns
    -------
    float
        0.0 if the parameters are allowed and the cosmological
        expansion history is valid, and -np.inf otherwise.
    """

    theta = np.asarray(theta,dtype=float)

    # Check parameter dimension
    if theta.shape != (2,):
        return -np.inf

    OM_M, OM_DE = theta

    # Prior limits
    if (OM_M < 0.0 or OM_M > 2.0):
        return -np.inf

    if (OM_DE < -1.0 or OM_DE > 3.0):
        return -np.inf

    OM_K = (1.0- OM_M - OM_DE)
    z = np.asarray(z,dtype=float)

    if not np.all(
        np.isfinite(z)
    ):
        return -np.inf

    E2 = (OM_M * (1.0 + z)**3 + OM_K * (1.0 + z)**2 + OM_DE)

    # Invalid cosmological expansion history
    if not np.all(
        np.isfinite(E2)
    ):
        return -np.inf

    if np.any(
        E2 <= 0.0
    ):
        return -np.inf

    return 0.0

# Log-likelihood
def log_likelihood(theta,L,z,mu_obs):
    """
    Compute the log-likelihood for the supernova data.

    The covariance matrix C must be previously factorized as

        C = L L^T

    where L is the lower triangular Cholesky factor.

    The absolute magnitude offset is analytically marginalized.

    Parameters
    ----------
    theta : array
        Parameter vector [Omega_M, Omega_DE].

    L : ndarray
        Lower triangular Cholesky factor of the covariance matrix.

    z : array
        Observed redshift values.

    mu_obs : array
        Observed distance moduli.

    Returns
    -------
    float
        Log-likelihood value.
    """

    theta = np.asarray(theta,dtype=float)

    # Check parameter dimension
    if theta.shape != (2,):
        return -np.inf

    OM_M, OM_DE = theta

    try:

        mu_th = distance_modulus(z,H_0,OM_M,OM_DE)

    except (
        ValueError,
        FloatingPointError
    ):

        return -np.inf

    if not np.all(
        np.isfinite(mu_th)
    ):
        return -np.inf

    mu_obs = np.asarray(mu_obs,dtype=float)
    r = (mu_obs- mu_th)

    if not np.all(np.isfinite(r)):
        return -np.inf

    ones = np.ones_like(r)

    try:
        r_t = solve_triangular(L,r,lower=True)
        ones_t = solve_triangular(L,ones,lower=True)

    except (ValueError,np.linalg.LinAlgError):
        return -np.inf

    A = r_t @ r_t
    B = ones_t @ r_t
    D = ones_t @ ones_t

    if (not np.isfinite(D) or D <= 0.0):
        return -np.inf

    chi2 = (A - B**2 / D)

    if not np.isfinite(chi2):
        return -np.inf

    return -0.5 * chi2

# Log-posterior
def log_posterior(theta,L,z,mu_obs):
    """
    Compute the log-posterior:

        log P(theta | data)
        = log_prior(theta)
        + log_likelihood(theta).

    Parameters
    ----------
    theta : array
        Parameter vector [Omega_M, Omega_DE].

    L : ndarray
        Lower triangular Cholesky factor of the covariance matrix.

    z : array
        Observed redshift values.

    mu_obs : array
        Observed distance moduli.

    Returns
    -------
    float
        Log-posterior value.
    """

    log_prior_value = log_prior(theta,z)

    if not np.isfinite(log_prior_value):
        return -np.inf

    log_likelihood_value = log_likelihood(theta,L,z,mu_obs)

    if not np.isfinite(log_likelihood_value):
        return -np.inf

    return (log_prior_value + log_likelihood_value)

# Cholesky decomposition
def cholesky_covariance(C):
    """
    Compute the Cholesky decomposition of the covariance matrix:

        C = L L^T

    This function should be called once before running the MCMC.

    Parameters
    ----------
    C : ndarray
        Covariance matrix.

    Returns
    -------
    L : ndarray
        Lower triangular Cholesky factor.
    """

    C = np.asarray(C,dtype=float)

    if C.ndim != 2:
        raise ValueError("Covariance matrix must be 2-dimensional.")

    if C.shape[0] != C.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    if not np.all(np.isfinite(C)):
        raise ValueError(
            "Covariance matrix contains "
            "NaN or inf values.")

    if not np.allclose(C,C.T):
        raise ValueError("Covariance matrix must be symmetric.")

    return np.linalg.cholesky(C)

# Gaussian proposal log-density
def log_pdf(theta_new,theta_old,proposal_cov):
    """
    Compute the log-probability density of a multivariate
    Gaussian proposal distribution.

    Parameters
    ----------
    theta_new : array
        Proposed parameter vector.

    theta_old : array
        Current parameter vector, used as the mean of the
        proposal distribution.

    proposal_cov : ndarray
        Covariance matrix of the proposal distribution.

    Returns
    -------
    float
        Log-probability density of theta_new given theta_old.
    """

    theta_new = np.asarray(theta_new,dtype=float)
    theta_old = np.asarray(theta_old,dtype=float)
    proposal_cov = np.asarray(proposal_cov,dtype=float)

    if theta_new.ndim != 1:
        raise ValueError("theta_new must be a one-dimensional array.")

    if theta_old.ndim != 1:
        raise ValueError("theta_old must be a one-dimensional array.")

    if theta_new.shape != theta_old.shape:
        raise ValueError(
            "theta_new and theta_old must have "
            "the same shape."
        )

    ndim = len(theta_old)

    if proposal_cov.shape != (ndim,ndim):
        raise ValueError(
            "proposal_cov must have shape "
            "(ndim, ndim).")

    L = np.linalg.cholesky(proposal_cov)
    delta = (theta_new - theta_old)
    delta_t = solve_triangular(L,delta,lower=True)
    quadratic = (delta_t @ delta_t)
    log_det_cov = (2.0 * np.sum(np.log(np.diag(L))))
    log_pdf_value = (
        -0.5 * ndim * np.log(2.0 * np.pi)
        -0.5 * log_det_cov
        -0.5 * quadratic)

    return log_pdf_value