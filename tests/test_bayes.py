import numpy as np

from union21_mcmc.Bayes import (
    log_likelihood,
    log_prior,
    log_posterior,
    log_pdf,
    cholesky_covariance
)

from union21_mcmc.Cosmology import distance_modulus


# ============================================================
# Fiducial cosmology
# ============================================================

H_0 = 70.0

OM_M = 0.3
OM_DE = 0.7

theta = np.array([
    OM_M,
    OM_DE
])

z = np.linspace(
    0.01,
    2.0,
    20
)


def generate_mu_obs():

    mu_obs = distance_modulus(
        z,
        H_0,
        OM_M,
        OM_DE
    )

    return mu_obs


# ============================================================
# Tests for log_prior
# ============================================================

# Test: Valid prior
def test_log_prior_valid():

    lp = log_prior(
        theta,
        z
    )

    assert np.isfinite(lp)

    assert lp == 0.0


# Test: Incorrect parameter dimension is rejected
def test_log_prior_invalid_dimension():

    theta_invalid = np.array([
        70.0,
        0.3,
        0.7
    ])

    lp = log_prior(
        theta_invalid,
        z
    )

    assert lp == -np.inf


# Test: Negative Omega_M is rejected
def test_log_prior_invalid_OM_M():

    theta_invalid = theta.copy()

    theta_invalid[0] = -0.1

    lp = log_prior(
        theta_invalid,
        z
    )

    assert lp == -np.inf


# Test: Omega_M above prior range is rejected
def test_log_prior_OM_M_above_range():

    theta_invalid = theta.copy()

    theta_invalid[0] = 2.1

    lp = log_prior(
        theta_invalid,
        z
    )

    assert lp == -np.inf


# Test: Omega_DE below prior range is rejected
def test_log_prior_invalid_OM_DE():

    theta_invalid = theta.copy()

    theta_invalid[1] = -1.1

    lp = log_prior(
        theta_invalid,
        z
    )

    assert lp == -np.inf


# Test: Omega_DE above prior range is rejected
def test_log_prior_OM_DE_above_range():

    theta_invalid = theta.copy()

    theta_invalid[1] = 3.1

    lp = log_prior(
        theta_invalid,
        z
    )

    assert lp == -np.inf


# Test: Invalid expansion history is rejected
def test_log_prior_invalid_expansion_history():

    theta_invalid = np.array([
        0.0,
        3.0
    ])

    z_high = np.linspace(
        0.01,
        10.0,
        1000
    )

    lp = log_prior(
        theta_invalid,
        z_high
    )

    assert lp == -np.inf


# ============================================================
# Tests for log_likelihood
# ============================================================

# Test: Likelihood of perfect model is finite
def test_log_likelihood_perfect_model():

    mu_obs = generate_mu_obs()

    C = np.eye(
        len(z)
    )

    L = cholesky_covariance(
        C
    )

    ll = log_likelihood(
        theta,
        L,
        z,
        mu_obs
    )

    assert np.isfinite(ll)


# Test: Perfect model gives marginalized chi-square = 0
def test_log_likelihood_perfect_model_value():

    mu_obs = generate_mu_obs()

    C = np.eye(
        len(z)
    )

    L = cholesky_covariance(
        C
    )

    ll = log_likelihood(
        theta,
        L,
        z,
        mu_obs
    )

    assert np.isclose(
        ll,
        0.0,
        atol=1e-12
    )


# Test: Invalid covariance matrix is rejected
def test_log_likelihood_invalid_covariance():

    C = np.zeros(
        (
            len(z),
            len(z)
        )
    )

    with np.testing.assert_raises(
        np.linalg.LinAlgError
    ):

        cholesky_covariance(
            C
        )


# Test: Non-positive-definite covariance is rejected
def test_log_likelihood_non_positive_definite_covariance():

    C = np.eye(
        len(z)
    )

    C[0, 0] = -1.0

    with np.testing.assert_raises(
        np.linalg.LinAlgError
    ):

        cholesky_covariance(
            C
        )


# Test: Incorrect model parameters change likelihood
def test_log_likelihood_changes_with_parameters():

    mu_obs = generate_mu_obs()

    C = np.eye(
        len(z)
    )

    L = cholesky_covariance(
        C
    )

    theta_wrong = theta.copy()

    theta_wrong[0] = 0.4

    ll_fiducial = log_likelihood(
        theta,
        L,
        z,
        mu_obs
    )

    ll_wrong = log_likelihood(
        theta_wrong,
        L,
        z,
        mu_obs
    )

    assert ll_wrong < ll_fiducial


# ============================================================
# Tests for log_posterior
# ============================================================

# Test: Posterior = prior + likelihood
def test_log_posterior_relation():

    mu_obs = generate_mu_obs()

    C = np.eye(
        len(z)
    )

    L = cholesky_covariance(
        C
    )

    lp = log_prior(
        theta,
        z
    )

    ll = log_likelihood(
        theta,
        L,
        z,
        mu_obs
    )

    posterior = log_posterior(
        theta,
        L,
        z,
        mu_obs
    )

    assert np.isclose(
        posterior,
        lp + ll
    )


# Test: Invalid prior gives invalid posterior
def test_log_posterior_invalid_prior():

    mu_obs = generate_mu_obs()

    C = np.eye(
        len(z)
    )

    L = cholesky_covariance(
        C
    )

    theta_invalid = theta.copy()

    theta_invalid[0] = -1.0

    posterior = log_posterior(
        theta_invalid,
        L,
        z,
        mu_obs
    )

    assert posterior == -np.inf


# Test: Invalid likelihood gives invalid posterior
def test_log_posterior_invalid_likelihood():

    mu_obs = generate_mu_obs()

    C = np.eye(
        len(z)
    )

    L = cholesky_covariance(
        C
    )

    mu_obs_invalid = mu_obs.copy()

    mu_obs_invalid[0] = np.nan

    posterior = log_posterior(
        theta,
        L,
        z,
        mu_obs_invalid
    )

    assert posterior == -np.inf


# ============================================================
# Tests for log_pdf
# ============================================================

# Test: log_pdf for equal states
def test_log_pdf_equal_states():

    proposal_cov = np.diag([
        0.01**2,    # Omega_M
        0.01**2     # Omega_DE
    ])

    log_q = log_pdf(
        theta,
        theta,
        proposal_cov
    )

    ndim = len(theta)

    L = np.linalg.cholesky(
        proposal_cov
    )

    log_det_cov = (
        2.0
        * np.sum(
            np.log(
                np.diag(L)
            )
        )
    )

    expected = (
        -0.5 * ndim * np.log(2.0 * np.pi)
        -0.5 * log_det_cov
    )

    assert np.isclose(
        log_q,
        expected
    )


# Test: log_pdf is symmetric for opposite displacements
def test_log_pdf_symmetry():

    proposal_cov = np.diag([
        0.01**2,    # Omega_M
        0.01**2     # Omega_DE
    ])

    theta_new = (
        theta
        + np.array([
            0.01,
            0.01
        ])
    )

    theta_old = theta

    log_q_forward = log_pdf(
        theta_new,
        theta_old,
        proposal_cov
    )

    log_q_backward = log_pdf(
        theta_old,
        theta_new,
        proposal_cov
    )

    assert np.isclose(
        log_q_forward,
        log_q_backward
    )


# Test: log_pdf is finite for valid covariance
def test_log_pdf_is_finite():

    proposal_cov = np.diag([
        0.01**2,    # Omega_M
        0.01**2     # Omega_DE
    ])

    theta_new = (
        theta
        + np.array([
            0.005,
            0.005
        ])
    )

    log_q = log_pdf(
        theta_new,
        theta,
        proposal_cov
    )

    assert np.isfinite(
        log_q
    )


# Test: log_pdf rejects incorrect parameter dimension
def test_log_pdf_invalid_dimension():

    theta_new = np.array([
        0.3,
        0.7,
        0.0
    ])

    proposal_cov = np.diag([
        0.01**2,
        0.01**2
    ])

    with np.testing.assert_raises(
        ValueError
    ):

        log_pdf(
            theta_new,
            theta,
            proposal_cov
        )


# Test: log_pdf rejects incorrect covariance shape
def test_log_pdf_invalid_covariance_shape():

    proposal_cov = np.eye(
        3
    )

    with np.testing.assert_raises(
        ValueError
    ):

        log_pdf(
            theta,
            theta,
            proposal_cov
        )