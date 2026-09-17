import numpy as np
from union21_mcmc.MCMC import metropolis_hastings,run_chains
from union21_mcmc.Bayes import log_posterior

from union21_mcmc.Cosmology import (
    distance_modulus
)


# Fiducial cosmology

H_0 = 70.0
OM_M = 0.3
OM_DE = 0.7

# MCMC parameters:
theta = np.array([
    OM_M,
    OM_DE,
])

z = np.linspace(0.01,2.0,20)

# Synthetic observations
mu_obs = distance_modulus(
    z,
    H_0,
    OM_M,
    OM_DE,
)

C = np.eye(len(z))
proposal_cov = np.diag([
    0.01**2,    # Omega_M
    0.01**2     # Omega_DE
])

# Test 1: Metropolis-Hastings output dimensions=
def test_metropolis_hastings_shape():

    np.random.seed(42)

    n_samples = 100

    samples, acceptance_rate = metropolis_hastings(
        log_posterior,
        theta,
        n_samples,
        proposal_cov,
        C,
        z,
        mu_obs
    )

    assert samples.shape == (n_samples,2)

# Test 2: Acceptance rate is valid
def test_metropolis_hastings_acceptance_rate():

    np.random.seed(42)

    n_samples = 100

    samples, acceptance_rate = metropolis_hastings(
        log_posterior,
        theta,
        n_samples,
        proposal_cov,
        C,
        z,
        mu_obs
    )

    assert 0.0 <= acceptance_rate <= 1.0

# Test 3: Samples are finite
def test_metropolis_hastings_finite_samples():

    np.random.seed(42)

    n_samples = 100

    samples, acceptance_rate = metropolis_hastings(
        log_posterior,
        theta,
        n_samples,
        proposal_cov,
        C,
        z,
        mu_obs
    )

    assert np.all(np.isfinite(samples))

# Test 4: Number of chains
def test_run_chains_number_of_chains():

    np.random.seed(42)

    theta_initials = np.array([
        [0.20, 0.80],
        [0.30, 0.70],
        [0.40, 0.60],
        [0.25, 0.75],
        [0.35, 0.65]
    ])

    n_samples = 100

    chains, acceptance_rates = run_chains(
        log_posterior,
        theta_initials,
        n_samples,
        proposal_cov,
        C,
        z,
        mu_obs
    )

    assert chains.shape == (5,n_samples,2)

# Test 5: Acceptance rates for all chains
def test_run_chains_acceptance_rates_shape():

    np.random.seed(42)

    theta_initials = np.array([
        [0.20, 0.80],
        [0.30, 0.70],
        [0.40, 0.60],
        [0.25, 0.75],
        [0.35, 0.65]
    ])

    n_samples = 100

    chains, acceptance_rates = run_chains(
        log_posterior,
        theta_initials,
        n_samples,
        proposal_cov,
        C,
        z,
        mu_obs
    )

    assert acceptance_rates.shape == (5,)

# Test 6: All acceptance rates are valid
def test_run_chains_acceptance_rates_valid():

    np.random.seed(42)

    theta_initials = np.array([
        [0.20, 0.80],
        [0.30, 0.70],
        [0.40, 0.60],
        [0.25, 0.75],
        [0.35, 0.65]
    ])

    n_samples = 100

    chains, acceptance_rates = run_chains(
        log_posterior,
        theta_initials,
        n_samples,
        proposal_cov,
        C,
        z,
        mu_obs
    )

    assert np.all(
        acceptance_rates >= 0.0
    )

    assert np.all(
        acceptance_rates <= 1.0
    )

# Test 7: All chains contain finite values
def test_run_chains_finite():

    np.random.seed(42)

    theta_initials = np.array([
        [0.20, 0.80],
        [0.30, 0.70],
        [0.40, 0.60],
        [0.25, 0.75],
        [0.35, 0.65]
    ])

    n_samples = 100

    chains, acceptance_rates = run_chains(
        log_posterior,
        theta_initials,
        n_samples,
        proposal_cov,
        C,
        z,
        mu_obs
    )

    assert np.all(np.isfinite(chains))

# Test 8: Number of parameters is preserved
def test_run_chains_parameter_dimension():

    np.random.seed(42)

    theta_initials = np.array([
        [0.20, 0.80],
        [0.30, 0.70],
        [0.40, 0.60]
    ])

    n_samples = 50

    chains, acceptance_rates = run_chains(
        log_posterior,
        theta_initials,
        n_samples,
        proposal_cov,
        C,
        z,
        mu_obs
    )

    assert chains.shape == (3,n_samples,2)

    assert chains.shape[2] == 2