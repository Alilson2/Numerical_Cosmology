import numpy as np
import pytest
from union21_mcmc.MCMC import (run_chains,metropolis_hastings)
from union21_mcmc.Bayes import log_posterior
from union21_mcmc.Cosmology import distance_modulus

from union21_mcmc.diagnostics import (
    autocorrelation,
    integrated_autocorrelation_time,
    effective_sample_size,
    split_rhat
)

N_CHAINS = 5
OMEGA_M_INDEX = 0
OMEGA_DE_INDEX = 1
N_SAMPLES = 100000
BURN_IN = 10000
H_0 = 70.0
OM_M = 0.3
OM_DE = 0.7

z = np.linspace(0.01, 2.0, 20)

mu_obs = distance_modulus(
    z,
    H_0,
    OM_M,
    OM_DE,
)

C = np.eye(len(z))

L = np.linalg.cholesky(C)

ndim = 2

proposal_cov_initial = np.diag([
    0.05**2,
    0.05**2
])

theta_initials = np.array([
    [0.28, 0.72],
    [0.29, 0.71],
    [0.30, 0.70],
    [0.31, 0.69],
    [0.32, 0.68]
])

# Check configuration

if len(theta_initials) != N_CHAINS:
    raise ValueError(
        "The number of initial states must match N_CHAINS."
    )

# Pilot MCMC
print("\nRunning pilot MCMC...")

pilot_samples, pilot_acceptance = metropolis_hastings(
    log_posterior=log_posterior,
    initial_state=theta_initials[0],
    num_samples=30000,
    proposal_cov=proposal_cov_initial,
    C=C,
    z=z,
    mu_obs=mu_obs,
    rng=np.random.default_rng(123)
)

print(
    f"Pilot acceptance rate: "
    f"{pilot_acceptance:.4f}"
)

# Estimate posterior covariance
burn_in_pilot = 5000

pilot_samples_burned = pilot_samples[burn_in_pilot:]

pilot_cov = np.cov(
    pilot_samples_burned.T
)


# Optimal Gaussian proposal covariance
proposal_cov = (
    (2.38**2 / ndim)
    * pilot_cov
)


print("\nPilot covariance:")
print(pilot_cov)

print("\nFinal proposal covariance:")
print(proposal_cov)

# Run final MCMC
def generate_chains():

    chains, acceptance_rates = run_chains(
        log_posterior,
        theta_initials,
        N_SAMPLES,
        proposal_cov,
        L,
        z,
        mu_obs,
        seed=42
    )

    print("\nAcceptance rates:")

    for i, rate in enumerate(acceptance_rates):

        print(
            f"Chain {i + 1}: "
            f"{rate:.4f}"
        )

    print("\nChain means BEFORE burn-in:")

    for i in range(N_CHAINS):

        mean_m = np.mean(
            chains[i, :, OMEGA_M_INDEX]
        )

        mean_de = np.mean(
            chains[i, :, OMEGA_DE_INDEX]
        )

        print(
            f"Chain {i + 1}: "
            f"Omega_M = {mean_m:.6f}, "
            f"Omega_DE = {mean_de:.6f}"
        )

    print("\nChain means AFTER burn-in:")

    for i in range(N_CHAINS):

        mean_m = np.mean(
            chains[i, BURN_IN:, OMEGA_M_INDEX]
        )

        mean_de = np.mean(
            chains[i, BURN_IN:, OMEGA_DE_INDEX]
        )

        print(
            f"Chain {i + 1}: "
            f"Omega_M = {mean_m:.6f}, "
            f"Omega_DE = {mean_de:.6f}"
        )

    return chains


# ============================================================
# Shared MCMC fixture
# ============================================================

@pytest.fixture(scope="module")
def chains():

    return generate_chains()


# ============================================================
# Test 1: Omega_M R-hat
# ============================================================

def test_omega_m_rhat(chains):

    omega_m = chains[
        :,
        :,
        OMEGA_M_INDEX
    ]

    rhat = split_rhat(omega_m)

    assert np.isfinite(rhat)

    assert rhat < 1.01


# ============================================================
# Test 2: Omega_DE R-hat
# ============================================================

def test_omega_de_rhat(chains):

    omega_de = chains[
        :,
        :,
        OMEGA_DE_INDEX
    ]

    rhat = split_rhat(omega_de)

    assert np.isfinite(rhat)

    assert rhat < 1.01


# ============================================================
# Test 3: Omega_M autocorrelation
# ============================================================

def test_omega_m_autocorrelation(chains):

    omega_m = chains[
        :,
        :,
        OMEGA_M_INDEX
    ]

    for chain in omega_m:

        acf = autocorrelation(chain)

        assert np.isfinite(acf).all()

        assert np.isclose(
            acf[0],
            1.0
        )

        assert acf[1] < acf[0]


# ============================================================
# Test 4: Omega_DE autocorrelation
# ============================================================

def test_omega_de_autocorrelation(chains):

    omega_de = chains[
        :,
        :,
        OMEGA_DE_INDEX
    ]

    for chain in omega_de:

        acf = autocorrelation(chain)

        assert np.isfinite(acf).all()

        assert np.isclose(
            acf[0],
            1.0
        )

        assert acf[1] < acf[0]


# ============================================================
# Test 5: Omega_M integrated autocorrelation time
# ============================================================

def test_omega_m_integrated_autocorrelation_time(chains):

    omega_m = chains[
        :,
        :,
        OMEGA_M_INDEX
    ]

    omega_m = omega_m.reshape(-1)

    acf = autocorrelation(omega_m)

    tau = integrated_autocorrelation_time(acf)

    assert np.isfinite(tau)

    assert tau >= 1.0


# ============================================================
# Test 6: Omega_DE integrated autocorrelation time
# ============================================================

def test_omega_de_integrated_autocorrelation_time(chains):

    omega_de = chains[
        :,
        :,
        OMEGA_DE_INDEX
    ]

    omega_de = omega_de.reshape(-1)

    acf = autocorrelation(omega_de)

    tau = integrated_autocorrelation_time(acf)

    assert np.isfinite(tau)

    assert tau >= 1.0


# ============================================================
# Test 7: Omega_M effective sample size
# ============================================================

def test_omega_m_effective_sample_size(chains):

    omega_m = chains[
        :,
        BURN_IN:,
        OMEGA_M_INDEX
    ]

    ess_values = []

    for i, chain in enumerate(omega_m):

        acf = autocorrelation(chain)

        tau = integrated_autocorrelation_time(acf)

        ess = effective_sample_size(
            len(chain),
            tau
        )

        ess_values.append(ess)

        print(
            f"\nChain {i + 1}: "
            f"tau = {tau:.2f}, "
            f"ESS = {ess:.2f}"
        )

    ess_values = np.asarray(
        ess_values
    )

    assert np.isfinite(
        ess_values
    ).all()

    assert np.all(
        ess_values > 1000
    )


# ============================================================
# Test 8: Omega_DE effective sample size
# ============================================================

def test_omega_de_effective_sample_size(chains):

    omega_de = chains[
        :,
        BURN_IN:,
        OMEGA_DE_INDEX
    ]

    ess_values = []

    for chain in omega_de:

        acf = autocorrelation(chain)

        tau = integrated_autocorrelation_time(acf)

        ess = effective_sample_size(
            len(chain),
            tau
        )

        ess_values.append(ess)

    ess_values = np.asarray(
        ess_values
    )

    assert np.isfinite(
        ess_values
    ).all()

    assert np.all(
        ess_values > 1000
    )


# ============================================================
# Test 9: Omega_M chain means
# ============================================================

def test_omega_m_chain_means(chains):

    omega_m = chains[
        :,
        BURN_IN:,
        OMEGA_M_INDEX
    ]

    chain_means = np.mean(
        omega_m,
        axis=1
    )

    mean_difference = (
        np.max(chain_means)
        - np.min(chain_means)
    )

    assert mean_difference < 0.02


# ============================================================
# Test 10: Omega_DE chain means
# ============================================================

def test_omega_de_chain_means(chains):

    omega_de = chains[
        :,
        BURN_IN:,
        OMEGA_DE_INDEX
    ]

    chain_means = np.mean(
        omega_de,
        axis=1
    )

    mean_difference = (
        np.max(chain_means)
        - np.min(chain_means)
    )

    assert mean_difference < 0.02