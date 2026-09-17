import numpy as np

# Metropolis-Hastings
def metropolis_hastings(
    log_posterior,
    initial_state,
    num_samples,
    proposal_cov,
    C,
    z,
    mu_obs,
    rng=None
):
    """
    Run one Metropolis-Hastings chain.

    Parameters
    ----------
    log_posterior : callable
        Function that computes the log-posterior probability.

    initial_state : array
        Initial parameter vector.

    num_samples : int
        Number of samples in the chain.

    proposal_cov : ndarray
        Covariance matrix of the proposal distribution.

    C : ndarray
        Cholesky factor of the data covariance matrix.

    z : array
        Observed redshift values.

    mu_obs : array
        Observed distance moduli.

    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    samples : ndarray
        MCMC chain with shape

            (num_samples, ndim)

    acceptance_rate : float
        Fraction of accepted proposals.
    """

    # Create a random number generator if none was provided
    if rng is None:
        rng = np.random.default_rng()

    initial_state = np.asarray(initial_state,dtype=float)
    proposal_cov = np.asarray(proposal_cov,dtype=float)
    ndim = len(initial_state)

    if proposal_cov.shape != (ndim,ndim):
        raise ValueError(
            "proposal_cov must have shape "
            "(ndim, ndim).")

    # Allocate chain
    samples = np.zeros((num_samples, ndim),dtype=float)

    # Initial state
    current_state = initial_state.copy()

    # Initial posterior probability
    current_log_prob = log_posterior(current_state,C,z,mu_obs)
    accepted_count = 0

    # Cholesky decomposition of proposal covariance
    L_proposal = np.linalg.cholesky(proposal_cov)

    # MCMC loop
    for i in range(num_samples):

        # Generate a random Gaussian step
        random_step = rng.normal(size=ndim)

        # Generate proposal
        proposal = (current_state + L_proposal @ random_step)

        # Evaluate posterior at proposal
        proposal_log_prob = log_posterior(proposal,C,z,mu_obs)

        # Symmetric Gaussian proposal:
        # q(theta_new | theta_old)=q(theta_old | theta_new)
        # Therefore, the proposal terms cancel
        log_alpha = (proposal_log_prob - current_log_prob)

        # Accept or reject
        if np.log(rng.uniform()) < log_alpha:

            current_state = proposal
            current_log_prob = proposal_log_prob
            accepted_count += 1

        # Store current state
        samples[i] = current_state

    acceptance_rate = (accepted_count / num_samples)

    print(
        f"Acceptance Rate: "
        f"{acceptance_rate:.2%}"
    )

    return (samples,acceptance_rate)

# Multiple Metropolis-Hastings chains
def run_chains(
    log_posterior,
    initial_states,
    num_samples,
    proposal_cov,
    C,
    z,
    mu_obs,
    seed=None
):
    """
    Run multiple independent Metropolis-Hastings chains.

    Parameters
    ----------
    log_posterior : callable
        Function that computes the log-posterior probability.

    initial_states : array
        Initial parameter vectors for all chains.

        Shape:

            (n_chains, ndim)

    num_samples : int
        Number of samples per chain.

    proposal_cov : ndarray
        Covariance matrix of the proposal distribution.

    C : ndarray
        Cholesky factor of the data covariance matrix.

    z : array
        Observed redshift values.

    mu_obs : array
        Observed distance moduli.

    seed : int, optional
        Seed used to initialize the random number generators.

    Returns
    -------
    chains : ndarray
        MCMC chains with shape

            (n_chains, num_samples, ndim)

    acceptance_rates : ndarray
        Acceptance rate of each chain.
    """

    initial_states = np.asarray(initial_states,dtype=float)

    # Check initial-state dimensions
    if initial_states.ndim != 2:
        raise ValueError("initial_states must be a 2D array.")

    n_chains, ndim = initial_states.shape

    if proposal_cov.shape != (ndim,ndim):
        raise ValueError(
            "proposal_cov must have shape "
            "(ndim, ndim).")
        
    chains = np.zeros((n_chains,num_samples,ndim),dtype=float)

    # Allocate acceptance rates
    acceptance_rates = np.zeros(n_chains,dtype=float)

    # random number generator
    rng = np.random.default_rng(seed)

    # ========================================================
    # Run each chain independently
    # ========================================================

    for i in range(n_chains):

        print(
            f"Running chain "
            f"{i + 1}/{n_chains}"
        )

        # Generate an independent seed
        chain_seed = rng.integers(0,2**32 - 1)

        # Independent random number generator
        chain_rng = np.random.default_rng(chain_seed)

        # Run one chain
        samples, acceptance_rate = (
            metropolis_hastings(
                log_posterior=log_posterior,
                initial_state=initial_states[i],
                num_samples=num_samples,
                proposal_cov=proposal_cov,
                C=C,
                z=z,
                mu_obs=mu_obs,
                rng=chain_rng
            )
        )

        # Store 
        chains[i] = samples
        acceptance_rates[i] = (acceptance_rate)

    return (chains,acceptance_rates)