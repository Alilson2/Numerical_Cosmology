import numpy as np


def autocorrelation(x, max_lag=None):
    """
    Compute the autocorrelation function of a time series.

    Parameters
    ----------
    x : array
        Input time series.

    max_lag : int, optional
        Maximum lag to compute. If None, half of the total
        number of samples is used.

    Returns
    -------
    acf : ndarray
        Autocorrelation function.
    """

    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array.")

    n = len(x)

    if n < 2:
        raise ValueError("x must contain at least two samples.")

    x = x - np.mean(x)

    if max_lag is None:
        max_lag = n // 2

    if max_lag < 0 or max_lag >= n:
        raise ValueError("max_lag must satisfy 0 <= max_lag < len(x).")

    acf = np.empty(max_lag + 1)

    # Compute the normalization factor
    variance = np.dot(x, x)

    if variance == 0.0:
        acf[:] = np.nan
        return acf

    # Autocorrelation at zero lag
    acf[0] = 1.0

    # Compute autocorrelation for each lag
    for lag in range(1, max_lag + 1):

        acf[lag] = (np.dot(x[:-lag], x[lag:]) / variance)

    return acf


def integrated_autocorrelation_time(acf):
    """
    Compute the integrated autocorrelation time.

    Parameters
    ----------
    acf : array
        Autocorrelation function.

    Returns
    -------
    tau : float
        Integrated autocorrelation time.
    """

    acf = np.asarray(acf, dtype=float)

    if acf.ndim != 1:
        raise ValueError("acf must be a one-dimensional array.")

    if len(acf) == 0:
        raise ValueError("acf must contain at least one value.")

    # Initialize the integrated autocorrelation time
    tau = 1.0

    # Sum until the first non-positive autocorrelation
    for rho in acf[1:]:

        if not np.isfinite(rho):
            break

        if rho <= 0.0:
            break

        tau += 2.0 * rho

    return tau


def effective_sample_size(n_samples, tau):
    """
    Compute the effective sample size.

    Parameters
    ----------
    n_samples : int
        Total number of samples.

    tau : float
        Integrated autocorrelation time.

    Returns
    -------
    ess : float
        Effective sample size.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive.")

    ess = n_samples / tau

    return ess


def split_rhat(chains):
    """
    Compute the split-Rhat convergence diagnostic.

    Parameters
    ----------
    chains : ndarray
        Array containing multiple chains for one parameter.

        Shape:

            (n_chains, n_samples)

    Returns
    -------
    Rhat : float
        Split-Rhat statistic.
    """

    chains = np.asarray(chains, dtype=float)

    if chains.ndim != 2:
        raise ValueError(
            "chains must have shape "
            "(n_chains, n_samples).")

    n_chains, n_samples = chains.shape

    if n_chains < 2:
        raise ValueError("At least two chains are required.")

    if n_samples < 4:
        raise ValueError("At least four samples per chain are required.")

    if not np.all(np.isfinite(chains)):
        return np.nan

    n_half = n_samples // 2

    split_chains = np.concatenate(
        [chains[:, :n_half],chains[:, -n_half:]],axis=0)

    n = split_chains.shape[1]

    # Mean of each split chain
    chain_means = np.mean(split_chains,axis=1)

    # Variance within each split chain
    chain_vars = np.var(split_chains,axis=1,ddof=1)
    W = np.mean(chain_vars)

    B = n * np.var(chain_means,ddof=1)

    # Estimated marginal posterior variance
    var_hat = (((n - 1.0) / n) * W + B / n)

    if W == 0.0:
        if B == 0.0:
            return 1.0
        return np.inf

    # Split-Rhat
    Rhat = np.sqrt(var_hat / W)

    return Rhat


def power_spectrum(x):
    """
    Compute the power spectrum of a time series.

    Parameters
    ----------
    x : array
        Input time series.

    Returns
    -------
    frequency : ndarray
        Fourier frequencies excluding zero frequency.

    P : ndarray
        Power spectrum excluding zero frequency.
    """

    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array.")

    N = len(x)

    if N < 2:
        raise ValueError("x must contain at least two samples.")

    x = x - np.mean(x)

    # Real-valued Fast Fourier Transform
    fft = np.fft.rfft(x)

    # Power spectrum
    P = np.abs(fft)**2 / N

    # Fourier frequencies
    frequency = np.fft.rfftfreq(N)

    return frequency[1:], P[1:]

def dunkley_power_spectrum(chain: np.ndarray) -> dict:
    """
    Power Spectrum convergence criteria from Dunkley et al. (2005).
    Returns P_0, j_*, alpha, r, and acceptance (r < 0.01).
    """
    
    N, D    = chain.shape
    results = {}
    
    for d in range( D ):
        x = chain[ :, d ] - np.mean( chain[ :, d ] )
       
        fft_vals = np.fft.rfft( x )
        P_j      = ( 1.0 / N ) * np.abs( fft_vals ) ** 2
        j_vals   = np.arange( len( P_j ) )
        
        P_0    = P_j[ 1 ] 
        j_star = N / 10.0 
        alpha  = 2.0
        
        r      = P_0 / ( N * np.var( x ) + 1e-15 )
        passed = r < 0.01
        
        results[ f"param_{d}" ] = {
            "P_0": float( P_0 ),
            "j_star": float( j_star ),
            "alpha": float( alpha ),
            "r": float( r ),
            "passed": bool( passed )
        }
    
    return results