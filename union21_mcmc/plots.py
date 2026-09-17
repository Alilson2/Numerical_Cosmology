import numpy as np
import matplotlib.pyplot as plt

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False


def plot_chains(
    chains,
    parameter_names,
    save_path=None
):
    """
    Plot MCMC chains for all parameters.

    Parameters
    ----------
    chains : numpy.ndarray
        MCMC chains with shape
        (n_chains, n_samples, n_parameters).

    parameter_names : list
        Names of the cosmological parameters.

    save_path : str, optional
        Path where the figure will be saved.
    """

    chains = np.asarray(chains, dtype=float)

    if chains.ndim != 3:
        raise ValueError(
            "chains must have shape "
            "(n_chains, n_samples, n_parameters)."
        )

    n_chains, n_samples, n_parameters = chains.shape

    if len(parameter_names) != n_parameters:
        raise ValueError(
            "The number of parameter names must match "
            "the number of parameters in chains."
        )

    fig, axes = plt.subplots(
        n_parameters,
        1,
        figsize=(10, 2.5 * n_parameters),
        sharex=True
    )

    if n_parameters == 1:
        axes = [axes]

    for i, parameter_name in enumerate(parameter_names):

        for chain in range(n_chains):

            axes[i].plot(
                np.arange(n_samples),
                chains[chain, :, i],
                alpha=0.5,
                linewidth=0.6
            )

        axes[i].set_ylabel(
            parameter_name
        )

        axes[i].grid(
            alpha=0.2
        )

    axes[-1].set_xlabel(
        "MCMC step"
    )

    fig.suptitle(
        "MCMC chains",
        fontsize=14
    )

    fig.tight_layout()

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.close(fig)


def plot_traces(
    chains,
    parameter_names,
    burn_in=0
):
    """
    Plot the trace of each parameter for all MCMC chains.

    Parameters
    ----------
    chains : ndarray
        MCMC chains with shape
        (n_chains, n_samples, n_parameters).

    parameter_names : list
        Names of the parameters.

    burn_in : int, optional
        Number of initial samples to mark as burn-in.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the trace plots.
    """

    chains = np.asarray(
        chains,
        dtype=float
    )

    if chains.ndim != 3:
        raise ValueError(
            "chains must have shape "
            "(n_chains, n_samples, n_parameters)."
        )

    n_chains, n_samples, n_parameters = (
        chains.shape
    )

    if len(parameter_names) != n_parameters:
        raise ValueError(
            "The number of parameter names must match "
            "the number of parameters in chains."
        )

    if burn_in < 0 or burn_in >= n_samples:
        raise ValueError(
            "burn_in must satisfy "
            "0 <= burn_in < n_samples."
        )

    fig, axes = plt.subplots(
        n_parameters,
        1,
        figsize=(12, 2.5 * n_parameters),
        sharex=True
    )

    if n_parameters == 1:
        axes = [axes]

    for j in range(n_parameters):

        for i in range(n_chains):

            axes[j].plot(
                chains[i, :, j],
                lw=0.5,
                alpha=0.5,
                label=(
                    f"Chain {i + 1}"
                    if j == 0
                    else None
                )
            )

        if burn_in > 0:

            axes[j].axvline(
                burn_in,
                linestyle="--",
                label=(
                    "Burn-in"
                    if j == 0
                    else None
                )
            )

        axes[j].set_ylabel(
            parameter_names[j]
        )

        axes[j].grid(
            alpha=0.2
        )

    axes[0].legend()

    axes[-1].set_xlabel(
        "Step"
    )

    fig.tight_layout()

    return fig


def plot_acceptance_rates(
    acceptance_rates
):
    """
    Plot the acceptance rate of each MCMC chain.

    Parameters
    ----------
    acceptance_rates : array-like
        Acceptance rate for each chain.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Acceptance rate figure.
    """

    acceptance_rates = np.asarray(
        acceptance_rates,
        dtype=float
    )

    if acceptance_rates.ndim != 1:
        raise ValueError(
            "acceptance_rates must be a 1D array."
        )

    if np.any(
        (acceptance_rates < 0.0)
        | (acceptance_rates > 1.0)
    ):
        raise ValueError(
            "Acceptance rates must be between 0 and 1."
        )

    n_chains = len(
        acceptance_rates
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        np.arange(1, n_chains + 1),
        acceptance_rates
    )

    ax.axhline(
        0.20,
        linestyle="--",
        label="20%"
    )

    ax.axhline(
        0.40,
        linestyle="--",
        label="40%"
    )

    ax.set_xlabel(
        "Chain"
    )

    ax.set_ylabel(
        "Acceptance rate"
    )

    ax.set_xticks(
        np.arange(1, n_chains + 1)
    )

    ax.legend()

    fig.tight_layout()

    return fig


def plot_autocorrelation(
    acf,
    parameter_name=None
):
    """
    Plot the autocorrelation function.

    Parameters
    ----------
    acf : array-like
        Autocorrelation function.

    parameter_name : str, optional
        Name of the parameter.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Autocorrelation figure.
    """

    acf = np.asarray(
        acf,
        dtype=float
    )

    if acf.ndim != 1:
        raise ValueError(
            "acf must be a 1D array."
        )

    lags = np.arange(
        len(acf)
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        lags,
        acf
    )

    ax.axhline(
        0.0,
        linestyle=":"
    )

    ax.set_xlabel(
        "Lag"
    )

    ax.set_ylabel(
        "Autocorrelation"
    )

    if parameter_name is not None:

        ax.set_title(
            f"Autocorrelation of {parameter_name}"
        )

    fig.tight_layout()

    return fig


def plot_power_spectrum(
    frequency,
    power,
    parameter_name=None
):
    """
    Plot the power spectrum of an MCMC chain.

    Parameters
    ----------
    frequency : array-like
        Fourier frequencies.

    power : array-like
        Power spectrum.

    parameter_name : str, optional
        Name of the parameter.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Power spectrum figure.
    """

    frequency = np.asarray(
        frequency,
        dtype=float
    )

    power = np.asarray(
        power,
        dtype=float
    )

    if frequency.ndim != 1 or power.ndim != 1:
        raise ValueError(
            "frequency and power must be 1D arrays."
        )

    if len(frequency) != len(power):
        raise ValueError(
            "frequency and power must have the same length."
        )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.loglog(
        frequency,
        power
    )

    ax.set_xlabel(
        "Frequency"
    )

    ax.set_ylabel(
        r"$P(f)$"
    )

    if parameter_name is not None:

        ax.set_title(
            f"Power spectrum of {parameter_name}"
        )

    fig.tight_layout()

    return fig


def plot_diagnostics(
    values,
    parameter_names,
    ylabel,
    title=None,
    reference=None
):
    """
    Plot a diagnostic value for each cosmological parameter.

    Parameters
    ----------
    values : array-like
        Diagnostic values.

    parameter_names : list
        Names of the parameters.

    ylabel : str
        Label for the y-axis.

    title : str, optional
        Figure title.

    reference : float, optional
        Reference value to display as a horizontal line.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Diagnostic figure.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be a 1D array."
        )

    if len(values) != len(parameter_names):
        raise ValueError(
            "The number of diagnostic values must match "
            "the number of parameter names."
        )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        np.arange(len(values)),
        values
    )

    if reference is not None:

        ax.axhline(
            reference,
            linestyle="--",
            label=f"Reference = {reference}"
        )

        ax.legend()

    ax.set_xticks(
        np.arange(len(values))
    )

    ax.set_xticklabels(
        parameter_names
    )

    ax.set_ylabel(
        ylabel
    )

    if title is not None:

        ax.set_title(
            title
        )

    fig.tight_layout()

    return fig

def plot_corner(samples: np.ndarray, weights: np.ndarray = None, param_names: list[str] = None, save_path: str = None) -> plt.Figure:
    """
    Generate a corner plot showing the posterior distributions of the
    cosmological parameters.

    The corner plot contains:
        - 1D marginalized posterior distributions along the diagonal;
        - 2D joint posterior distributions below the diagonal;
        - 68% and 95% credible contours in the 2D distributions;
        - 16%, 50%, and 84% quantiles for the 1D distributions.


    """
    
    if not HAS_CORNER:
        raise ImportError("'corner' is necessary. You may install running: pip install corner")

    fig = corner.corner(
        samples,
        weights         = weights,
        labels          = param_names,
        levels          = ( 0.68, 0.95 ),            
        quantiles       = [ 0.16, 0.50, 0.84 ],   
        show_titles     = True,
        title_fmt       = ".3f",
        plot_datapoints = False,          
        fill_contours   = True,             
        smooth          = 1.0
    )

    if save_path:
        fig.savefig( save_path, bbox_inches = "tight", dpi = 300 )

    return fig


def plot_hubble_diagram(
    z_obs: np.ndarray,
    mu_obs: np.ndarray,
    sigma_mu: np.ndarray,
    model_mu: np.ndarray,
    save_path: str = None
) -> plt.Figure:

    """
    Generates a Hubble diagram for the Union2.1 supernova sample
    together with the best-fit cosmological model and residuals.

    Parameters
    ----------
    z_obs : np.ndarray
        Observed redshifts of the supernovae.

    mu_obs : np.ndarray
        Observed distance moduli.

    sigma_mu : np.ndarray
        Individual uncertainties on the observed distance moduli.
        These uncertainties are used only for visualization.

    model_mu : np.ndarray
        Theoretical distance modulus evaluated at z_obs.

    save_path : str, optional
        Path where the figure will be saved.

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the Hubble diagram and residuals.
    """

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Sort the data by redshift
    sort_idx = np.argsort(z_obs)
    z_sorted = z_obs[sort_idx]
    mu_obs_sorted = mu_obs[sort_idx]
    sigma_sorted = sigma_mu[sort_idx]
    model_sorted = model_mu[sort_idx]

    # Hubble diagram
    ax1.errorbar(
        z_sorted,
        mu_obs_sorted,
        yerr=sigma_sorted,
        fmt="o",
        ms=3,
        alpha=0.5,
        label="Union2.1"
    )

    ax1.plot(
        z_sorted,
        model_sorted,
        lw=2,
        label="Posterior Mean Model"
    )

    ax1.set_ylabel(r"Distance Modulus $\mu(z)$")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    residuals = mu_obs_sorted - model_sorted

    ax2.errorbar(
        z_sorted,
        residuals,
        yerr=sigma_sorted,
        fmt="o",
        ms=3,
        alpha=0.5
    )

    ax2.axhline(
        0.0,
        linestyle="--",
        linewidth=1.5
    )

    ax2.set_xlabel(r"Redshift $z$")
    ax2.set_ylabel(r"$\Delta\mu$")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300
        )

    return fig