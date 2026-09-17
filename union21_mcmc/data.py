import numpy as np
from pathlib import Path


# ============================================================
# Data paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

UNION21_DATA_FILE = (
    BASE_DIR / "SCPUnion2.1_mu_vs_z.txt"
)

UNION21_COVARIANCE_FILE = (
    BASE_DIR / "SCPUnion2.1_covmat_sys.txt"
)


# ============================================================
# Load Union2.1 data
# ============================================================

def load_union21_data(
    filename=UNION21_DATA_FILE
):
    """
    Load the Union2.1 supernova data.

    Parameters
    ----------
    filename : str or pathlib.Path, optional
        Path to the Union2.1 data file.

    Returns
    -------
    z : numpy.ndarray
        Supernova redshifts.

    mu_obs : numpy.ndarray
        Observed distance moduli.
    """

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"Union2.1 data file not found: {filename}"
        )

    try:
        data = np.loadtxt(
            filename,
            comments="#",
            usecols=(1, 2)
        )

    except Exception as error:
        raise RuntimeError(
            "Could not load the Union2.1 "
            "supernova data."
        ) from error

    data = np.asarray(
        data,
        dtype=float
    )

    if data.ndim != 2:
        raise ValueError(
            "Union2.1 data must be a two-dimensional array."
        )

    if data.shape[1] != 2:
        raise ValueError(
            "Union2.1 data must contain redshift and "
            "distance modulus columns."
        )

    z = data[:, 0]
    mu_obs = data[:, 1]

    if not np.all(np.isfinite(z)):
        raise ValueError(
            "Redshift data contain non-finite values."
        )

    if not np.all(np.isfinite(mu_obs)):
        raise ValueError(
            "Distance modulus data contain non-finite values."
        )

    if np.any(z < 0):
        raise ValueError(
            "Redshift values must be non-negative."
        )

    if len(z) != len(mu_obs):
        raise ValueError(
            "Redshift and distance modulus arrays "
            "must have the same length."
        )

    return z, mu_obs


# ============================================================
# Load Union2.1 covariance matrix
# ============================================================

def load_union21_covariance(
    filename=UNION21_COVARIANCE_FILE
):
    """
    Load the Union2.1 covariance matrix.

    Parameters
    ----------
    filename : str or pathlib.Path, optional
        Path to the Union2.1 covariance matrix.

    Returns
    -------
    C : numpy.ndarray
        Union2.1 covariance matrix.
    """

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"Union2.1 covariance file not found: {filename}"
        )

    try:
        C = np.loadtxt(filename)

    except Exception as error:
        raise RuntimeError(
            "Could not load the Union2.1 "
            "covariance matrix."
        ) from error

    C = np.asarray(
        C,
        dtype=float
    )

    if C.ndim != 2:
        raise ValueError(
            "Covariance matrix must be two-dimensional."
        )

    if C.shape[0] != C.shape[1]:
        raise ValueError(
            "Covariance matrix must be square."
        )

    if not np.all(np.isfinite(C)):
        raise ValueError(
            "Covariance matrix contains non-finite values."
        )

    if not np.allclose(
        C,
        C.T,
        rtol=1e-10,
        atol=1e-12
    ):
        raise ValueError(
            "Covariance matrix must be symmetric."
        )

    return C