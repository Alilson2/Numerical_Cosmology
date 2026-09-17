# Union2.1 MCMC

Python implementation for cosmological parameter inference using the
Union2.1 supernova dataset and (MCMC), developed for the Numerical Cosmology  graduate course

The project implements the components required for a cosmological
analysis, including cosmological calculations, Bayesian inference,
Metropolis-Hastings sampling, statistical diagnostics, data handling,
visualization, and automated tests.

## Features

- **Cosmological calculations**  
  Computation of the Hubble parameter, comoving distance, luminosity distance,
  angular diameter distance, distance modulus, conformal time, cosmic time,
  and cosmological volume element.

- **Bayesian inference**  
  Implementation of prior, likelihood, and posterior probability functions
  for cosmological parameter estimation.

- **Metropolis-Hastings MCMC**  
  Sampling of the cosmological parameter space using the Metropolis-Hastings
  algorithm with support for multiple independent chains.

- **Covariance matrix treatment**  
  Cholesky decomposition and efficient evaluation of the covariance-weighted
  likelihood.

- **MCMC diagnostics**  
  Analysis of autocorrelation, integrated autocorrelation time, effective
  sample size, split-$\hat{R}$, power spectrum, and Dunkley convergence
  criteria.

- **Visualization**  
  Generation of trace plots, acceptance-rate plots, autocorrelation plots,
  power spectra, corner plots, convergence diagnostics, and Hubble diagrams.

- **Automated testing**  
  Unit tests for the cosmological calculations, Bayesian functions, MCMC
  implementation, and statistical diagnostics. Run with pytest 

- **Reproducible analysis**  
  Support for controlled random seeds and a main script that integrates the
  complete analysis pipeline.


## Project Structure

```text
bib.EP1/
│
├── pyproject.toml
│
├── union21_mcmc/
│   ├── __init__.py
│   ├── Bayes.py
│   ├── Cosmology.py
│   ├── data.py
│   ├── diagnostics.py
│   ├── MCMC.py
│   └── plots.py
│
├── tests/
│   ├── test_bayes.py
│   ├── test_cosmology.py
│   ├── test_diagnostics.py
│   └── test_MCMC.py
│
├── SCPUnion2.1_mu_vs_z.txt
├── SCPUnion2.1_covmat_nosys.txt
├── SCPUnion2.1_covmat_sys.txt
├── SCPUnion2.1_AIISNe.tex
│
└── Example_Notebook.ipynb



