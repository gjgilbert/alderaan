# ALDERAAN
**ALDERAAN** is a pipeline for **A**utomated **L**ightcurve **D**etrending, **E**xoplanet **R**ecovery, and **A**nalysis of **A**utocorrelated **N**oise.

The pipeline is currently capable of processing photometric lightcurve data from the *Kepler* Space Telescope, but in the future will be extended to handle data from *K2* and *TESS*.

Detrending and transit fitting are optimized for high-fidelity measurements of transit parameters {$P, t_0, R_p/R_s, b, T_{14}$} and dynamical transit timing variations (TTVs). Autocorrelated noise arising from both instrumental and astrophysical sources is handled using a combination of Gaussian Processes (GP) regression, Fourier analysis, and narrow bandstop filters. Sampling is performed either Dynamic Nested Sampling.

The core scientific dependencies for this software are  ``astropy``, ``batman``, ``celerite``, ``dynesty``, ``numpy``, ``PyMC3``, and ``scipy``.


# Installation instructions

ALDERAAN requires a complex set of dependencies in order to run. To create a conda environment capable of running the ALDERAAN pipeline, follow the instructions below.

```
$ git clone https://github.com/gjgilbert/alderaan <LOCAL_DIR>
$ conda env create -n <ENV_NAME> -f <LOCAL_DIR>/environment.yml

if <ENV_NAME> is not specified, the conda environment will be named "alderaan"
```

# Running the pipeline

To test running the pipeline, navigate into <LOCAL_DIR> and run the following commmand

```
$ python tests/test_transit_model.py
```

The test is hard-coded to use data from K00148. All the necessary data is in the directory /tests/testdata and /tests/catalogs. `test_transit_model.py` will autmoatically pull from these directories.
