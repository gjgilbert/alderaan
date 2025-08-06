# ALDERAAN
**ALDERAAN** is a pipeline for **A**utomated **L**ightcurve **D**etrending, **E**xoplanet **R**ecovery, and **A**nalysis of **A**utocorrelated **N**oise.

The pipeline is currently capable of processing photometric lightcurve data from the *Kepler* Space Telescope, but in the future will be extended to handle data from *K2* and *TESS*.

Detrending and transit fitting are optimized for high-fidelity measurements of [P, t0, Rp/Rs, b, T14] and for inference of dynamical transit timing variations (TTVs). Autocorrelated noise arising from instrumental and astrophysical sources is handled using a combination of Gaussian Processes (GP) regression, Fourier analysis, and narrow bandstop filters. Sampling is performed either Dynamic Nested Sampling.

The core scientific dependencies for this software are  ``astropy``, ``batman``, ``celerite``, ``dynesty``, ``numpy``, ``PyMC3``, and ``scipy``.


# Installation instructions

ALDERAAN requires a complex set of dependencies in order to run. To create a conda environment capable of running the ALDERAAN pipeline, copy environment.yml to your local machine and run:

```
$ conda env create -n <ENV_NAME> -f environment.yml

```
If <ENV_NAME> is not specified, the conda environment will be named "alderaan".

You can then activate your environment and safely pip install the package:

```
$ conda activate alderaan-env
$ pip install alderaan
```




# Running the pipeline

To test running the pipeline, navigate into <LOCAL_DIR> and run the following commmand

```
$ python tests/test_transit_model.py
```

The test is hard-coded to use data from K00148. All the necessary data is in the directory /tests/testdata and /tests/catalogs. `test_transit_model.py` will autmoatically pull from these directories.
