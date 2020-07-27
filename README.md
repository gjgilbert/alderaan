# ALDERAAN
**ALDERAAN** is a pipeline for **A**utomatic **L**ightcurve **D**etrending, **E**xoplanet **R**ecovery, and **A**nalysis of **A**utocorrelated **N**oise.

The pipeline is currently capable of processing photometric lightcurve data from the *Kepler* Space Telescope, but in the future will be extended to handle data from *K2* and *TESS*.

Detrending and transit fitting are optimized to detect low-amplitude transit timing variations (TTVs) and transit duration variations (TDVs). Autocorrelated noise arising from both instrumental and astrophysical sources is handled using a combination of narrow bandstop filters and Gaussian Process regression.

This software is powered by ``exoplanet``, ``starry``, ``celerite``, ``PyMC3``, ``scipy``, and ``astropy``.
