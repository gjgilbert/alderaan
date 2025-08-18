# ALDERAAN
**A**utomated **L**ightcurve **D**etrending, **E**xoplanet **R**ecovery, and **A**nalysis of **A**utocorrelated **N**oise

ALDERAAN is a fast and reliable pipeline for processing exoplanet transit photometry. The pipeline is currently capable of handling data from the *Kepler* Space Telescope, but in the near future will be extended to handle data from *K2* and *TESS*.

Detrending and transit fitting are optimized for high-fidelity measurements of [P, t0, Rp/Rs, b, T14] and for inference of dynamical transit timing variations (TTVs). Noise arising from instrumental and astrophysical sources is handled using a combination of Gaussian Processes (GP) regression and autocorrelated frequency analysis. Model sampling is performed using dynamic nested sampling.

For detailed documentation, see [readthedocs.org](https://alderaan.readthedocs.io/en/latest/)

## Installation instructions

ALDERAAN requires a complex set of dependencies in order to run. To create a conda environment capable of running the ALDERAAN pipeline, copy environment.yml from the `alderaan` github repository to your local machine:

```
curl -o ./environment.yml https://raw.githubusercontent.comefs/heads/develop/environment.yml
```

Then run:

```
conda env create -n <ENV_NAME> -f environment.yml
```

If <ENV_NAME> is not specified, the conda environment will be named "alderaan-env".

You can then activate your environment and safely pip install the package:

```
conda activate alderaan-env
```

then

```
pip install alderaan
```


## Running the pipeline

To run the pipeline, navigate into the `alderaan` source directory and run the following commmand:

```
python alderaan/recipes/main.rcp -m Kepler -t K00137 -c alderaan/configs/default_config.cfg 
```

The flags -m (mission) -t (target) and -c (config) are required and set the pipeline run conditions.


## Attribution
If you make use of `alderaan` in your work, please cite [Gilbert, Petigura, & Entrican (2025)](https://ui.adsabs.harvard.edu/abs/2025PNAS..12205295G/abstract).

The mathematical formalism and motivation behind `alderaan` transit modeling choices was developed in [Kipping (2013)](https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract), [Gilbert, MacDougall, & Petigura (2022)](https://ui.adsabs.harvard.edu/abs/2022AJ....164...92G/abstract), and [MacDougall, Gilbert, & Petigura (2023)](https://ui.adsabs.harvard.edu/abs/2023AJ....166...61M/abstract).

Foundational work on transit modeling which informed the development of `alderaan` is described in [Mandel & Agol (2002)](https://ui.adsabs.harvard.edu/abs/2002ApJ...580L.171M/abstract), [Seager & Mallén-Ornelas (2003)](https://ui.adsabs.harvard.edu/abs/2003ApJ...585.1038S/abstract), [Carter et al. (2008)](https://ui.adsabs.harvard.edu/abs/2008ApJ...689..499C/abstract), [Kipping (2010)](https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.1758K/abstract), [Eastman, Gaudi, & Agol (2013)](https://ui.adsabs.harvard.edu/abs/2013PASP..125...83E/abstract), and [Foreman-Mackey et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021JOSS....6.3285F/abstract).

The `alderaan` pipeline relies on the following core dependencies:
* `astropy` [Astropy Collaboration et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...935..167A/abstract)
* `batman` [Kreidberg (2015)](https://ui.adsabs.harvard.edu/abs/2015ascl.soft10002K/abstract)
* `celerite` [Foreman-Mackey (2018)](https://ui.adsabs.harvard.edu/abs/2018RNAAS...2...31F/abstract)
* `dynesty` [Speagle (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract)
* `numpy` [Harris et al, (2020)](https://ui.adsabs.harvard.edu/abs/2020Natur.585..357H/abstract)
* `PyMC3` [Salvatier, Wiecki, & Fonnesbeck (2016)](https://ui.adsabs.harvard.edu/abs/2016ascl.soft10016S/abstract)
* `scipy` [Virtanen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2016ascl.soft10016S/abstract)
