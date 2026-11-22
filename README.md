# ALDERAAN
**A**utomated **L**ightcurve **D**etrending, **E**xoplanet **R**ecovery, and **A**nalysis of **A**utocorrelated **N**oise

ALDERAAN is a fast and reliable pipeline for processing exoplanet transit photometry. The pipeline is currently capable of handling data from the *Kepler* Space Telescope, but in the near future will be extended to handle data from *K2* and *TESS*.

Detrending and transit fitting are optimized for high-fidelity measurements of [P, t0, Rp/Rs, b, T14] and for inference of dynamical transit timing variations (TTVs). Noise arising from instrumental and astrophysical sources is handled using a combination of Gaussian Processes (GP) regression and autocorrelated frequency analysis. Model sampling is performed using dynamic nested sampling.

For detailed documentation, see [readthedocs.org](https://alderaan.readthedocs.io/en/latest/)

## Installation instructions

`alderaan` pins its dependencies, including a fork of `batman`. Clone the repository and create the conda environment, which is named `alderaan`:

```
git clone https://github.com/gjgilbert/alderaan.git
cd alderaan
conda env create -f environment.yml
conda activate alderaan
```

Then install the package from the repository root:

```
pip install -e .
```


## Running the pipeline

Run the pipeline on one target with:

```
alderaan run -m Kepler -t K00137 --run_id my_run \
    --input_dir /path/to/MAST_downloads \
    --output_dir /path/to/outputs \
    --catalog_dir /path/to/catalogs
```

Required flags:
* `-t/--target`: KOI identifier, e.g. K00137
* `-m/--mission`: observing mission (currently only Kepler)
* `--run_id`: run identifier; outputs of different runs never mix, so a rerun needs a new run_id
* `--input_dir`: directory of MAST photometry downloads (`kplr*_llc.fits`)
* `--output_dir`: directory for outputs
* `--catalog_dir`: directory holding `kepler_merged_alderaan.csv` (from `scripts/merge_kepler.py`) and `holczer_2016_kepler_ttvs.txt`

Optional flags: `--visits` (Kepler quarters to load; default all) and `--log_level` (console level; the log file always records DEBUG).

Outputs, under `{output_dir}/{run_id}/`:
* `results/{target}/{target}_results.fits`: posterior samples (SAMPLES, EPHEMERIS) and transit times (TTIMES_nn); the PRIMARY header records RUN_ID, VERSION, and GIT_HASH
* `results/{target}/{target}_detrended.fits`: the detrended litecurve
* `plots/{target}/`: litecurves, O-C diagrams, and dynesty diagnostics
* `logs/`: one log file per run

When running many targets in parallel, set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to 1 so thread pools do not oversubscribe the cores; the log records their values.


## Attribution
If you make use of `alderaan` in your work, please cite [Gilbert, Petigura, & Entrican (2025)](https://ui.adsabs.harvard.edu/abs/2025PNAS..12205295G/abstract).

The mathematical formalism and motivation behind `alderaan` transit modeling choices was developed in [Carter et al. (2008)](https://ui.adsabs.harvard.edu/abs/2008ApJ...689..499C/abstract), [Kipping (2013)](https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract), [Gilbert, MacDougall, & Petigura (2022)](https://ui.adsabs.harvard.edu/abs/2022AJ....164...92G/abstract), and [MacDougall, Gilbert, & Petigura (2023)](https://ui.adsabs.harvard.edu/abs/2023AJ....166...61M/abstract).

Foundational work on transit modeling which informed the development of `alderaan` is described in [Mandel & Agol (2002)](https://ui.adsabs.harvard.edu/abs/2002ApJ...580L.171M/abstract), [Seager & Mallén-Ornelas (2003)](https://ui.adsabs.harvard.edu/abs/2003ApJ...585.1038S/abstract), [Kipping (2010)](https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.1758K/abstract), [Eastman, Gaudi, & Agol (2013)](https://ui.adsabs.harvard.edu/abs/2013PASP..125...83E/abstract), and [Foreman-Mackey et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021JOSS....6.3285F/abstract).

The `alderaan` pipeline relies on the following core dependencies:
* `astropy` [Astropy Collaboration et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...935..167A/abstract)
* `batman` [Kreidberg (2015)](https://ui.adsabs.harvard.edu/abs/2015ascl.soft10002K/abstract)
* `celerite2` [Foreman-Mackey (2018)](https://ui.adsabs.harvard.edu/abs/2018RNAAS...2...31F/abstract)
* `dynesty` [Speagle (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract)
* `numpy` [Harris et al, (2020)](https://ui.adsabs.harvard.edu/abs/2020Natur.585..357H/abstract)
* `scipy` [Virtanen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020NatMe..17..261V/abstract)
