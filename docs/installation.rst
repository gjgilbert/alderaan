.. _installation:

Installation
++++++++++++

ALDERAAN requires a complex set of dependencies in order to run. To create a conda environment capable of running the ALDERAAN pipeline, copy environment.yml from the `alderaan` github repository to your local machine:

.. code-block:: console

   curl -o ./environment.yml https://raw.githubusercontent.comefs/heads/develop/environment.yml

Then run:

.. code-block:: console

   conda env create -n <ENV_NAME> -f environment.yml

If <ENV_NAME> is not specified, the conda environment will be named "alderaan-env"

You can then activate your environment and safely pip install the package:

.. code-block:: console

   conda activate alderaan-env

then:

.. code-block:: console

   pip install alderaan
