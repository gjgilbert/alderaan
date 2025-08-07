.. _installation:

Installation
++++++++++++

ALDERAAN requires a complex set of dependencies in order to run. To create a conda environment capable of running the ALDERAAN pipeline, copy environment.yml to your local machine and run:

.. code-block:: console

   conda env create -n <ENV_NAME> -f environment.yml

If <ENV_NAME> is not specified, the conda environment will be named "alderaan"

You can then activate your environment and safely pip install the package:

.. code-block:: console

   conda activate alderaan-env

then:

.. code-block:: console

   pip install alderaan
