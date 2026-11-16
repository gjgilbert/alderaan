.. _installation:

Installation
++++++++++++

ALDERAAN pins its dependencies, including a fork of ``batman``. Clone the repository and create the conda environment, which is named ``alderaan``:

.. code-block:: console

   git clone https://github.com/gjgilbert/alderaan.git
   cd alderaan
   conda env create -f environment.yml
   conda activate alderaan

Then install the package from the repository root:

.. code-block:: console

   pip install -e .
