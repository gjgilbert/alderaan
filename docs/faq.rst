.. _faq:

Frequently Asked Questions
==========================

How do I run the pipeline?

.. code-block:: console

   alderaan run -m Kepler -t K00137 --run_id my_run \
       --input_dir /path/to/MAST_downloads \
       --output_dir /path/to/outputs \
       --catalog_dir /path/to/catalogs

``-t`` (target), ``-m`` (mission; currently only Kepler), ``--run_id``, ``--input_dir``, ``--output_dir``, and ``--catalog_dir`` are required. ``--visits`` selects Kepler quarters (default all) and ``--log_level`` sets the console log level. Run ``alderaan run --help`` for details.

Where are the outputs?

Under ``{output_dir}/{run_id}/``: ``results/{target}/`` holds ``{target}_results.fits`` (posterior samples and transit times) and ``{target}_detrended.fits`` (the detrended litecurve); ``plots/{target}/`` holds diagnostic plots; ``logs/`` holds one log file per run. A run refuses to write into a non-empty results or plots directory, so a rerun needs a new run_id.
