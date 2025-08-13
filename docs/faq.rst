.. _faq:

Frequently Asked Questions
==========================

How do I run the pipeline?

To run the pipeline, navigate into the ALDERAAN source directory and run the following commmand:

.. code-block:: console

   python alderaan/pipelines/alderaan_pipeline.py -m Kepler -t K00148 -c configs/default_config.cfg
 
The flags -m (mission) -t (target) and -c (config) are required and set the pipeline run conditions.