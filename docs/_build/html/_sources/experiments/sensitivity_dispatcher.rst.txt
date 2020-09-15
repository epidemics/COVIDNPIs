*******************************
Sensitivity Dispatcher
*******************************

The sensitivity dispatcher is a tool for running one or more predefined sensitivity experiments using the dataset ``notebooks/double-entry-data/double_entry_final.csv``. It is intended to use when running experiments on a server.

.. seealso::
	| Documentation for :ref:`sensitivity_dispatcher_examples`
	| Documentation for :ref:`individual_sensitivity_experiments`

.. _sensitivity_dispatcher:

Sensitivity Dispatcher Usage
===============================

.. argparse:: 
   :module: scripts.sensitivity_dispatcher
   :func: argparser
   :prog: python scripts/sensitivity_dispatcher.py


.. _sensitivity_dispatcher_run_types:

Sensitivity Analysis Dispatcher Run Types
=========================================

Full specifications can be found in :code:`scripts/sensitivity_analysis/sensitivity_analysis.yaml`, which can also be customised to your needs.

.. autoyaml:: /scripts/sensitivity_analysis/sensitivity_analysis.yaml
