.. _individual_sensitivity_experiments:

**********************************
Individual Sensitivity Experiments
**********************************


Agg Holdout
=====================

Holdout the final 20 days of data.

.. argparse:: 
   :module: scripts.sensitivity_analysis.agg_holdout
   :func: argparser
   :prog: python agg_holdout.py


Alternative Priors
==================

Alternative priors for basic reproductive number ``R0`` and NPI effectiveness.

.. argparse::
   :module: scripts.sensitivity_analysis.alternative_build_param
   :func: argparser
   :prog: python scripts/alternative_build_param.py

Any NPI Active
==============

Add an additional dummy NPI representing whether any major NPI is active

.. argparse::
   :module: scripts.sensitivity_analysis.any_npi_active
   :func: argparser
   :prog: python scripts/any_npi_active.py


Delay Schools
==================

Add artificial 5 day delay to school closure NPI.

.. argparse::
   :module: scripts.sensitivity_analysis.delay_schools
   :func: argparser
   :prog: python scripts/delay_schools.py

NPI Leaveout
============

Remove NPIs from the dataset, fitting data to the remaining NPIs.

.. argparse::
   :module: scripts.sensitivity_analysis.npi_leaveout
   :func: argparser
   :prog: python scripts/npi_leaveout.py

NPI Timing
============

Replace NPIs with indices 0-8 representing 0,...,8 active NPIs, ignoring NPI type

.. argparse::
   :module: scripts.sensitivity_analysis.npi_timing
   :func: argparser
   :prog: python scripts/npi_timing.py

OxCGRT Leavin
=============

Include additional NPIs from OxCGRT

.. argparse::
   :module: scripts.sensitivity_analysis.oxcgrt_leavein
   :func: argparser
   :prog: python scripts/oxcgrt_leavein.py



Preprocessing Tests
=====================

Modify the smoothing window, threshold number of cases or threshold number of deaths below which data is masked.

.. argparse:: 
   :module: scripts.sensitivity_analysis.preprocessing_tests
   :func: argparser
   :prog: python scripts/preprocessing_tests.py


Structural
==========

Alternative model structures.

.. argparse:: 
   :module: scripts.sensitivity_analysis.structural
   :func: argparser
   :prog: python scripts/structural.py





