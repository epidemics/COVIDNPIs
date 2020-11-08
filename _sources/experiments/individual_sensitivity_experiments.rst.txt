.. _individual_sensitivity_experiments:

**********************************
Individual Sensitivity Experiments
**********************************


Alternative Priors
==================

Alternative priors for basic reproductive number ``R0`` and NPI effectiveness.

.. argparse::
   :module: scripts.sensitivity_analysis.alternative_build_param
   :func: argparser
   :prog: python scripts/alternative_build_param.py


NPI Leaveout
============

Remove NPIs from the dataset, fitting data to the remaining NPIs.

.. argparse::
   :module: scripts.sensitivity_analysis.npi_leaveout
   :func: argparser
   :prog: python scripts/npi_leaveout.py


NPI Add-in
=============

Include additional NPIs, primarily from OxCGRT

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


Multivariate Epidemiological Parameter Prior Sensitivity
========================================================

Jointly change priors over epidemiological parameters.

.. argparse::
   :module: scripts.sensitivity_analysis.epiparam
   :func: argparser
   :prog: python scripts/epiparam.py

Iceland Sweden Holdout
======================

Holdout Iceland and Sweden Together

.. argparse::
   :module: scripts.sensitivity_analysis.iceswe
   :func: argparser
   :prog: python scripts/iceswe.py

Scalings
======================

Scale cases.

.. argparse::
   :module: scripts.sensitivity_analysis.scaling
   :func: argparser
   :prog: python scripts/scaling.py
