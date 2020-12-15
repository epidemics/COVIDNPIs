.. COVIDNPIs documentation master file, created by
   sphinx-quickstart on Wed Sep  9 10:48:13 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to COVIDNPIs' documentation!
=====================================

This is the documentation for the `COVIDNPIs project <https://github.com/epidemics/COVIDNPIs/tree/manuscript>`_, Bayesian modelling the impact of non-pharmaceutical interventions (NPIs) on the rate of transmission of COVID-19 in 41 countries around the world. See the paper `Infering the effectiveness of government interventions against COVID-19 <https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v3>`_ by Brauner et. al. for model details, data collection methodology and more.

COVIDNPIs provides a :ref:`data_preprocessor` for converting time-series case and death data along with NPI activation indicators to :ref:`PreprocessedData<preprocessed_data>` objects, ready to use for inference in any of several :ref:`NPI models<cm_model_zoo>`. In addition, the :ref:`model_parameters` module provides utilities for computing delay distributions, which can then be provided as initialisation parameters to the NPI models.

The :ref:`examples` walk through using the PreprocessedData object, initialising and running a model with custom delay parameters. Many pre-defined :ref:`experiments` are can also be run as scripts.

Default Model
=============
Warning! The `DefaultModel` is currently not the default model. `ComplexDifferentEffectsModel` is the latest default model used in our research.

Installation
============
This project uses `Poetry <https://python-poetry.org/>` to manage dependencies. You will need to first install Poetry.

Install dependencies, then activate the virtual environment:

.. code-block::

    poetry install # install project dependencies

.. code-block::

    poetry shell # activate the Python virtualenvironment that Poetry automatically creates.

Minimal Example
===============

.. seealso::
    `Default Model Example`_

.. _Default Model Example: examples/CM_Model_Examples.ipynb


The following steps are sufficient to run the default model with the dataset ``notebooks/double-entry-data/double_entry_final.csv`` and save the NPI reduction trace to ``CMReduction_trace.txt`` which can be loaded with :code:`numpy.loadtxt`


.. code-block::

    from epimodel.preprocessing.data_preprocessor import preprocess_data
    from epimodel.pymc3_models.models import DefaultModel, ComplexDifferentEffectsModel
    from epimodel.pymc3_models.epi_params import EpidemiologicalParameters, bootstrapped_negbinom_values
    import pymc3 as pm

    data = preprocess_data('merged_data/double_entry_final.csv')
    data.mask_reopenings()

    ep = EpidemiologicalParameters() # object containing epi params

    with ComplexDifferentEffectsModel(data) as model:
        # run using latest epidemiological parameters
        model.build_model(**ep.get_model_build_dict())

    with model.model:
        model.trace = pm.sample(2000, tune=1000, cores=4, chains=4, max_treedepth=14, target_accept=0.96)

    numpy.savetxt('CMReduction_trace.txt', model.trace['CMReduction'])



Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples/examples
   module_documentation/module_documentation
   experiments/experiments
   reproduction/reproduction
   troubleshooting/troubleshooting



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`