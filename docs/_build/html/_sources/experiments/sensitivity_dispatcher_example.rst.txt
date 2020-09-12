
.. _sensitivity_dispatcher_examples:

Sensitivity Dispatcher Examples
-------------------------------

The sensitivity dispatcher is a tool for running one or more predefined sensitivity experiments using the dataset ``notebooks/double-entry-data/double_entry_final.csv``. It is intended to use when running experiments on a server. This page gives some example commands to run with it.

.. seealso::
	The :ref:`sensitivity_dispatcher` documentation.

Example 1::

	python scripts/sensitivity_dispatcher.py --max_processes 4 --categories region_holdout

Masks data for each of the 41 regions, conditions the default model on the masked data and saves predicted infection course of held out region to ``sensitivity_default/region_holdout/{held out region}.pkl`` (note that with 41 runs, this takes a very long time).

Example 2::

	python scripts/sensitivity_dispatcher.py --max_processes 4 --dry_run --categories npi_leaveout cases_threshold

Prints the commands that would be run for the ``npi_leaveout`` and ``cases_threshold`` sensitivity experiments.

Output::

	Running Univariate Sensitivity Analysis
	---------------------------------------

	Categories: ['npi_leaveout', 'cases_threshold']
	You have requested 14 runs
	Performing Dry Run
	python scripts/sensitivity_analysis/npi_leaveout.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag npi_leaveout --npis 0
	python scripts/sensitivity_analysis/npi_leaveout.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag npi_leaveout --npis 1
	python scripts/sensitivity_analysis/npi_leaveout.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag npi_leaveout --npis 2
	python scripts/sensitivity_analysis/npi_leaveout.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag npi_leaveout --npis 3
	python scripts/sensitivity_analysis/npi_leaveout.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag npi_leaveout --npis 4
	python scripts/sensitivity_analysis/npi_leaveout.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag npi_leaveout --npis 5
	python scripts/sensitivity_analysis/npi_leaveout.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag npi_leaveout --npis 6
	python scripts/sensitivity_analysis/npi_leaveout.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag npi_leaveout --npis 7
	python scripts/sensitivity_analysis/npi_leaveout.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag npi_leaveout --npis 8
	python scripts/sensitivity_analysis/npi_leaveout.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag npi_leaveout --npis 6 7
	python scripts/sensitivity_analysis/preprocessing_tests.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag cases_threshold --smoothing 1 --deaths_threshold 10 --cases_threshold 10
	python scripts/sensitivity_analysis/preprocessing_tests.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag cases_threshold --smoothing 1 --deaths_threshold 10 --cases_threshold 50
	python scripts/sensitivity_analysis/preprocessing_tests.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag cases_threshold --smoothing 1 --deaths_threshold 10 --cases_threshold 150
	python scripts/sensitivity_analysis/preprocessing_tests.py --model_type default --n_samples 2000 --n_chains 4 --exp_tag cases_threshold --smoothing 1 --deaths_threshold 10 --cases_threshold 200

