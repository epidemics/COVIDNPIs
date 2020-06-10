# Reproductions

## Install
* [Get Poetry](https://python-poetry.org/docs/#installation)
* Install the dependencies and this lib `poetry install` (creates a virtual env by default). Note that you'll need to install PyMC3, which shows up as a tool. 

## Cross-Validation
The file `scripts/run_cv_v2.py` can be used to run cross-validation. Call it as:

``
python scripts/run_cv_v2.py --m MODEL --f FOLD --s NS --c NC
``

`MODEL` is the model to use. The coding is:``
0: Baseline Model
1: Noisy-R Model
2: Discrete Renewal Model
3: Different Effects Model
4: Additive Model. `` `FOLD` takes values `0, 1, 2, 3` i.e., there are four folds. `NS` is the number of samples per chain, defaulting to 2000. `NC` is the number of chains, defaulting to 4.
The output files (pickled python objects) will be saved in a `cv` folder, which at the moment is currently not automatically created.  

See `notebooks/cross_validation/cross_validation.ipynb` for plotting and likelihood calculations. 


## Sensitivity Suite
The file `notebooks/sensitivity_stability_checks/run_sensitivity_suite.sh` shows how to run a full set of experiments for a particularly model. This will save traces `\alpha_i` for each NPI for each model in a folder named `out`.  


## Plotting & Misc
Python notebooks used for other experiments are found in the `notebooks/neurips` folder. Note, that you will need to save the sensitivity traces, categorised by model into different folders (i.e., all baseline results into the `notebooks/neurips/cm_traces/baselin`) model folder for the plotting to work correctly.

## Data
Data is in the `notebooks/final_data/` folder in CSV format. The file used for Monte Carlo Discretisation, `generating_delays.ipynb` can also be found there. 