# COVID-19 Countermeasure Effectiveness

This repo contains the code used for [Brauner et al. *The effectiveness and perceived burden of nonpharmaceutical interventions against COVID-19 transmission: a modelling study with 41 countries* (2020)](https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v2.article-info)

## Just here for the data?
The data we collected, including sources, is available in the 'data' folder. Please refer to the paper for a description of the data and the data collection process.


## Install

* [Get Poetry](https://python-poetry.org/docs/#installation)
* Clone this repository.
* Install the dependencies and this lib `poetry install` (creates a virtual env by default).

## Brief Description
* The final version of the PyMC3 model we used is found in `epimodel/pymc3_models/cm_effect/models.py` and is called `CMCombined_Final`. This file also contains alternative model implementations that we used for structural sensitivity analysis. 

* `epimodel/pymc3_models/cm_effect/datapreprocessor.py` contains `DataPreprocessor` classes that are used for data preprocessing with different options.  

* `sensitivitylib.py` contains a number of sensitivity analyses in library form. 

## NPI Data
`notebooks/double-entry-data/double_entry_final.csv` has the final data CSV, containing NPI data for 9 NPIs that we collected across 41 regions. It also includes additional NPIs taken from the [OxCGRT](https://github.com/OxCGRT/covid-policy-tracker) dataset. This is the latest version of data, using NPI data that has had double-entry.  


## Reproducibility
**Please see `notebooks/double-entry-data`**. 

### Main results
Please run `notebooks/double-entry-data/final_results.ipynb` to do a complete model run. This saves the trace as a Python pickle. 

### Additional Experiments
`scripts/run_add_exps.sh` is used to run "additional" validation experiments, and this file saves full model traces as pickle files (these end up being quite large files). The experiments run are:
1. Additive model. 
2. Noisy-R model. 
3. Different-effects model (each NPI has a different effectiveness in each country).
3. A noisy discrete-renewal model. 
4. Inclusion of OxCGRT Travel NPIs. 
5. Inclusion of OxCGRT public transport NPI. 
5. Inclusion of OxCGRT internal movement NPI. 
5. Inclusion of OxCGRT information campaign NPI. 
5. Inclusion of OxCGRT symptomatic testing NPI. 
5. Inclusion of bonus NPI indicating the onset of intervention.
5. Estimating the effecting using the timing of each NPI i.e., the n-th NPI is used rather than the specific NPI . 
5. Different delays in countries that had symptomatic testing. 
5. Delaying schools and universities by approx. 1 generation interval. 
5. Performing aggregrated holdouts. 
5. Cases only model. 
6. Deaths only model. 

### Country Holdouts
Please run `scripts/run_ho_exps.sh`. Custom `ResultsObject` pickle files are saved with data pertaining to the estimated NPI effectiveness and predictions for the unseen countries. We mask cases after the first 14 days of cases (likewise for deaths) and use this to assess model quality. 

### Sensitivity Analyses
Please run `scripts/run_sensitivity_suite_extended.sh`. This will automatically save effectiveness estimates, which you will need to move to the `notebooks/double-entry-data/traces` folder (create the folder if it does not exist) for plotting code. 

### Plotting code
See `notebooks/double-entry-data` for plotting code (plotting notebooks have `_plotter` in their name). These notebooks expect the relevant `.pkl` files and NPI effectiveness traces from the above experiments placed into the appropriate directories. Usually, `.pkl` files are expected in the notebook directory `notebooks/double-entry-data` (using the `local` flag in the notebooks). NPI effectiveness traces (`.txt` files produced from the sensitivity analysis suite) are expected in the `notebooks/double-entry-data/traces` directory. 
