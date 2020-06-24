#!/bin/bash

# Parameters
#------------
# model_types: list of strings
#   accepted values are: "combined", "active", "death", "combined_no_noise", 
#   "combined_v3", "combined_icl", "combined_icl_no_noise", "combined_dif"
#   Must be entered as a list, even if only one element. 
# daily_growth_noise: float
#   If not specified, it uses the default. Can't enter more than one daily_growth_noise, except for 
#   in daily_growth_noise_sensitivity, when it can be a string. If no_noise model, daily_growth_noise is 
#   ignored, whether it's specified or not. 
# min_deaths: int or float
#   If specified, filters deaths less than this number. If not specified, there will be no deaths filter.
# region_var_noise: float
#   default value is 0.2. Only affects results for model_type = "combined_dif". If entered for other model_types,
#   the value is ignored.
# other parameters: 
#   see sensitivitylib.py, most parameters for testing have optional 
#   parameters if you want to try different values


/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; cm_leavout_sensitivity(["combined"]")' &
/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; cm_prior_sensitivity(["combined"]")' &
/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; daily_growth_noise_sensitivity(["combined"]")' &
#/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; data_mob_sensitivity(["combined"], data_path="notebooks/final_data/final_data_extended.csv")' &
#/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; data_schools_open_sensitivity(["combined"], data_path="notebooks/final_data/final_data_extended.csv")' &
/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; delay_mean_sensitivity(["combined"]")' &
#/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; MCMC_stability(["combined"], data_path="notebooks/final_data/final_data_extended.csv")'
/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; min_num_confirmed_sensitivity(["combined"]")' &
/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; R_hyperprior_mean_sensitivity(["combined"]")' &
/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; serial_interval_sensitivity(["combined"]")' &
/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; region_holdout_sensitivity(["combined"]")'&
/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; min_num_deaths_sensitivity(["combined"]")' &
/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; smoothing_sensitivity(["combined"]")'&

# examples using some optional parameters
#/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; region_holdout_sensitivity(["combined_icl_no_noise"], min_deaths=50)'
#/home/mrinank/.cache/pypoetry/virtualenvs/epimodel-KvSMb--q-py3.7/bin/python -c 'from epimodel.pymc3_models.cm_effect.sensitivitylib import *; region_holdout_sensitivity(["combined_dif"], region_var_noise=0.2)'