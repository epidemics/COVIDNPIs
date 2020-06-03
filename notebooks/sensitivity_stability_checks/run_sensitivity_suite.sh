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


python3 -c 'from sensitivitylib import *; cm_leavout_sensitivity(["combined"], daily_growth_noise=0.05)' 
python3 -c 'from sensitivitylib import *; cm_prior_sensitivity(["combined"], daily_growth_noise=0.05)'
python3 -c 'from sensitivitylib import *; daily_growth_noise_sensitivity(["combined"])'
python3 -c 'from sensitivitylib import *; data_mob_sensitivity(["combined"], daily_growth_noise=0.05)'
python3 -c 'from sensitivitylib import *; data_schools_open_sensitivity(["combined"], daily_growth_noise=0.05)'
python3 -c 'from sensitivitylib import *; delay_mean_sensitivity(["combined"], daily_growth_noise=0.05)'
python3 -c 'from sensitivitylib import *; MCMC_stability(["combined"], daily_growth_noise=0.05)'
python3 -c 'from sensitivitylib import *; min_num_confirmed_sensitivity(["combined"], daily_growth_noise=0.05)'
python3 -c 'from sensitivitylib import *; R_hyperprior_mean_sensitivity(["combined"], daily_growth_noise=0.05)'
python3 -c 'from sensitivitylib import *; serial_interval_sensitivity(["combined"], daily_growth_noise=0.05)'
python3 -c 'from sensitivitylib import *; region_holdout_sensitivity(["combined"], daily_growth_noise=0.05)'

# examples using some optional parameters
python3 -c 'from sensitivitylib import *; region_holdout_sensitivity(["combined_icl_no_noise"], min_deaths=50)'
python3 -c 'from sensitivitylib import *; region_holdout_sensitivity(["combined_dif"], region_var_noise=0.2)'