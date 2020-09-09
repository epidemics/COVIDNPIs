import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()
add_argparse_arguments(argparser)
# this is a hack to make this work easily.
argparser.add_argument('--model_structure', dest='model_structure', type=str, 
    help='| model structure choice:\
          | - additive: the reproduction rate is given by R_t=R0*(sum_i phi_{i,t} beta_i)\
          | - discrete_renewal_fixed_gi: uses discrete renewal model to convert reproduction rate R into growth rate g with fixed generation interval\
          | - discrete_renewal: uses discrete renewal model to convert reproduction rate R into growth rate g with prior over generation intervals\
          | - noisy_r: noise is added to R_t before conversion to growth rate g_t (default model adds noise to g_t after conversion)\
          | - different_effects: each region c has a unique NPI reduction coefficient alpha_{i,c}\
          | - cases_only: the number of infections is estimated from case data only\
          | - deaths_only: the number of infections is estimated from death data only')
args = argparser.parse_args()

if __name__ == '__main__':
    data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30')
    data.mask_reopenings()

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_structure)

    with model_class(data) as model:
        model.build_model(**ep.get_model_build_dict())

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.95)

    save_cm_trace(f'{args.model_structure}.txt', model.trace.CMReduction, args.exp_tag, args.model_type)
