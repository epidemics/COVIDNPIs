"""
:code:`iceswe.py`

Holdout both iceland and sweden
"""

import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse
import pickle

from scripts.sensitivity_analysis.utils import *


import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

argparser = argparse.ArgumentParser()
add_argparse_arguments(argparser)

if __name__ == '__main__':
    args, extras = argparser.parse_known_args()

    data = preprocess_data('merged_data/data_final_nov.csv', last_day='2020-05-30')
    data.mask_reopenings()

    if 'deaths_only' in args.model_type:
        data.remove_regions_min_deaths(5)

    data.mask_region('IS')
    data.mask_region('SE')

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    bd = {**ep.get_model_build_dict(), **parse_extra_model_args(extras)}
    pprint_mb_dict(bd)

    with model_class(data) as model:
        model.build_model(**bd)

    ta = get_target_accept_from_model_str(args.model_type)

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=ta, init='adapt_diag')

    save_cm_trace(f'iceswe.txt', model.trace.CMReduction, args.exp_tag,
                  generate_base_output_dir(args.model_type, parse_extra_model_args(extras)))
