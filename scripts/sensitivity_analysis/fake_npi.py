"""
:code:`region_holdout.py`

Hold out data for a specified region. Useful for exploring how well the model predicts the infection course in held-out data.
"""

import numpy as np
import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse

from scripts.sensitivity_analysis.utils import *

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', dest='seed', type=int, help='seed used to generate fake NPI')
add_argparse_arguments(argparser)

if __name__ == '__main__':
    args, extras = argparser.parse_known_args()

    data = preprocess_data(get_data_path(), last_day='2020-05-30')
    data.mask_reopenings()

    # set seed used to generate fake NPI dates
    np.random.seed(args.seed)

    # generate fake NPI
    npi_onset_dates = np.argmax(np.sum(data.ActiveCMs, axis=1) > 0, axis=-1)

    nRs, nCMs_old, nDs = data.ActiveCMs.shape
    newActiveCMs = np.zeros((nRs, nCMs_old + 1, nDs))
    newActiveCMs[:, :-1, :] = data.ActiveCMs
    for r_i in range(nRs):
        start_date = np.random.randint(low=npi_onset_dates[r_i], high=nDs + 1)
        newActiveCMs[r_i, -1, start_date:] = 1

    data.CMs.append('Fake NPI')
    data.ActiveCMs = newActiveCMs

    # model run as usual
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

    save_cm_trace(f'{str(args.seed)}.txt', model.trace.CMReduction, args.exp_tag,
                  generate_base_output_dir(args.model_type, parse_extra_model_args(extras)))

    if model.country_specific_effects:
        nS, nCMs = model.trace.CMReduction.shape
        full_trace = np.exp(
            np.log(model.trace.CMReduction) + np.random.normal(size=(nS, nCMs)) * model.trace.CMAlphaScales)
        save_cm_trace(f'{str(args.seed)}-cs.txt', full_trace, args.exp_tag,
                      generate_base_output_dir(args.model_type, parse_extra_model_args(extras)))
