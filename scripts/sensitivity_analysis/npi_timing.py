"""
:code:`npi_timing.py`

Model the effect of the number of NPIs active, rather than the particular NPIs deployed.
"""

import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse
import copy

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()
add_argparse_arguments(argparser)

if __name__ == '__main__':

    args = argparser.parse_args()
    
    data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30')
    data.mask_reopenings()

    ActiveCMs = copy.deepcopy(data.ActiveCMs)
    nRs, nCMs, nDs = ActiveCMs.shape

    for r in range(nRs):
        n_active = np.sum(data.ActiveCMs[r, :, :], axis=0)
        for i in range(nCMs):
            ActiveCMs[r, i, :] = n_active > i

    data.CMs = [f'NPI {i + 1}' for i in range(nCMs)]
    data.ActiveCMs = ActiveCMs

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    with model_class(data) as model:
        model.build_model(**ep.get_model_build_dict())

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.925)

    save_cm_trace('result.txt', model.trace.CMReduction, args.exp_tag, args.model_type)
