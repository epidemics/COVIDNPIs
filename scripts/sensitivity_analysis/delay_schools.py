"""
:code:`delay_schools.py`

Add an artificial delay of 5 days to the school and university closure NPIs.
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

    data = preprocess_data('merged_data/double_entry_final.csv', last_day='2020-05-30')
    data.mask_reopenings()

    n_delay = 5
    to_delay_index = [data.CMs.index(cm) for cm in ['School Closure', 'University Closure']]
    active_cms = copy.deepcopy(data.ActiveCMs)
    data.ActiveCMs[:, to_delay_index, n_delay:] = active_cms[:, to_delay_index, :-n_delay]
    data.ActiveCMs[:, to_delay_index, :n_delay] = 0

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    with model_class(data) as model:
        model.build_model(**ep.get_model_build_dict())

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.925, init='adapt_diag')

    save_cm_trace('result.txt', model.trace.CMReduction, args.exp_tag, args.model_type)
