"""
:code:`agg_holdout.py`

Mask 20 extra days from the end of the data.
"""


import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse
import pickle

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()
add_argparse_arguments(argparser)

if __name__ == '__main__':

    args = argparser.parse_args()

    class ResultsObject():
        def __init__(self, trace):
            self.CMReduction = trace.CMReduction
            self.RegionR = trace.RegionR
            self.InfectedCases = trace.InfectedCases
            self.InfectedDeaths = trace.InfectedDeaths
            self.ExpectedCases = trace.ExpectedCases
            self.ExpectedDeaths = trace.ExpectedDeaths
            self.PsiCases = trace.PsiCases
            self.PsiDeaths = trace.PsiDeaths

    data = preprocess_data('merged_data/double_entry_final.csv', last_day='2020-05-30')
    # mask 20 extra days
    data.mask_reopenings(n_extra=20)

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    with model_class(data) as model:
        model.build_model(**ep.get_model_build_dict())

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.925)

    out_dir = os.path.join(f'sensitivity_{args.model_type}', 'agg_holdout')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    results_obj = ResultsObject(model.trace)
    pickle.dump(results_obj, open(os.path.join(out_dir, f'results.pkl'), 'wb'))