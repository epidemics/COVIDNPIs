import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse
import pickle

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--rg', dest='rg', type=str)
add_argparse_arguments(argparser)
args = argparser.parse_args()

if __name__ == '__main__':
    class ResultsObject():
        def __init__(self, indx, trace):
            self.CMReduction = trace.CMReduction
            self.RegionR = trace.RegionR[:, indx]
            self.InfectedCases = trace.InfectedCases[:, indx, :]
            self.InfectedDeaths = trace.InfectedDeaths[:, indx, :]
            self.ExpectedCases = trace.ExpectedCases[:, indx, :]
            self.ExpectedDeaths = trace.ExpectedDeaths[:, indx, :]
            self.PsiCases = trace.PsiCases
            self.PsiDeaths = trace.PsiDeaths


    data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30')
    data.mask_reopenings()
    data.mask_region(args.rg)
    region_index = data.Rs.index(args.rg)

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    with model_class(data) as model:
        model.build_model(**ep.get_model_build_dict())

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.95)

    results_obj = ResultsObject(region_index, model.trace)
    out_dir = os.path.join(f'sensitivity_{args.model_type}', 'region_holdout')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pickle.dump(results_obj, open(os.path.join(out_dir, f'{args.rg}.pkl'), 'wb'))

    save_cm_trace(f'{args.rg}.txt', model.trace.CMReduction, args.exp_tag, args.model_type)
