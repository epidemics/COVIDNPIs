import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse
import copy

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()
add_argparse_arguments(argparser)
# this is a hack to make this work easily.
argparser.add_argument('--model_structure', dest='model_structure', type=str)
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
                                target_accept=0.925)

    save_cm_trace(f'{args.model_structure}.txt', model.trace.CMReduction, args.exp_tag, args.model_type)
