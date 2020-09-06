import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse

from .utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--smoothing', dest='smoothing', type=int)
argparser.add_argument('--cases_threshold', dest='cases_threshold', type=int)
argparser.add_argument('--deaths_threshold', dest='deaths_threshold', type=int)
add_argparse_arguments(argparser)
args = argparser.parse_args()

if __name__ == '__main__':
    data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30',
                           smoothing=args.smoothing, min_confirmed=args.cases_threshold,
                           min_deaths=args.deaths_threshold)
    data.mask_reopenings()
    output_fname = f'smooth{args.smoothing}-cases_t{args.cases_threshold}-deaths_t{args.deaths_threshold}.txt'

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    with model_class(data) as model:
        model.build_model(**ep.get_model_build_dict())

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.95)

    save_cm_trace(output_fname, model.trace.CMReduction, args.exp_tag, args.model_type)
