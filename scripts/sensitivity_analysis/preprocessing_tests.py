"""
:code:`preprocessing_tests.py`

Control the smoothing of case and death data, threshold number of cases for data inclusion or threshold number of deaths for data inclusion via command line options.
"""


import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--smoothing', dest='smoothing', type=int, help='Number of days over which to smooth. This should be an odd number. If 1, no smoothing occurs.')
argparser.add_argument('--cases_threshold', dest='cases_threshold', type=int, help='Deaths threshold, below which new daily deaths are ignored.')
argparser.add_argument('--deaths_threshold', dest='deaths_threshold', type=int, help='Confirmed cases threshold, below which new daily cases are ignored.')
add_argparse_arguments(argparser)


if __name__ == '__main__':

    args = argparser.parse_args()
    
    data = preprocess_data('merged_data/double_entry_final.csv', last_day='2020-05-30',
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
                                target_accept=0.96, init='adapt_diag')

    save_cm_trace(output_fname, model.trace.CMReduction, args.exp_tag, args.model_type)
