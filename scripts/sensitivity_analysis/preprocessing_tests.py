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

    args, extras = argparser.parse_known_args()
    
    data = preprocess_data('merged_data/double_entry_final.csv', last_day='2020-05-30',
                           smoothing=args.smoothing, min_confirmed=args.cases_threshold,
                           min_deaths=args.deaths_threshold)
    data.mask_reopenings()

    if 'deaths_only' in args.model_type:
        data.remove_regions_min_deaths(5)

    output_fname = f'smooth{args.smoothing}-cases_t{args.cases_threshold}-deaths_t{args.deaths_threshold}.txt'

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

    save_cm_trace(f'{output_fname}.txt', model.trace.CMReduction, args.exp_tag,
                  generate_base_output_dir(args.model_type, parse_extra_model_args(extras)))
