"""
:code:`epiparam.py`

Specify the prior generation interval, case reporting and death delay distributions using command line parameters.
"""

import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--gi_mean_mean', dest='gi_mean_mean', type=float,
                       help='Mean of the prior over generation interval means')
argparser.add_argument('--gi_mean_sd', dest='gi_mean_sd', type=float,
                       help='Standard deviation of the prior over generation interval means')
argparser.add_argument('--deaths_mean_mean', dest='deaths_mean_mean', type=float,
                       help='Mean of the prior over infection-to-death delay means')
argparser.add_argument('--deaths_mean_sd', dest='deaths_mean_sd', type=float,
                       help='Standard deviation of the prior over infection-to-death delay means')
argparser.add_argument('--cases_mean_mean', dest='cases_mean_mean', type=float,
                       help='Mean of the prior over infection-to-reporting delay means')
argparser.add_argument('--cases_mean_sd', dest='cases_mean_sd', type=float,
                       help='Mean of the prior over infection-to-reporting delay means')

add_argparse_arguments(argparser)

if __name__ == '__main__':
    args, extras = argparser.parse_known_args()

    data = preprocess_data('merged_data/double_entry_final.csv', last_day='2020-05-30')
    data.mask_reopenings()

    output_fname = f'gi_mean_mean-{args.gi_mean_mean}-gi_mean_sd-{args.gi_mean_sd}' \
                   f'deaths_mean_mean-{args.deaths_mean_mean}-deaths_mean_sd-{args.deaths_mean_sd}' \
                   f'cases_mean_mean-{args.cases_mean_mean}-cases_mean_sd-{args.cases_mean_sd}'

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    bd = {**ep.get_model_build_dict(), **parse_extra_model_args(extras)}
    pprint_mb_dict(bd)

    # update params from args
    bd['gi_mean_mean'] = args.gi_mean_mean
    bd['gi_mean_sd'] = args.gi_mean_sd
    bd['deaths_delay_mean_mean'] = args.deaths_mean_mean
    bd['deaths_delay_mean_sd'] = args.deaths_mean_sd
    bd['cases_delay_mean_mean'] = args.cases_mean_mean
    bd['cases_delay_mean_sd'] = args.cases_mean_sd

    with model_class(data) as model:
        model.build_model(**bd)

    with model.model:
        # some traces don't run here without init='adapt_diag'
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.95, init='adapt_diag')

    save_cm_trace(f'{output_fname}.txt', model.trace.CMReduction, args.exp_tag,
                  generate_base_output_dir(args.model_type, parse_extra_model_args(extras)))
