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
argparser.add_argument('--gi_mean_mean', dest='gi_mean_mean', type=float, help='Mean of the prior over generation interval means')
argparser.add_argument('--gi_mean_sd', dest='gi_mean_sd', type=float, help='Standard deviation of the prior over generation interval means')
argparser.add_argument('--deaths_mean_mean', dest='deaths_mean_mean', type=float, help='Mean of the prior over infection-to-death delay means')
argparser.add_argument('--deaths_mean_sd', dest='deaths_mean_sd', type=float, help='Standard deviation of the prior over infection-to-death delay means')
argparser.add_argument('--cases_mean_mean', dest='cases_mean_mean', type=float, help='Mean of the prior over infection-to-reporting delay means')
argparser.add_argument('--cases_mean_sd', dest='cases_mean_sd', type=float, help='Mean of the prior over infection-to-reporting delay means')

add_argparse_arguments(argparser)
args = argparser.parse_args()

if __name__ == '__main__':
    data = preprocess_data('merged_data/data_final_nov.csv', last_day='2020-05-30')
    data.mask_reopenings()

    output_fname = f'gi_mean_mean-{args.gi_mean_mean}-gi_mean_sd-{args.gi_mean_sd}' \
                   f'deaths_mean_mean-{args.deaths_mean_mean}-deaths_mean_sd-{args.deaths_mean_sd}' \
                   f'cases_mean_mean-{args.cases_mean_mean}-cases_mean_sd-{args.cases_mean_sd}'

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)
    model_build_dict = ep.get_model_build_dict()

    # update params from args
    model_build_dict['gi_mean_mean'] = args.gi_mean_mean
    model_build_dict['gi_mean_sd'] = args.gi_mean_sd
    model_build_dict['deaths_delay_mean_mean'] = args.deaths_mean_mean
    model_build_dict['deaths_delay_mean_sd'] = args.deaths_mean_sd
    model_build_dict['cases_delay_mean_mean'] = args.cases_mean_mean
    model_build_dict['cases_delay_mean_sd'] = args.cases_mean_sd

    with model_class(data) as model:
        model.build_model(**model_build_dict)

    with model.model:
        # some traces don't run here without init='adapt_diag'
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.96, init='adapt_diag')

    save_cm_trace(f'{output_fname}.txt', model.trace.CMReduction, args.exp_tag, args.model_type)

    # save extra epiparam stuff, so we can check
    varnames = ['GI_mean', 'GI_sd', 'CasesDelayMean', 'CasesDelayDisp', 'DeathsDelayMean', 'DeathsDelayDisp']
    for v in varnames:
        # save as a '.csv' file for convenience. All cmreds are .txts.
        try:
            save_cm_trace(f'{output_fname}_{v}.csv', model.trace[v], args.exp_tag, args.model_type)
        except:
            print(f'Skipped saving {v}')
