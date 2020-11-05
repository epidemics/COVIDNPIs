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
args, extras = argparser.parse_known_args()

if __name__ == '__main__':
    data = preprocess_data(get_data_path(), last_day='2020-05-30')
    data.mask_reopenings()

    output_fname = f'gi_mean_mean-{args.gi_mean_mean}-gi_mean_sd-{args.gi_mean_sd}' \
                   f'deaths_mean_mean-{args.deaths_mean_mean}-deaths_mean_sd-{args.deaths_mean_sd}' \
                   f'cases_mean_mean-{args.cases_mean_mean}-cases_mean_sd-{args.cases_mean_sd}.txt'

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)
    bd = {**ep.get_model_build_dict(), **parse_extra_model_args(extras)}

    # update params from args
    bd['gi_mean_mean'] = args.gi_mean_mean
    bd['gi_mean_sd'] = args.gi_mean_sd
    bd['deaths_delay_mean_mean'] = args.deaths_mean_mean
    bd['deaths_delay_mean_sd'] = args.deaths_mean_sd
    bd['cases_delay_mean_mean'] = args.cases_mean_mean
    bd['cases_delay_mean_sd'] = args.cases_mean_sd

    pprint_mb_dict(bd)

    with model_class(data) as model:
        model.build_model(**bd)

    with model.model:
        # some traces don't run here without init='adapt_diag'
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.96, init='adapt_diag')

    save_cm_trace({output_fname}, model.trace.CMReduction, args.exp_tag,
                  generate_base_output_dir(args.model_type, parse_extra_model_args(extras)))

    if model.country_specific_effects:
        output_fname.replace('.txt', '-cs.txt')
        nS, nCMs = model.trace.CMReduction.shape
        full_trace = np.exp(np.log(model.trace.CMReduction) + np.random.normal(size=(nS, nCMs)) * model.trace.CMAlphaScales)
        save_cm_trace(output_fname, full_trace, args.exp_tag,
                      generate_base_output_dir(args.model_type, parse_extra_model_args(extras)))
