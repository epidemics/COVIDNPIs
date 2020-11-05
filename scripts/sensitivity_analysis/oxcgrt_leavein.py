"""
:code:'oxcgrt_leavein.py'

Include extra NPIs from the OxCGRT dataset.
"""


import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse
import copy

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--npis', nargs='+', dest='npis', type=int, help = '''OxCGRT NPIs to include. One or more of:
                                                                        | Travel Screen/Quarantine
                                                                        | Travel Bans
                                                                        | Public Transport Limited
                                                                        | Internal Movement Limited
                                                                        | Public Information Campaigns
                                                                        | Symptomatic Testing''')
add_argparse_arguments(argparser)


if __name__ == '__main__':

    args, extras = argparser.parse_known_args()

    # this is the default drop values
    drop_features_full = ['Mask Wearing', 'Travel Screen/Quarantine', 'Travel Bans', 'Public Transport Limited',
                          'Internal Movement Limited', 'Public Information Campaigns', 'Symptomatic Testing']

    drop_features = copy.deepcopy(drop_features_full)

    output_string = ''
    for npi_index in args.npis:
        drop_features.remove(drop_features_full[npi_index])
        output_string = f'{output_string}{npi_index}'
    output_string = f'{output_string}.txt'

    data = preprocess_data(get_data_path(), last_day='2020-05-30',
                           drop_features=drop_features)
    data.mask_reopenings()

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    bd = {**ep.get_model_build_dict(), **parse_extra_model_args(extras)}

    with model_class(data) as model:
        model.build_model(**bd)

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.96, init='adapt_diag')

    save_cm_trace(output_string, model.trace.CMReduction, args.exp_tag,
                  generate_base_output_dir(args.model_type, parse_extra_model_args(extras)))

    if model.country_specific_effects:
        output_string.replace('.txt', '-cs.txt')
        nS, nCMs = model.trace.CMReduction.shape
        full_trace = np.exp(np.log(model.trace.CMReduction) + np.random.normal(size=(nS, nCMs)) * model.trace.CMAlphaScales)
        save_cm_trace(output_string, full_trace, args.exp_tag,
                      generate_base_output_dir(args.model_type, parse_extra_model_args(extras)))
