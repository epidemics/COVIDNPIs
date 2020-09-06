import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse
import copy

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--npis', nargs='+', dest='npis', type=int)
add_argparse_arguments(argparser)
args = argparser.parse_args()

if __name__ == '__main__':
    # this is the default drop values
    drop_features_full = ['Travel Screen/Quarantine', 'Travel Bans', 'Public Transport Limited',
                          'Internal Movement Limited', 'Public Information Campaigns', 'Symptomatic Testing']

    drop_features = copy.deepcopy(drop_features_full)

    output_string = ''
    for npi_index in args.npis:
        drop_features.remove(drop_features_full[npi_index])
        output_string = f'{output_string}{npi_index}'
    output_string = f'{output_string}.txt'

    data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30',
                           drop_features=drop_features)
    data.mask_reopenings()

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    with model_class(data) as model:
        model.build_model(**ep.get_model_build_dict())

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.95)

    save_cm_trace(output_string, model.trace.CMReduction, args.exp_tag, args.model_type)
