import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse

from .utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--R_prior_mean', dest='R_prior', type=float)
argparser.add_argument('--NPI_prior', nargs=2, dest='NPI_prior', type=str)

add_argparse_arguments(argparser)
args = argparser.parse_args()

if __name__ == '__main__':
    data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30')
    data.mask_reopenings()

    prior_type = args.NPI_prior[0]
    prior_scale = float(args.NPI_prior[1])

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    with model_class(data) as model:
        model.build_model(**ep.get_model_build_dict(), R_prior_mean=args.R_prior, cm_prior=prior_type,
                          cm_prior_scale=prior_scale)

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.925)

    save_cm_trace(output_string, model.trace.CMReduction, args.exp_tag, args.model_type)
