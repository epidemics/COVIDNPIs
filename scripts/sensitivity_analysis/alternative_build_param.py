"""
:code:`alternative_build_param.py`

Initialise model with alternative command line specified build parameters:
| --R_prior_mean: Prior mean basic reproductive number R0
| --NPI_prior: Prior for NPI effectiveness
| --growth_noise: Add noise to case growth rate
"""

import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()

argparser.add_argument('--R_prior_mean', dest='R_prior', type=float, help='Prior mean basic reproductive number R0')
argparser.add_argument('--NPI_prior', nargs=2, dest='NPI_prior', type=str, help='Prior for NPI effectiveness')
argparser.add_argument('--growth_noise', dest='growth_noise', type=float, help='Growth noise scale parameter')


add_argparse_arguments(argparser)


if __name__ == '__main__':

    args = argparser.parse_args()

    data = preprocess_data('merged_data/double_entry_final.csv', last_day='2020-05-30')
    data.mask_reopenings()

    prior_type = args.NPI_prior[0]
    prior_scale = float(args.NPI_prior[1])

    output_fname = f'R_prior-{args.R_prior}-npi_prior-{args.NPI_prior[0]}-npi_prior_scale-{args.NPI_prior[1]}-' \
                   f'growth_noise-{args.growth_noise}.txt'
    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    with model_class(data) as model:
        model.build_model(**ep.get_model_build_dict(), R_prior_mean=args.R_prior, cm_prior=prior_type,
                          cm_prior_scale=prior_scale, growth_noise_scale=args.growth_noise)

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.925)

    save_cm_trace(output_fname, model.trace.CMReduction, args.exp_tag, args.model_type)
