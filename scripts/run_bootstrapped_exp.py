### threading
import os

os.environ['THEANO_FLAGS'] = 'OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

### Initial imports
import logging
import numpy as np
import pymc3 as pm
import seaborn as sns

sns.set_style('ticks')

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from epimodel.preprocessing.data_preprocessor import preprocess_data
from epimodel import EpidemiologicalParameters
from epimodel.pymc3_models.models import DefaultModelFixedDispersion

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--base_seed', dest='seed', type=int)
argparser.add_argument('--n_runs', dest='n_runs', type=int)
argparser.add_argument('--parallel_runs', dest='parallel_runs', type=int)
argparser.add_argument('--prior', dest='prior', type=str)
args = argparser.parse_args()

if __name__ == '__main__':
    seed = args.seed

    data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30')
    data.mask_reopenings()

    for run in range(args.n_runs):
        print(f'New Run: Seed: {seed} Prior: {args.prior}')
        ep = EpidemiologicalParameters(seed)
        DelayProbCases, DelayProbDeaths = ep.generate_reporting_and_fatality_delays()
        GI_MEAN, GI_SD = ep.generate_gi()

        prior_scale = 0.2 if args.prior == 'normal' else 10

        with DefaultModelFixedDispersion(data) as model:
            model.build_model(generation_interval_mean=GI_MEAN, generation_interval_sigma=GI_SD,
                              reporting_delay=DelayProbCases,
                              fatality_delay=DelayProbDeaths, cm_prior=args.prior, cm_prior_scale=prior_scale)

        with model.model:
            model.trace = pm.sample(1500, tune=500, cores=4, chains=4, max_treedepth=14, target_accept=0.95)

        out_dir = f'bootstrapped_exps_{args.prior}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.savetxt(f'{out_dir}/{seed}.txt', model.trace.CMReduction)

        seed += args.parallel_runs
