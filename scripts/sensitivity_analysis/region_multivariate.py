"""
:code:`region_holdout.py`

Hold out data for a specified region. Useful for exploring how well the model predicts the infection course in held-out data.
"""


import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse
import pickle

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', dest='seed', type=int, help='base seed')
argparser.add_argument('--n_runs', dest='n_runs', type=int, help='num runs per seed')

add_argparse_arguments(argparser)


if __name__ == '__main__':

    args = argparser.parse_args()

    n_runs = args.n_runs
    np.random.seed(args.seed)

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    data = preprocess_data('merged_data/double_entry_final.csv', last_day='2020-05-30')
    data.mask_reopenings(print_out=False)

    nRs, nCMs, nDs = data.ActiveCMs.shape
    sampled_rs = np.random.choice(data.Rs, size=(len(data.Rs), n_runs))

    for i in range(n_runs):
        rs = sampled_rs[:, i]

        new_data = copy.deepcopy(data)
        new_data.Rs = rs

        for r_i, r in enumerate(rs):
            r_oi = data.Rs.index(r)
            new_data.ActiveCMs[r_i, :, :] = copy.deepcopy(data.ActiveCMs[r_oi, :, :])
            new_data.NewCases[r_i, :] = copy.deepcopy(data.NewCases[r_oi, :])
            new_data.NewDeaths[r_i, :] = copy.deepcopy(data.NewDeaths[r_oi, :])

        with model_class(data) as model:
            model.build_model(**ep.get_model_build_dict())

        with model.model:
            model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                    target_accept=0.95, init='adapt_diag')

        out_dir = os.path.join(f'sensitivity_{args.model_type}', 'region_holdout')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        save_cm_trace(f'{args.seed}-{i}.txt', model.trace.CMReduction, args.exp_tag, args.model_type)

        out_dir = os.path.join(f'sensitivity_{args.model_type}', args.exp_tag)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.savetxt(os.path.join(out_dir, f'{args.seed}-{i}-rs.txt'), rs, fmt='%s')

