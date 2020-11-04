"""
:code:`utils.py`

Utilities to support the use of command line sensitivity experiments
"""

import epimodel.pymc3_models.models as epm
import os
import numpy as np


def get_model_class_from_str(model_type_str):
    if model_type_str == 'default':
        return epm.DefaultModel
    elif model_type_str == 'additive':
        return epm.AdditiveModel
    elif model_type_str == 'discrete_renewal':
        return epm.DiscreteRenewalModel
    elif model_type_str == 'noisy_r':
        return epm.NoisyRModel
    elif model_type_str == 'different_effects':
        return epm.DifferentEffectsModel
    elif model_type_str == 'cases_only':
        return epm.CasesOnlyModel
    elif model_type_str == 'deaths_only':
        return epm.DeathsOnlyModel
    elif model_type_str == 'deaths_only_discrete_renewal':
        return epm.DeathsOnlyDiscreteRenewalModel
    elif model_type_str == 'complex':
        return epm.ComplexDifferentEffectsModel

def get_target_accept_from_model_str(model_type_str):
    if model_type_str == 'default':
        return 0.905
    elif model_type_str == 'additive':
        return 0.905
    elif model_type_str == 'discrete_renewal':
        return 0.964
    elif model_type_str == 'noisy_r':
        return 0.905
    elif model_type_str == 'different_effects':
        return 0.905
    elif model_type_str == 'cases_only':
        return 0.964
    elif model_type_str == 'deaths_only':
        return 0.964
    elif model_type_str == 'deaths_only_discrete_renewal':
        return 0.964
    elif model_type_str == 'complex':
        return 0.96
    elif model_type_str == 'complex-T':
        return 0.96


def add_argparse_arguments(argparse):
    argparse.add_argument('--model_type', dest='model_type', type=str,
                          help="""model structure choice:
              | - additive: the reproduction rate is given by R_t=R0*(sum_i phi_{i,t} beta_i)
              | - discrete_renewal_fixed_gi: uses discrete renewal model to convert reproduction rate R into growth rate g with fixed generation interval
              | - noisy_r: noise is added to R_t before conversion to growth rate g_t (default model adds noise to g_t after conversion)
              | - different_effects: each region c has a unique NPI reduction coefficient alpha_{i,c}
              | - cases_only: the number of infections is estimated from case data only
              | - deaths_only: the number of infections is estimated from death data only""")
    argparse.add_argument('--exp_tag', dest='exp_tag', type=str, help='experiment identification tag')
    argparse.add_argument('--n_chains', dest='n_chains', type=int, help='the number of chains to run in parallel')
    argparse.add_argument('--n_samples', dest='n_samples', type=int, help='the number of samples to draw')


def parse_extra_model_args(extras):
    kvs = zip(extras[::2], extras[1::2])
    extra_mb_args_dict = {}

    for k, a in kvs:
        if k.startswith('--model_build_arg_'):
            try:
                extra_mb_args_dict[k[len('--model_build_arg_'):]] = float(a)
            except:
                extra_mb_args_dict[k[len('--model_build_arg_'):]] = str(a)

    print('Extra Model Build Args Dictionary')
    print(extra_mb_args_dict)

    return extra_mb_args_dict


def pprint_mb_dict(d):
    print('Model Build Dict'
          '----------------')

    for k, v in d.items():
        print(f'    {k}: {v}')


def generate_base_output_dir(model_type, extra_mb_args_dict):
    out_path = f'sensitivity_{model_type}'
    for k, v in extra_mb_args_dict.items():
        out_path += f'_{k}_{v}'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    return out_path


def save_cm_trace(name, trace, tag, base_path):
    out_dir = os.path.join(base_path, tag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savetxt(os.path.join(out_dir, name), trace)
