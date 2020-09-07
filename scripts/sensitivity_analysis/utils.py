import epimodel.pymc3_models.models as epm
import os
import numpy as np


def get_model_class_from_str(model_type_str):
    if model_type_str == 'default':
        return epm.DefaultModel
    elif model_type_str == 'additive':
        return epm.AdditiveModel
    elif model_type_str == 'discrete_renewal_fixed_gi':
        return epm.DiscreteRenewalFixedGIModel
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


def add_argparse_arguments(argparse):
    argparse.add_argument('--model_type', dest='model_type', type=str)
    argparse.add_argument('--exp_tag', dest='exp_tag', type=str)
    argparse.add_argument('--n_chains', dest='n_chains', type=int)
    argparse.add_argument('--n_samples', dest='n_samples', type=int)


def save_cm_trace(name, trace, tag, model_type):
    out_dir = os.path.join(f'sensitivity_{model_type}', tag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savetxt(os.path.join(out_dir, name), trace)
