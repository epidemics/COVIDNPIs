"""
:code:`any_npi_active.py`

Add an additional NPI that indicates whether any major NPI is active. Major NPIs: 
|['School Closure', 'Stay Home Order', 'Some Businesses Suspended', 'University Closure',
|'Most Businesses Suspended', 'Gatherings <10', 'Gatherings <1000', 'Gatherings <100']

"""


import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse

from scripts.sensitivity_analysis.utils import *

argparser = argparse.ArgumentParser()
add_argparse_arguments(argparser)


if __name__ == '__main__':

    args = argparser.parse_args()
    
    data = preprocess_data('merged_data/double_entry_final.csv', last_day='2020-05-30')
    data.mask_reopenings()

    major_interventions = ['School Closure', 'Stay Home Order', 'Some Businesses Suspended', 'University Closure',
                           'Most Businesses Suspended', 'Gatherings <10', 'Gatherings <1000', 'Gatherings <100']

    nRs, nCMs, nDs = data.ActiveCMs.shape

    ActiveCMs = np.zeros((nRs, nCMs + 1, nDs))
    ActiveCMs[:, :-1, :] = data.ActiveCMs

    maj_indxs = np.array([data.CMs.index(x) for x in major_interventions])

    for r in range(nRs):
        maj_active = np.sum(data.ActiveCMs[r, maj_indxs, :], axis=0)
        # bonus NPI is **any** major NPI is active.
        ActiveCMs[r, -1, :] = maj_active > 0

    data.CMs = [*data.CMs, 'Any NPI Active']
    data.ActiveCMs = ActiveCMs

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    with model_class(data) as model:
        model.build_model(**ep.get_model_build_dict())

    with model.model:
        model.trace = pm.sample(args.n_samples, tune=500, chains=args.n_chains, cores=args.n_chains, max_treedepth=14,
                                target_accept=0.925, init='adapt_diag')

    save_cm_trace('result.txt', model.trace.CMReduction, args.exp_tag, args.model_type)
