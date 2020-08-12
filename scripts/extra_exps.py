### Initial imports
import logging
import numpy as np
import pymc3 as pm
import seaborn as sns

sns.set_style("ticks")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from epimodel.pymc3_models import cm_effect
from epimodel.pymc3_models.cm_effect.datapreprocessor import DataPreprocessor
import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument("--e", dest="exp", type=int)
args = argparser.parse_args()

def add_extra_cms(data, cms):
    dp = DataPreprocessor(min_confirmed=100)
    data2 = dp.preprocess_data("notebooks/final_data/double_entry_oxcgrt_data.csv")

    nRs, nCMs_orig, nDs = data.ActiveCMs.shape
    nCMs = nCMs_orig + len(cms)

    ActiveCMs = np.zeros((nRs, nCMs, nDs))
    ActiveCMs[:, :nCMs_orig, :] = data.ActiveCMs

    for i, cm in enumerate(cms):
        ActiveCMs[:, nCMs_orig+i, :] = data2.ActiveCMs[:, data2.CMs.index(cm), :nDs]
        data.CMs.append(cm)

    data.ActiveCMs = ActiveCMs
    return data


if __name__ == "__main__":

    dp = DataPreprocessor(min_confirmed=100, drop_HS=True)
    data = dp.preprocess_data("notebooks/final_data/data_final.csv")

    if args.exp == 0:
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()
    elif args.exp == 1:
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model(cm_prior='icl')
    elif args.exp == 2:
        extra_cms = ["Travel Screen/Quarantine", "Travel Bans"]
        data = add_extra_cms(data, extra_cms)
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()
    elif args.exp == 3:
        extra_cms = ["Public Transport Limited"]
        data = add_extra_cms(data, extra_cms)
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()
    elif args.exp == 4:
        extra_cms = ["Internal Movement Limited"]
        data = add_extra_cms(data, extra_cms)
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()
    elif args.exp == 5:
        extra_cms = ["Public Information Limited"]
        data = add_extra_cms(data, extra_cms)
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()


    with model.model:
        model.trace = pm.sample(2000, tune=500, cores=4, chains=4, max_treedepth=12)

    out_dir = "additional_exps"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savetxt(os.path.join(out_dir, f'exp-{args.exp}.txt'), model.trace['CMReduction'])

