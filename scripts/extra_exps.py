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
import pickle
import os

argparser = argparse.ArgumentParser()
argparser.add_argument("--e", dest="exp", type=int)
args = argparser.parse_args()




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
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model(cm_prior='icl')
    elif args.exp == 3:
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model(cm_prior='icl')
    elif args.exp == 4:
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model(cm_prior='icl')
    elif args.exp == 5:
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model(cm_prior='icl')


    with model.model:
        model.trace = pm.sample(2000, tune=500, cores=4, chains=4, max_treedepth=12)

    out_dir = "additional_exps"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savetxt(os.path.join(out_dir, f'exp-{args.exp}.txt'), model.trace['CMReduction'])

