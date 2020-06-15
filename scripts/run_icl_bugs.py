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
import copy

argparser = argparse.ArgumentParser()
argparser.add_argument("--m", dest="model", type=int)
args = argparser.parse_args()


def mask_region(d, region, days=14):
    i = d.Rs.index(region)
    c_s = np.nonzero(np.cumsum(d.NewCases.data[i, :] > 0) == days + 1)[0][0]
    d_s = np.nonzero(np.cumsum(d.NewDeaths.data[i, :] > 0) == days + 1)[0]
    if len(d_s) > 0:
        d_s = d_s[0]
    else:
        d_s = len(d.Ds)

    d.Active.mask[i, c_s:] = True
    d.Confirmed.mask[i, c_s:] = True
    d.Deaths.mask[i, d_s:] = True
    d.NewDeaths.mask[i, d_s:] = True
    d.NewCases.mask[i, c_s:] = True

    return c_s, d_s


def unmask_all(d):
    d.Active.mask = False
    d.Confirmed.mask = False
    d.Deaths.mask = False
    d.NewDeaths.mask = False
    d.NewCases.mask = False



if __name__ == "__main__":
    
    dp = DataPreprocessor(drop_HS=True)
    data = dp.preprocess_data("notebooks/final_data/data_final.csv")

    if args.model == 0:
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    elif args.model == 1:
        with cm_effect.models.CMCombined_Final_V3(data, None) as model:
            model.build_model()

    elif args.model == 2:
        with cm_effect.models.CMCombined_Final_NoNoise(data, None) as model:
            model.build_model()

    elif args.model == 3:
        with cm_effect.models.CMCombined_Final_ICL(data, None) as model:
            model.build_model()

    elif args.model == 4:
        with cm_effect.models.CMCombined_ICL_NoNoise(data, None) as model:
            model.build_model()

    elif args.model == 5:
        with cm_effect.models.CMCombined_Final_DifEffects(data, None) as model:
            model.build_model()

    elif args.model == 6:
        with cm_effect.models.CMCombined_Additive(data, None) as model:
            model.build_model()

    with model.model:
        model.trace = pm.sample(args.nS, chains=args.nC, target_accept=0.9)

    results_obj = ResultsObject(r_is, model.trace)
    pickle.dump(results_obj, open(f"cv_orders/model_{args.model}_fold_{args.fold}.pkl", "wb"))
