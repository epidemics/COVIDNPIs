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

argparser = argparse.ArgumentParser()
argparser.add_argument("--l", dest="last_date", type=int)
argparser.add_argument("--m", dest="model", type=int)
args = argparser.parse_args()

if __name__ == "__main__":

    last_dates = ["2020-04-27", "2020-05-04", "2020-05-11", "2020-05-18", "2020-05-25", None]
    dp = DataPreprocessor(min_confirmed=100)
    data = dp.preprocess_data("notebooks/double-entry-data/double_entry_with_OxCGRT.csv",
                              last_day=last_dates[args.last_date])


    if args.model == 0:
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()
    else:
        with cm_effect.models.CMCombined_Final_ICL(data, None) as model:
            model.build_model()

    with model.model:
        model.trace = pm.sample(2000, chains=6, target_accept=0.9)

    np.savetxt(f"double_final/res_model_{args.model}_date_{args.last_date}.csv", model.trace.CMReduction)
