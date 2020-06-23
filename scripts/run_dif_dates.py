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

    last_dates = ["2020-04-25", "2020-05-05", "2020-05-15", "2020-05-25", None]
    dp = DataPreprocessor()
    data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv",
                              last_day=last_dates[args.last_date], merge_schools_unis=False)
    data.mask_reopenings()

    if args.model == 0:
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()
    else:
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model(serial_interval_mean=5.1, serial_interval_sigma=1.8)

    with model.model:
        model.trace = pm.sample(1000, cores=4, chains=4, target_accept=0.9)

    np.savetxt(f"double_fff/res_model_{args.model}_date_{args.last_date}.csv", model.trace.CMReduction)
