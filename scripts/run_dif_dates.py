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

    last_dates = ["2020-04-25", "2020-05-05", "2020-05-15", "2020-05-25", "2020-05-30"]
    dp = DataPreprocessor()
    data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv",
                              last_day=last_dates[args.last_date])
    data.mask_reopenings()

    if args.model == 0:
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model(serial_interval_mean=6.7, serial_interval_sigma=2.1)
    elif args.model == 1:
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model(serial_interval_mean=5.1, serial_interval_sigma=1.8)
    elif args.model == 2:
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model(serial_interval_mean=6.68, serial_interval_sigma=4.88)
    elif args.model == 3:
        with cm_effect.models.CMCombined_Final_Reset1(data, None) as model:
            model.build_model(serial_interval_mean=6.68, serial_interval_sigma=4.88)
    elif args.model == 4:
        with cm_effect.models.CMCombined_Final_Reset2(data, None) as model:
            model.build_model(serial_interval_mean=6.68, serial_interval_sigma=4.88)

    with model.model:
        model.trace = pm.sample(1000, cores=4, chains=4, target_accept=0.9, max_treedepth=10)

    np.savetxt(f"double_fff/res_model_{args.model}_date_{args.last_date}.csv", model.trace.CMReduction)
