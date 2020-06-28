### threading
import os
os.environ["THEANO_FLAGS"] = "OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1"
os.environ["OMP_NUM_THREADS"] =  "1"
os.environ["MKL_NUM_THREADS"] =  "1"
os.environ["OPENBLAS_NUM_THREADS"] =  "1"

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
argparser.add_argument('--rg', nargs='+', dest="rgs", type=str)
argparser.add_argument("--s", dest="nS", type=int)
argparser.add_argument("--c", dest="nC", type=int)
args = argparser.parse_args()

def mask_region(d, region, days=14):
    i = d.Rs.index(region)
    c_s = np.nonzero(np.cumsum(d.NewCases.data[i, :] > 0)==days+1)[0][0]
    d_s = np.nonzero(np.cumsum(d.NewDeaths.data[i, :] > 0)==days+1)[0]
    if len(d_s) > 0:
        d_s = d_s[0]
    else:
        d_s = len(d.Ds)

    d.Active.mask[i,c_s:] = True
    d.Confirmed.mask[i,c_s:] = True
    d.Deaths.mask[i,d_s:] = True
    d.NewDeaths.mask[i,d_s:] = True
    d.NewCases.mask[i,c_s:] = True

if __name__ == "__main__":

    class ResultsObject():
        def __init__(self, indx, trace):
            self.CMReduction = trace.CMReduction
            self.RegionR = trace.RegionR[:, indx]
            self.InfectedCases = trace.InfectedCases[:, indx, :]
            self.InfectedDeaths = trace.InfectedDeaths[:, indx, :]
            self.ExpectedCases = trace.ExpectedCases[:, indx, :]
            self.ExpectedDeaths = trace.ExpectedDeaths[:, indx, :]

    print(args.rgs)
    for rg in args.rgs:
        dp = DataPreprocessor()
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30", schools_unis="whoops")

        data.mask_reopenings()
        mask_region(data, rg)
        indx = data.Rs.index(rg)

        print(f"holdout {rg} w/ {indx}")
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

        with model.model:
            model.trace = pm.sample(1500, tune=500, cores=4, max_treedepth=12)

        results_obj = ResultsObject(indx, model.trace)
        pickle.dump(results_obj, open(f"ho_results_final4/{rg}.pkl","wb"))
