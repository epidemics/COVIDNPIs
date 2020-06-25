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

    d.Active.mask = False
    d.Confirmed.mask = False
    d.Deaths.mask = False
    d.NewDeaths.mask = False
    d.NewCases.mask = False
    d.Active.mask[i,c_s:] = True
    d.Confirmed.mask[i,c_s:] = True
    d.Deaths.mask[i,d_s:] = True
    d.NewDeaths.mask[i,d_s:] = True
    d.NewCases.mask[i,c_s:] = True

if __name__ == "__main__":

    class ResultsObject():
        def __init__(self, indx, trace):
            self.CMReduction = trace.CMReduction
            self.RegionLogR = trace.RegionLogR[:, indx]
            self.Z1C = trace.Z1C[:, indx, :]
            self.Z1D = trace.Z1D[:, indx, :]
            self.InfectedCases = trace.InfectedCases[:, indx, :]
            self.InfectedDeaths = trace.InfectedDeaths[:, indx, :]
            self.ExpectedCases = trace.ExpectedCases[:, indx, :]
            self.ExpectedDeaths = trace.ExpectedDeaths[:, indx, :]

    print(args.rgs)
    for rg in args.rgs:
        dp = DataPreprocessor()
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30", schools_unis="whoops")

        mask_region(data, rg)
        indx = data.Rs.index(rg)

        print(f"holdout {rg} w/ {indx}")
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

        with model.model:
            model.trace = pm.sample(2000, tune=500, cores=4, target_accept=0.9, max_treedepth=10)

        results_obj = ResultsObject(indx, model.trace)
        pickle.dump(results_obj, open(f"ho_results_final/{rg}.pkl","wb"))
