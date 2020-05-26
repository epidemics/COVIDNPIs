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
argparser.add_argument("--g", dest="g", type=float)
argparser.add_argument("--s", dest="nS", type=int)
argparser.add_argument("--c", dest="nC", type=int)
args = argparser.parse_args()

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,floatX=float32,device=gpu"

def unmask_all(d):
    d.Active.mask = False
    d.Confirmed.mask = False
    d.Deaths.mask = False
    d.NewDeaths.mask = False
    d.NewCases.mask = False

def mask_region(d, region, days=14):
    i = d.Rs.index(region)
    c_s = np.nonzero(np.cumsum(d.NewCases.data[i, :] > 0)==days+1)[0][0]
    d_s = np.nonzero(np.cumsum(d.NewDeaths.data[i, :] > 0)==days+1)[0][0]
    print(f"Region {region} masking from day {c_s} and {d_s}")
    d.Active.mask[i,c_s:] = True
    d.Confirmed.mask[i,c_s:] = True
    d.Deaths.mask[i,d_s:] = True
    d.NewDeaths.mask[i,d_s:] = True
    d.NewCases.mask[i,c_s:] = True

if __name__ == "__main__":
    class ResultsObject():
        def __init__(self, indxs, trace):
            self.CMReduction = trace.CMReduction
            self.RegionLogR = trace.RegionLogR[:, indxs]
            self.Z1C = trace.Z1C[:, indxs, :]
            self.Z1D = trace.Z1D[:, indxs, :]
            self.InfectedCases = trace.InfectedCases[:, indxs, :]
            self.InfectedDeaths = trace.InfectedDeaths[:, indxs, :]
            self.ExpectedCases = trace.ExpectedCases[:, indxs, :]
            self.ExpectedDeaths = trace.ExpectedDeaths[:, indxs, :]
            self.Phi = trace.Phi_1

    dp = DataPreprocessor(min_confirmed=100, drop_HS=True)
    data = dp.preprocess_data("notebooks/final_data/data_final.csv")

    HO_rs = ["DE", "PT", "CZ", "PL", "MX", "NL"]
    indxs = [data.Rs.index(rg) for rg in HO_rs]
    unmask_all(data)
    for region in HO_rs:
        mask_region(data, region)

    print(f"Growth Noise {args.g}")
    with cm_effect.models.CMCombined_Final(data, None) as model:
        model.DailyGrowthNoise = args.g
        model.build_model()

    with model.model:
        model.trace = pm.sample(args.nS, chains=args.nC, target_accept=0.95)

    results_obj = ResultsObject(indxs, model.trace)
    pickle.dump(results_obj, open(f"g_results/{args.g}.pkl","wb"))
