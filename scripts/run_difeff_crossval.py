### Initial imports
import os
import numpy as np
import pymc3 as pm

os.environ["THEANO_FLAGS"] = "OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from datapreprocessor import DataPreprocessor
import argparse
import pickle

argparser = argparse.ArgumentParser()
argparser.add_argument("--g", dest="growth_noise", type=float)
argparser.add_argument("--a", dest="country_noise", type=float)
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


class ResultsObject():
    def __init__(self, indxs, trace):
        if "CMReduction" in trace.varnames:
            self.CMReduction = trace.CMReduction

        if "RegionLogR" in trace.varnames:
            self.RegionLogR = trace.RegionLogR[:, indxs]

        if "InfectedCases" in trace.varnames:
            self.InfectedCases = trace.InfectedCases[:, indxs, :]

        if "InfectedDeaths" in trace.varnames:
            self.InfectedDeaths = trace.InfectedDeaths[:, indxs, :]

        if "ExpectedCases" in trace.varnames:
            self.ExpectedCases = trace.ExpectedCases[:, indxs, :]

        if "ExpectedDeaths" in trace.varnames:
            self.ExpectedDeaths = trace.ExpectedDeaths[:, indxs, :]

        if "Phi" in trace.varnames:
            self.Phi = trace.Phi
        elif "Phi_1" in trace.varnames:
            self.Phi = trace.Phi_1


if __name__ == "__main__":
    folds = [['FR', 'GR', 'NL', 'BA', 'LV'],
             ['SE', 'DE', 'LT', 'MY', 'BG'],
             ['FI', 'DK', 'CZ', 'RS', 'BE'],
             ['NO', 'SK', 'IL', 'CH', 'ES'],
             ['ZA', 'MX', 'IT', 'IE', 'GE'],
             ['RO', 'PL', 'MA', 'HU', 'SI'],
             ['NZ', 'SG', 'PT', 'HR', 'EE']]

    eval_fold = ['AL', 'AT', 'GB', 'AD', 'IS', 'MT']

    dp = DataPreprocessor()
    for fold_i, fold in enumerate(folds):
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        r_is = []
        for rg in fold:
            c_s, d_s = mask_region(data, rg)
            r_is.append(data.Rs.index(rg))

        with models.CMCombined_Final_DifEffects(data, None) as model:
            model.DailyGrowthNoise = args.growth_noise
            model.RegionVariationNoise = args.country_noise
            model.build_model()

        with model.model:
            model.trace = pm.sample(2000, cores=4, chains=4, max_treedepth=12)

        results_obj = ResultsObject(r_is, model.trace)
        out_dir = "diffeff_crossval"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        pickle.dump(results_obj,
                    open(os.path.join(out_dir, f'{args.country_noise}-{args.growth_noise}-f{fold_i}.pkl'), 'wb'))
