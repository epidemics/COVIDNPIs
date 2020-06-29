### threading
import os
os.environ["THEANO_FLAGS"] = "OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

### Initial imports
import logging
import copy
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
argparser.add_argument("--exp", dest="exp", type=int)
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

    dp = DataPreprocessor()
    exp_num = args.exp

    print(f"running exp {exp_num}")
    # structural sensitivity
    if exp_num == 1:
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30", schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Additive(data, None) as model:
            model.build_model()

    elif exp_num == 2:
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final_V3(data, None) as model:
            model.build_model()

    elif exp_num == 3:
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final_DifEffects(data, None) as model:
            model.build_model()

    elif exp_num == 4:
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final_ICL(data, None) as model:
            model.build_model()

    # OxCGRT Checks
    elif exp_num == 5:
        dp.drop_features = ["Public Transport Limited",
                              "Internal Movement Limited",
                              "Public Information Campaigns", "Symptomatic Testing"]

        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    elif exp_num == 6:
        dp.drop_features = ["Travel Screen/Quarantine", "Travel Bans",
                              "Internal Movement Limited",
                              "Public Information Campaigns", "Symptomatic Testing"]

        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    elif exp_num == 7:
        dp.drop_features = ["Travel Screen/Quarantine", "Travel Bans", "Public Transport Limited",
                              "Public Information Campaigns", "Symptomatic Testing"]

        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    elif exp_num == 8:
        dp.drop_features = ["Travel Screen/Quarantine", "Travel Bans", "Public Transport Limited",
                              "Internal Movement Limited", "Symptomatic Testing"]

        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    elif exp_num == 9:
        dp.drop_features = ["Travel Screen/Quarantine", "Travel Bans", "Public Transport Limited",
                              "Internal Movement Limited"]

        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    elif exp_num == 10:
        dp.drop_features.append("Mobility - retail and rec")
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final_mob.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    elif exp_num == 11:
        # mobility 2
        dp.drop_features.append("Mobility - workplace")
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final_mob.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    elif exp_num == 12:
        # mobility 3
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final_mob.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    # any NPI active
    elif exp_num == 13:
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")

        major_interventions = ["School Closure", "Stay Home Order", "Some Businesses Suspended",  "University Closure",
                               "Most Businesses Suspended", "Gatherings <10", "Gatherings <1000", "Gatherings <100"]

        nRs, nCMs, nDs = data.ActiveCMs.shape

        ActiveCMs = np.zeros((nRs, nCMs + 1, nDs))
        ActiveCMs[:, :-1, :] = data.ActiveCMs

        maj_indxs = np.array([data.CMs.index(x) for x in major_interventions])

        for r in range(nRs):
            maj_active = np.sum(data.ActiveCMs[r, maj_indxs, :], axis=0)
            # bonus NPI is **any** major NPI is active.
            ActiveCMs[r, -1, :] = maj_active > 0

        data.CMs = [*data.CMs, "Any NPI Active"]
        data.ActiveCMs = ActiveCMs

        data.mask_reopenings()
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    # NPI timing
    elif exp_num == 14:
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")

        major_interventions = ["School Closure", "Stay Home Order", "Some Businesses Suspended", "University Closure",
                               "Most Businesses Suspended", "Gatherings <10", "Gatherings <1000", "Gatherings <100"]
        minor_interventions = ["Mask Wearing"]

        ActiveCMs = copy.deepcopy(data.ActiveCMs)

        maj_indxs = np.array([data.CMs.index(x) for x in major_interventions])
        min_indxs = np.array([data.CMs.index(x) for x in minor_interventions])

        nRs, nCMs, nDs = ActiveCMs.shape

        maj_mat = np.zeros((len(major_interventions), len(major_interventions)))
        min_mat = np.zeros((len(minor_interventions), len(minor_interventions)))

        for r in range(nRs):
            maj_active = np.sum(data.ActiveCMs[r, maj_indxs, :], axis=0)
            for i in range(len(major_interventions)):
                ActiveCMs[r, i, :] = maj_active > i
            min_active = np.sum(data.ActiveCMs[r, min_indxs, :], axis=0)
            for i in range(len(minor_interventions)):
                ActiveCMs[r, i + len(major_interventions), :] = min_active > i

        data.CMs = [*[f"Major {i + 1}" for i in range(len(major_interventions))],
                    *[f"Minor {i + 1}" for i in range(len(minor_interventions))]]
        data.ActiveCMs = ActiveCMs

        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    # Different Delays
    elif exp_num == 15:
        dp.drop_features = ["Travel Screen/Quarantine", "Travel Bans", "Public Transport Limited",
                            "Internal Movement Limited"]
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMCombined_Final_DifDelays(data, None) as model:
            model.build_model()

    # delayed schools and universities
    elif exp_num == 16:
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()
        n_delay = 6
        to_delay_index = [data.CMs.index(cm) for cm in ["School Closure", "University Closure"]]
        active_cms = copy.deepcopy(data.ActiveCMs)
        data.ActiveCMs[:, to_delay_index, n_delay:] = active_cms[:, to_delay_index, :-n_delay]
        data.ActiveCMs[:, to_delay_index, :n_delay] = 0
        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    # aggregated holdouts
    elif exp_num == 17:
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        # mask last 20 days everywhere
        for rg in data.Rs:
            mask_region(data, rg)

        # and mask earlier if needed
        data.mask_reopenings(n_extra=20)

        with cm_effect.models.CMCombined_Final(data, None) as model:
            model.build_model()

    elif exp_num == 18:
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMActive_Final(data, None) as model:
            model.build_model()

    elif exp_num == 19:
        data = dp.preprocess_data("notebooks/double-entry-data/double_entry_final.csv", last_day="2020-05-30",
                                  schools_unis="whoops")
        data.mask_reopenings()

        with cm_effect.models.CMDeath_Final(data, None) as model:
            model.build_model()


    with model.model:
        model.trace = pm.sample(2000, tune=500, cores=4, chains=4, max_treedepth=12)

    out_dir = "additional_exps"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pickle.dump(model.trace, open(f"additional_exps/exp_{exp_num}.pkl","wb"))
