import os

os.environ["THEANO_FLAGS"] = "OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
print("setting environment variables properly now done.")

### imports
import logging
import numpy as np
import copy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from .datapreprocessor import DataPreprocessor
import os
import arviz as az
import matplotlib.pyplot as plt


def generate_out_dir(daily_growth_noise):
    out_dir = 'sensitivity_tests_longer'

    # if region_heldout is not None:
    #    out_dir = out_dir + '_rho' + region_heldout 
    if daily_growth_noise is not None:
        out_dir = out_dir + '_dgn' + str(daily_growth_noise)[2:len(str(daily_growth_noise))]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return out_dir


def save_traces(model, model_type, filename):
    cm_trace = model.trace["CMReduction"]
    np.savetxt(filename, cm_trace)

    if model_type == 'combined_additive':
        cm_base_trace = model.trace["Beta_hat"]
        np.savetxt(filename[0:len(filename) - 4] + '_base.txt', cm_base_trace)

    rhats = calc_trace_statistic(model, 'rhat')
    ess = calc_trace_statistic(model, 'ess')

    np.savetxt(filename[:-4] + "rhats.txt", rhats)
    np.savetxt(filename[:-4] + "ess.txt", ess)


def mask_region(d, region, days=14):
    i = d.Rs.index(region)
    c_s = np.nonzero(np.cumsum(d.NewCases.data[i, :] > 0) == days + 1)[0][0]
    d_s = np.nonzero(np.cumsum(d.NewDeaths.data[i, :] > 0) == days + 1)[0]
    if len(d_s) > 0:
        d_s = d_s[0]
    else:
        d_s = len(d.Ds)

    # mask the right days
    d.Active.mask[i, c_s:] = True
    d.Confirmed.mask[i, c_s:] = True
    d.Deaths.mask[i, d_s:] = True
    d.NewDeaths.mask[i, d_s:] = True
    d.NewCases.mask[i, c_s:] = True


def leavout_cm(data, cm_leavouts, i):
    data_cm_leavout = copy.deepcopy(data)
    print('CM left out: ' + cm_leavouts[i])
    if cm_leavouts[i] == 'None':
        pass
    else:
        data_cm_leavout.ActiveCMs = np.delete(data_cm_leavout.ActiveCMs, i, 1)
        data_cm_leavout.CMs = np.delete(data_cm_leavout.CMs, i)
    return data_cm_leavout


def region_holdout_sensitivity(model_types, regions_heldout=["NL", "PL", "PT", "CZ", "DE", "MX"],
                               daily_growth_noise=None, min_deaths=None, region_var_noise=0.1,
                               data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)

    for region in regions_heldout:
        data = dp.preprocess_data(data_path, "2020-05-30")
        data.mask_reopenings()
        if min_deaths is not None:
            data.filter_region_min_deaths(min_deaths)
        mask_region(data, region)

        for model_type in model_types:
            print('Model: ' + str(model_type))
            print('Heldout Region: ' + str(region))
            if model_type == 'combined':
                with models.CMCombined_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'active':
                with models.CMActive_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'death':
                with models.CMDeath_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_v3':
                with models.CMCombined_Final_V3(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_icl':
                with models.CMCombined_Final_ICL(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_dif':
                with models.CMCombined_Final_DifEffects(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.RegionVariationNoise = region_var_noise
                    model.build_model()
            if model_type == 'combined_no_noise':
                with models.CMCombined_Final_NoNoise(data) as model:
                    model.build_model()
            if model_type == 'combined_icl_no_noise':
                with models.CMCombined_ICL_NoNoise(data) as model:
                    model.build_model()
            if model_type == 'combined_additive':
                with models.CMCombined_Additive(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()

            model.run(2000, tune=500, chains=4, cores=4)
            out_dir = generate_out_dir(daily_growth_noise)
            filename = out_dir + '/regions_heldout_' + region + '_' + model_type + '.txt'
            save_traces(model, model_type, filename)


def cm_leavout_sensitivity(model_types, daily_growth_noise=None, min_deaths=None,
                           region_var_noise=0.1, data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)
    data = dp.preprocess_data(data_path, "2020-05-30")
    data.mask_reopenings()
    if min_deaths is not None:
        data.filter_region_min_deaths(min_deaths)

    cm_leavouts = copy.deepcopy(data.CMs)
    # cm_leavouts.append('None')

    for model_type in model_types:
        for i in range(len(cm_leavouts)):
            data_cm_leavout = leavout_cm(data, cm_leavouts, i)
            print('Model: ' + str(model_type))
            if model_type == 'combined':
                with models.CMCombined_Final(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'active':
                with models.CMActive_Final(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'death':
                with models.CMDeath_Final(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_v3':
                with models.CMCombined_Final_V3(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_icl':
                with models.CMCombined_Final_ICL(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_dif':
                with models.CMCombined_Final_DifEffects(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.RegionVariationNoise = region_var_noise
                    model.build_model()
            if model_type == 'combined_no_noise':
                with models.CMCombined_Final_NoNoise(data_cm_leavout) as model:
                    model.build_model()
            if model_type == 'combined_icl_no_noise':
                with models.CMCombined_ICL_NoNoise(data_cm_leavout) as model:
                    model.build_model()
            if model_type == 'combined_additive':
                with models.CMCombined_Additive(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()

            model.run(2000, tune=500, chains=4, cores=4)
            out_dir = generate_out_dir(daily_growth_noise)
            filename = out_dir + '/cm_leavout_' + model_type + '_' + str(i) + '.txt'
            save_traces(model, model_type, filename)


def cm_leavout_sensitivity(model_types, daily_growth_noise=None, min_deaths=None,
                           region_var_noise=0.1, data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)
    data = dp.preprocess_data(data_path, "2020-05-30")
    data.mask_reopenings()
    if min_deaths is not None:
        data.filter_region_min_deaths(min_deaths)

    cm_leavouts = copy.deepcopy(data.CMs)[:5]
    # cm_leavouts.append('None')

    for model_type in model_types:
        for i in range(len(cm_leavouts)):
            data_cm_leavout = leavout_cm(data, cm_leavouts, i)
            print('Model: ' + str(model_type))
            if model_type == 'combined':
                with models.CMCombined_Final(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'active':
                with models.CMActive_Final(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'death':
                with models.CMDeath_Final(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_v3':
                with models.CMCombined_Final_V3(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_icl':
                with models.CMCombined_Final_ICL(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_dif':
                with models.CMCombined_Final_DifEffects(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.RegionVariationNoise = region_var_noise
                    model.build_model()
            if model_type == 'combined_no_noise':
                with models.CMCombined_Final_NoNoise(data_cm_leavout) as model:
                    model.build_model()
            if model_type == 'combined_icl_no_noise':
                with models.CMCombined_ICL_NoNoise(data_cm_leavout) as model:
                    model.build_model()
            if model_type == 'combined_additive':
                with models.CMCombined_Additive(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()

            model.run(2000, tune=500, chains=4, cores=4)
            out_dir = generate_out_dir(daily_growth_noise)
            filename = out_dir + '/cm_leavout_' + model_type + '_' + str(i) + '.txt'
            save_traces(model, model_type, filename)


def cm_leavout_sensitivity2(model_types, daily_growth_noise=None, min_deaths=None,
                            region_var_noise=0.1, data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)
    data = dp.preprocess_data(data_path, "2020-05-30")
    data.mask_reopenings()
    if min_deaths is not None:
        data.filter_region_min_deaths(min_deaths)

    cm_leavouts = copy.deepcopy(data.CMs)[5:]
    # cm_leavouts.append('None')

    for model_type in model_types:
        for i in range(len(cm_leavouts)):
            data_cm_leavout = leavout_cm(data, cm_leavouts, i + 5)
            print('Model: ' + str(model_type))
            if model_type == 'combined':
                with models.CMCombined_Final(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'active':
                with models.CMActive_Final(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'death':
                with models.CMDeath_Final(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_v3':
                with models.CMCombined_Final_V3(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_icl':
                with models.CMCombined_Final_ICL(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_dif':
                with models.CMCombined_Final_DifEffects(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.RegionVariationNoise = region_var_noise
                    model.build_model()
            if model_type == 'combined_no_noise':
                with models.CMCombined_Final_NoNoise(data_cm_leavout) as model:
                    model.build_model()
            if model_type == 'combined_icl_no_noise':
                with models.CMCombined_ICL_NoNoise(data_cm_leavout) as model:
                    model.build_model()
            if model_type == 'combined_additive':
                with models.CMCombined_Additive(data_cm_leavout) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()

            model.run(2000, tune=500, chains=4, cores=4)
            out_dir = generate_out_dir(daily_growth_noise)
            filename = out_dir + '/cm_leavout_' + model_type + '_' + str(i + 5) + '.txt'
            save_traces(model, model_type, filename)


def cm_prior_sensitivity(model_types, priors=['half_normal', 'wide', "icl"], sigma_wide=10,
                         daily_growth_noise=None, min_deaths=None, region_var_noise=0.1,
                         data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)
    data = dp.preprocess_data(data_path, "2020-05-30")
    data.mask_reopenings()
    if min_deaths is not None:
        data.filter_region_min_deaths(min_deaths)

    for model_type in model_types:
        for prior in priors:
            print('Prior: ' + str(prior))
            print('Model: ' + model_type)
            if model_type == 'combined':
                with models.CMCombined_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    if prior == 'default':
                        model.build_model()
                    if prior == 'wide':
                        model.build_model(cm_prior_sigma=sigma_wide)
                    if prior == 'half_normal':
                        model.build_model(cm_prior='half_normal')
                    if prior == 'icl':
                        model.build_model(cm_prior='icl')
            if model_type == 'active':
                with models.CMActive_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    if prior == 'default':
                        model.build_model()
                    if prior == 'wide':
                        model.build_model(cm_prior_sigma=sigma_wide)
                    if prior == 'half_normal':
                        model.build_model(cm_prior='half_normal')
            if model_type == 'death':
                with models.CMDeath_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    if prior == 'default':
                        model.build_model()
                    if prior == 'wide':
                        model.build_model(cm_prior_sigma=sigma_wide)
                    if prior == 'half_normal':
                        model.build_model(cm_prior='half_normal')
            if model_type == 'combined_v3':
                with models.CMCombined_Final_V3(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    if prior == 'default':
                        model.build_model()
                    if prior == 'wide':
                        model.build_model(cm_prior_sigma=sigma_wide)
                    if prior == 'half_normal':
                        model.build_model(cm_prior='half_normal')
            if model_type == 'combined_icl':
                with models.CMCombined_Final_ICL(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    if prior == 'default':
                        model.build_model()
                    if prior == 'wide':
                        model.build_model(cm_prior_sigma=sigma_wide)
                    if prior == 'half_normal':
                        model.build_model(cm_prior='half_normal')
            if model_type == 'combined_dif':
                with models.CMCombined_Final_DifEffects(data) as model:
                    model.RegionVariationNoise = region_var_noise
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    if prior == 'default':
                        model.build_model()
                    if prior == 'wide':
                        model.build_model(cm_prior_sigma=sigma_wide)
                    if prior == 'half_normal':
                        model.build_model(cm_prior='half_normal')
            if model_type == 'combined_no_noise':
                with models.CMCombined_Final_NoNoise(data) as model:
                    if prior == 'default':
                        model.build_model()
                    if prior == 'wide':
                        model.build_model(cm_prior_sigma=sigma_wide)
                    if prior == 'half_normal':
                        model.build_model(cm_prior='half_normal')
            if model_type == 'combined_icl_no_noise':
                with models.CMCombined_ICL_NoNoise(data) as model:
                    if prior == 'default':
                        model.build_model()
                    if prior == 'wide':
                        model.build_model(cm_prior_sigma=sigma_wide)
                    if prior == 'half_normal':
                        model.build_model(cm_prior='half_normal')
            if model_type == 'combined_additive':
                with models.CMCombined_Additive(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    if prior == 5:
                        model.build_model(cm_prior_conc=5)
                    if prior == 10:
                        model.build_model(cm_prior_conc=10)

            model.run(2000, tune=500, chains=4, cores=4)
            out_dir = generate_out_dir(daily_growth_noise)
            filename = out_dir + '/cm_prior_' + model_type + '_' + str(prior) + '.txt'
            save_traces(model, model_type, filename)


def data_mob_sensitivity(model_types, daily_growth_noise=None, min_deaths=None,
                         region_var_noise=0.1, data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)
    data_mob_no_work = dp.preprocess_data("notebooks/final_data/data_mob_no_work.csv")
    data_mob = dp.preprocess_data("notebooks/final_data/data_mob.csv")

    data_mobility_types = ['no_work', 'rec_work']

    for data_mobility_type in data_mobility_types:
        if data_mobility_type == 'no_work':
            data = copy.deepcopy(data_mob_no_work)
        if data_mobility_type == 'rec_work':
            data = copy.deepcopy(data_mob)
        if min_deaths is not None:
            data.filter_region_min_deaths(min_deaths)
        for model_type in model_types:
            print('Model: ' + str(model_type))
            print('Data Mobility Type: ' + str(data_mobility_type))
            if model_type == 'combined':
                with models.CMCombined_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'active':
                with models.CMActive_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'death':
                with models.CMDeath_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_v3':
                with models.CMCombined_Final_V3(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_icl':
                with models.CMCombined_Final_ICL(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_dif':
                with models.CMCombined_Final_DifEffects(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.RegionVariationNoise = region_var_noise
                    model.build_model()
            if model_type == 'combined_no_noise':
                with models.CMCombined_Final_NoNoise(data) as model:
                    model.build_model()
            if model_type == 'combined_icl_no_noise':
                with models.CMCombined_ICL_NoNoise(data) as model:
                    model.build_model()
            if model_type == 'combined_additive':
                with models.CMCombined_Additive(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()

            model.run(2000, tune=500, chains=4, cores=4)
            out_dir = generate_out_dir(daily_growth_noise)
            filename = out_dir + '/data_mobility_' + data_mobility_type + '_' + model_type + '.txt'
            save_traces(model, model_type, filename)


def data_schools_open_sensitivity(model_types, daily_growth_noise=None, min_deaths=None,
                                  region_var_noise=0.1, data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)
    data = dp.preprocess_data("notebooks/final_data/data_SE_schools_open.csv")
    if min_deaths is not None:
        data.filter_region_min_deaths(min_deaths)

    for model_type in model_types:
        print('Model: ' + str(model_type))
        if model_type == 'combined':
            with models.CMCombined_Final(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.build_model()
        if model_type == 'active':
            with models.CMActive_Final(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.build_model()
        if model_type == 'death':
            with models.CMDeath_Final(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.build_model()
        if model_type == 'combined_v3':
            with models.CMCombined_Final_V3(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.build_model()
        if model_type == 'combined_icl':
            with models.CMCombined_Final_ICL(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.build_model()
        if model_type == 'combined_dif':
            with models.CMCombined_Final_DifEffects(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.RegionVariationNoise = region_var_noise
                model.build_model()
        if model_type == 'combined_no_noise':
            with models.CMCombined_Final_NoNoise(data) as model:
                model.build_model()
        if model_type == 'combined_icl_no_noise':
            with models.CMCombined_ICL_NoNoise(data) as model:
                model.build_model()
        if model_type == 'combined_additive':
            with models.CMCombined_Additive(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.build_model()

        model.run(2000, tune=500, chains=4, cores=4)
        out_dir = generate_out_dir(daily_growth_noise)
        filename = out_dir + '/schools_open_' + model_type + '.txt'
        save_traces(model, model_type, filename)


def daily_growth_noise_sensitivity(model_types, daily_growth_noise=[0.05, 0.1, 0.4],
                                   min_deaths=None, region_var_noise=0.1,
                                   data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)
    data = dp.preprocess_data(data_path, "2020-05-30")
    data.mask_reopenings()
    if min_deaths is not None:
        data.filter_region_min_deaths(min_deaths)

    for i in range(len(daily_growth_noise)):
        for model_type in model_types:
            print('Daily Growth Noise: ' + str(daily_growth_noise[i]))
            print('Model: ' + str(model_type))
            if model_type == 'combined':
                with models.CMCombined_Final(data) as model:
                    model.DailyGrowthNoise = daily_growth_noise[i]
                    model.build_model()
            if model_type == 'active':
                with models.CMActive_Final(data) as model:
                    model.DailyGrowthNoise = daily_growth_noise[i]
                    model.build_model()
            if model_type == 'death':
                with models.CMDeath_Final(data) as model:
                    model.DailyGrowthNoise = daily_growth_noise[i]
                    model.build_model()
            if model_type == 'combined_v3':
                with models.CMCombined_Final_V3(data) as model:
                    model.DailyGrowthNoise = daily_growth_noise[i]
                    model.build_model()
            if model_type == 'combined_icl':
                with models.CMCombined_Final_ICL(data) as model:
                    model.DailyGrowthNoise = daily_growth_noise[i]
                    model.build_model()
            if model_type == 'combined_dif':
                with models.CMCombined_Final_DifEffects(data) as model:
                    model.DailyGrowthNoise = daily_growth_noise[i]
                    model.RegionVariationNoise = region_var_noise
                    model.build_model()
            if model_type == 'combined_additive':
                with models.CMCombined_Additive(data) as model:
                    model.DailyGrowthNoise = daily_growth_noise[i]
                    model.build_model()

            model.run(2000, tune=500, chains=4, cores=4)
            out_dir = generate_out_dir(daily_growth_noise)
            filename = out_dir + '/growth_noise_' + model_type + '_' + str(i) + '.txt'
            save_traces(model, model_type, filename)


def min_num_confirmed_sensitivity(model_types, min_conf_cases=[10, 30, 300, 500],
                                  daily_growth_noise=None, min_deaths=None,
                                  region_var_noise=0.1, data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)

    for model_type in model_types:
        for i in range(len(min_conf_cases)):
            print('Model: ' + str(model_type))
            print('Minimum number of confirmed cases: ' + str(min_conf_cases[i]))
            dp.min_confirmed = min_conf_cases[i]
            data = dp.preprocess_data(data_path, "2020-05-30")
            data.mask_reopenings()
            if min_deaths is not None:
                data.filter_region_min_deaths(min_deaths)

            if model_type == 'active':
                with models.CMActive_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined':
                with models.CMCombined_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_v3':
                with models.CMCombined_Final_V3(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_icl':
                with models.CMCombined_Final_ICL(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_dif':
                with models.CMCombined_Final_DifEffects(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.RegionVariationNoise = region_var_noise
                    model.build_model()
            if model_type == 'combined_no_noise':
                with models.CMCombined_Final_NoNoise(data) as model:
                    model.build_model()
            if model_type == 'combined_icl_no_noise':
                with models.CMCombined_ICL_NoNoise(data) as model:
                    model.build_model()
            if model_type == 'combined_additive':
                with models.CMCombined_Additive(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            model.run(2000, tune=500, chains=4, cores=4)
            out_dir = generate_out_dir(daily_growth_noise)
            filename = out_dir + '/min_confirmed_' + str(model_type) + '_' + str(min_conf_cases[i]) + '.txt'
            save_traces(model, model_type, filename)


def min_num_deaths_sensitivity(model_types, min_deaths_ths=[3, 5, 30, 50],
                               daily_growth_noise=None, min_deaths=None,
                               region_var_noise=0.1, data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)

    for model_type in model_types:
        for i in range(len(min_deaths_ths)):
            print('Model: ' + str(model_type))
            print('Minimum number of death cases: ' + str(min_deaths_ths[i]))
            dp.min_deaths = min_deaths_ths[i]
            data = dp.preprocess_data(data_path, "2020-05-30")
            data.mask_reopenings()
            if min_deaths is not None:
                data.filter_region_min_deaths(min_deaths)

            if model_type == 'active':
                with models.CMActive_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined':
                with models.CMCombined_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_v3':
                with models.CMCombined_Final_V3(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_icl':
                with models.CMCombined_Final_ICL(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_dif':
                with models.CMCombined_Final_DifEffects(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.RegionVariationNoise = region_var_noise
                    model.build_model()
            if model_type == 'combined_no_noise':
                with models.CMCombined_Final_NoNoise(data) as model:
                    model.build_model()
            if model_type == 'combined_icl_no_noise':
                with models.CMCombined_ICL_NoNoise(data) as model:
                    model.build_model()
            if model_type == 'combined_additive':
                with models.CMCombined_Additive(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            model.run(2000, tune=500, chains=4, cores=4)
            out_dir = generate_out_dir(daily_growth_noise)
            filename = out_dir + '/min_deaths_' + str(model_type) + '_' + str(min_deaths_ths[i]) + '.txt'
            save_traces(model, model_type, filename)


def smoothing_sensitivity(model_types, N_days=[1, 3, 7, 15],
                          daily_growth_noise=None, min_deaths=None,
                          region_var_noise=0.1, data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)

    for model_type in model_types:
        for i in range(len(N_days)):
            print('Model: ' + str(model_type))
            print('Minimum number of days: ' + str(N_days[i]))
            dp.N_smooth = N_days[i]
            data = dp.preprocess_data(data_path, "2020-05-30")
            data.mask_reopenings()
            if min_deaths is not None:
                data.filter_region_min_deaths(min_deaths)

            if model_type == 'active':
                with models.CMActive_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined':
                with models.CMCombined_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_v3':
                with models.CMCombined_Final_V3(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_icl':
                with models.CMCombined_Final_ICL(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            if model_type == 'combined_dif':
                with models.CMCombined_Final_DifEffects(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.RegionVariationNoise = region_var_noise
                    model.build_model()
            if model_type == 'combined_no_noise':
                with models.CMCombined_Final_NoNoise(data) as model:
                    model.build_model()
            if model_type == 'combined_icl_no_noise':
                with models.CMCombined_ICL_NoNoise(data) as model:
                    model.build_model()
            if model_type == 'combined_additive':
                with models.CMCombined_Additive(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model()
            model.run(2000, tune=500, chains=4, cores=4)
            out_dir = generate_out_dir(daily_growth_noise)
            filename = out_dir + '/smoothing_' + str(model_type) + '_' + str(N_days[i]) + '.txt'
            save_traces(model, model_type, filename)


def calc_trace_statistic(model, stat_type):
    if stat_type == 'rhat':
        stat = az.rhat(model.trace)
    if stat_type == 'ess':
        stat = az.ess(model.trace, relative=True)
    stat_all = []
    stat_nums = []
    for i in range(len(model.vars)):
        var = model.vars[i]
        if str(var)[-6:-1] == '_log_':
            var = str(var)[0:-6]
        if '_stickbreaking__' in str(var):
            var = str(var)[0:-16]
        if stat[str(var)].size > 1:
            stat_all.append(stat[str(var)].to_dataframe().to_numpy().flatten())
        else:
            stat_nums.append(float(stat[str(var)]))
    stat_all = np.concatenate(np.array(stat_all))
    stat_all = np.concatenate([stat_all, stat_nums])
    return stat_all


def MCMC_stability(model_types, daily_growth_noise=None, min_deaths=None,
                   region_var_noise=0.1, data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)
    data = dp.preprocess_data(data_path, "2020-05-30")
    data.mask_reopenings()
    if min_deaths is not None:
        data.filter_region_min_deaths(min_deaths)

    for model_type in model_types:
        print('Model: ' + str(model_type))
        if model_type == 'combined':
            with models.CMCombined_Final(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.build_model()
        if model_type == 'active':
            with models.CMActive_Final(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.build_model()
        if model_type == 'death':
            with models.CMDeath_Final(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.build_model()
        if model_type == 'combined_v3':
            with models.CMCombined_Final_V3(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.build_model()
        if model_type == 'combined_icl':
            with models.CMCombined_Final_ICL(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.build_model()
        if model_type == 'combined_dif':
            with models.CMCombined_Final_DifEffects(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.RegionVariationNoise = region_var_noise
                model.build_model()
        if model_type == 'combined_no_noise':
            with models.CMCombined_Final_NoNoise(data) as model:
                model.build_model()
        if model_type == 'combined_icl_no_noise':
            with models.CMCombined_ICL_NoNoise(data) as model:
                model.build_model()
        if model_type == 'combined_additive':
            with models.CMCombined_Additive(data) as model:
                if daily_growth_noise is not None:
                    model.DailyGrowthNoise = daily_growth_noise
                model.build_model()

        model.run(2000, tune=500, chains=4, cores=4)
        rhats = calc_trace_statistic(model, 'rhat')
        ess = calc_trace_statistic(model, 'ess')

        out_dir = generate_out_dir(daily_growth_noise)
        np.savetxt(out_dir + '/rhats_' + model_type + '.txt', rhats)
        np.savetxt(out_dir + '/ess_' + model_type + '.txt', ess)
        filename = out_dir + '/default_' + model_type + '.txt'
        save_traces(model, model_type, filename)


def R_hyperprior_mean_sensitivity(model_types, hyperprior_means=[2.5, 4.5],
                                  daily_growth_noise=None, min_deaths=None,
                                  region_var_noise=0.1, data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)
    data = dp.preprocess_data(data_path, "2020-05-30")
    data.mask_reopenings()
    if min_deaths is not None:
        data.filter_region_min_deaths(min_deaths)

    for i in range(len(hyperprior_means)):
        for model_type in model_types:
            print('R Hyperprior mean: ' + str(hyperprior_means[i]))
            print('Model: ' + str(model_type))
            if model_type == 'combined':
                with models.CMCombined_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model(R_hyperprior_mean=hyperprior_means[i])
            if model_type == 'active':
                with models.CMActive_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model(R_hyperprior_mean=hyperprior_means[i])
            if model_type == 'death':
                with models.CMDeath_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model(R_hyperprior_mean=hyperprior_means[i])
            if model_type == 'combined_v3':
                with models.CMCombined_Final_V3(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model(R_hyperprior_mean=hyperprior_means[i])
            if model_type == 'combined_icl':
                with models.CMCombined_Final_ICL(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model(R_hyperprior_mean=hyperprior_means[i])
            if model_type == 'combined_dif':
                with models.CMCombined_Final_DifEffects(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.RegionVariationNoise = region_var_noise
                    model.build_model(R_hyperprior_mean=hyperprior_means[i])
            if model_type == 'combined_no_noise':
                with models.CMCombined_Final_NoNoise(data) as model:
                    model.build_model(R_hyperprior_mean=hyperprior_means[i])
            if model_type == 'combined_icl_no_noise':
                with models.CMCombined_ICL_NoNoise(data) as model:
                    model.build_model(R_hyperprior_mean=hyperprior_means[i])
            if model_type == 'combined_additive':
                with models.CMCombined_Additive(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model(R_hyperprior_mean=hyperprior_means[i])

            model.run(2000, tune=500, chains=4, cores=4)
            out_dir = generate_out_dir(daily_growth_noise)
            filename = out_dir + '/R_hyperprior_' + model_type + '_' + str(i) + '.txt'
            save_traces(model, model_type, filename)


def serial_interval_sensitivity(model_types, serial_interval=[4, 5, 6, 7, 8],
                                daily_growth_noise=None, min_deaths=None,
                                region_var_noise=0.1, data_path="notebooks/double-entry-data/double_entry_final.csv"):
    dp = DataPreprocessor(drop_HS=True)
    data = dp.preprocess_data(data_path, "2020-05-30")
    data.mask_reopenings()
    if min_deaths is not None:
        data.filter_region_min_deaths(min_deaths)

    for i in range(len(serial_interval)):
        for model_type in model_types:
            print('Serial interval mean: ' + str(serial_interval[i]))
            print('Model: ' + str(model_type))
            if model_type == 'combined':
                with models.CMCombined_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model(serial_interval_mean=serial_interval[i])
            if model_type == 'active':
                with models.CMActive_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model(serial_interval_mean=serial_interval[i])
            if model_type == 'death':
                with models.CMDeath_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model(serial_interval_mean=serial_interval[i])
            if model_type == 'combined_v3':
                with models.CMCombined_Final_V3(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model(serial_interval_mean=serial_interval[i])
            if model_type == 'combined_icl':
                with models.CMCombined_Final_ICL(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model(serial_interval_mean=serial_interval[i])
            if model_type == 'combined_dif':
                with models.CMCombined_Final_DifEffects(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.RegionVariationNoise = region_var_noise
                    model.build_model(serial_interval_mean=serial_interval[i])
            if model_type == 'combined_no_noise':
                with models.CMCombined_Final_NoNoise(data) as model:
                    model.build_model(serial_interval_mean=serial_interval[i])
            if model_type == 'combined_icl_no_noise':
                with models.CMCombined_ICL_NoNoise(data) as model:
                    model.build_model(serial_interval_mean=serial_interval[i])
            if model_type == 'combined_additive':
                with models.CMCombined_Additive(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.build_model(serial_interval_mean=serial_interval[i])

            model.run(2000, tune=500, chains=4, cores=4)
            out_dir = generate_out_dir(daily_growth_noise)
            filename = out_dir + '/serial_int_' + model_type + '_SI' + str(serial_interval[i]) + '.txt'
            save_traces(model, model_type, filename)


######## delay mean #############

def gamma_mu_cov_to_shape_scale(mu, cov):
    shape = 1 / (cov ** 2)
    scale = mu * (cov ** 2)
    return shape, scale


def calc_shifted_delay_mean_death(mean_shift):
    nRVs = int(9e7)
    shp1, scl1 = gamma_mu_cov_to_shape_scale(5.1 + mean_shift, 0.86)
    shp2, scl2 = gamma_mu_cov_to_shape_scale(17.8 + mean_shift, 0.45)
    samples = np.random.gamma(shape=shp1, scale=scl1, size=nRVs) + np.random.gamma(shape=shp2, scale=scl2, size=nRVs)
    bins = np.arange(-1, 64.0)
    bins[2:] += 0.5
    # print(f"Binned As {bins}")
    n, _, _ = plt.hist(samples, bins);
    delay_prob = n / np.sum(n)
    # print(f"Expectation: {np.sum([(i) * delay_prob[i] for i in range(64)])}")
    # print(f"True Mean: {np.mean(samples)}")
    # print(f"Delay Prob: {delay_prob}")
    return delay_prob


def calc_shifted_delay_mean_conf(mean_shift):
    m = 5.25 + mean_shift
    r = 1.57
    p = m / (m + r)

    nRVs = int(9e7)
    shp1, scl1 = gamma_mu_cov_to_shape_scale(5.1 + mean_shift, 0.86)
    samples = np.random.gamma(shape=shp1, scale=scl1, size=nRVs) + np.random.negative_binomial(r, (1 - p), size=nRVs)
    bins = np.arange(-1, 32.0)
    bins[2:] += 0.5
    # print(f"Binned As {bins}")
    n, _, _ = plt.hist(samples, bins)
    delay_prob = n / np.sum(n)
    # print(f"Expectation: {np.sum([(i) * delay_prob[i] for i in range(64)])}")
    # print(f"True Mean: {np.mean(samples)}")
    # print(f"Delay Prob: {delay_prob}")
    return delay_prob


def vary_delay_mean(model, mean_shift, model_type):
    '''to use for deaths model or active cases model'''
    default_mean = np.trapz(model.DelayProb * (np.arange(len(model.DelayProb))))
    print('Default mean: ' + str(default_mean))
    if (mean_shift + default_mean) == default_mean:
        pass
    else:
        if model_type == 'active':
            delay_prob = calc_shifted_delay_mean_conf(mean_shift)
        if model_type == 'death':
            delay_prob = calc_shifted_delay_mean_death(mean_shift)
        model.DelayProb = delay_prob
    return model


def vary_delay_mean_confirmed(model, mean_shift):
    '''to use for combined model'''
    default_mean = np.trapz(model.DelayProbCases * (np.arange(len(model.DelayProbCases[0]))))
    print('Default mean conf: ' + str(default_mean))
    if (mean_shift + default_mean) == default_mean:
        pass
    else:
        delay_prob = calc_shifted_delay_mean_conf(mean_shift)
        model.DelayProbCases = delay_prob
    return model


def vary_delay_mean_death(model, mean_shift):
    '''to use for combined model'''
    default_mean = np.trapz(model.DelayProbDeaths * (np.arange(len(model.DelayProbDeaths[0]))))
    print('Default mean death: ' + str(default_mean))
    if (mean_shift + default_mean) == default_mean:
        pass
    else:
        delay_prob = calc_shifted_delay_mean_death(mean_shift)
        model.DelayProbDeaths = delay_prob
    return model


def delay_mean_sensitivity(model_types, mean_shift=[-2, -1, 1, 2], daily_growth_noise=None,
                           min_deaths=None, region_var_noise=0.1,
                           data_path="notebooks/double-entry-data/double_entry_final.csv"):
    '''
    mean_shift shifts the mean of the underlying distributions that will combine
    to form the delay distrobution. So the resulting total delay distribution
    will be shifted more than the specified mean_shift. 
    '''

    samples = 1000
    chains = 8
    cores = 8

    delay_probs_conf_combined = []
    delay_probs_death_combined = []
    delay_probs_death = []
    delay_probs_active = []

    delay_probs_conf_combined = []
    delay_probs_death_combined = []
    delay_probs_death = []
    delay_probs_active = []

    dp = DataPreprocessor(drop_HS=True)
    data = dp.preprocess_data(data_path, "2020-05-30")
    data.mask_reopenings()
    if min_deaths is not None:
        data.filter_region_min_deaths(min_deaths)
    out_dir = generate_out_dir(daily_growth_noise)

    for model_type in model_types:
        plt.figure()
        if model_type == 'combined':
            # for combined model vary confirmed and deaths delay
            for i in range(len(mean_shift)):
                print('Delay Mean Shift: ' + str(mean_shift[i]))
                print('Model: ' + str(model_type))
                with models.CMCombined_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model = vary_delay_mean_confirmed(model, mean_shift[i])
                    delay_probs_conf_combined.append(model.DelayProbCases)
                    model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_confirmed_combined_' + str(i) + '.txt'
                save_traces(model, model_type, filename)

                with models.CMCombined_Final(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model = vary_delay_mean_death(model, mean_shift[i])
                    delay_probs_death_combined.append(model.DelayProbDeaths)
                    model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_death_combined_' + str(i) + '.txt'
                save_traces(model, model_type, filename)
        elif model_type == 'combined_v3':
            # for combined model vary confirmed and deaths delay
            for i in range(len(mean_shift)):
                print('Delay Mean Shift: ' + str(mean_shift[i]))
                print('Model: ' + str(model_type))
                with models.CMCombined_Final_V3(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model = vary_delay_mean_confirmed(model, mean_shift[i])
                    # delay_probs_conf_combined.append(model.DelayProbCases) 
                    model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_confirmed_combined_v3_' + str(i) + '.txt'
                save_traces(model, model_type, filename)

                with models.CMCombined_Final_V3(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model = vary_delay_mean_death(model, mean_shift[i])
                    # delay_probs_death_combined.append(model.DelayProbDeaths) 
                    model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_death_combined_v3_' + str(i) + '.txt'
                save_traces(model, model_type, filename)
        elif model_type == 'combined_icl':
            # for combined model vary confirmed and deaths delay
            for i in range(len(mean_shift)):
                print('Delay Mean Shift: ' + str(mean_shift[i]))
                print('Model: ' + str(model_type))
                with models.CMCombined_Final_ICL(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model = vary_delay_mean_confirmed(model, mean_shift[i])
                    # delay_probs_conf_combined.append(model.DelayProbCases) 
                    model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_confirmed_combined_icl_' + str(i) + '.txt'
                save_traces(model, model_type, filename)

                with models.CMCombined_Final_ICL(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model = vary_delay_mean_death(model, mean_shift[i])
                    # delay_probs_death_combined.append(model.DelayProbDeaths) 
                    model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_death_combined_icl_' + str(i) + '.txt'
                save_traces(model, model_type, filename)
        elif model_type == 'combined_dif':
            # for combined model vary confirmed and deaths delay
            for i in range(len(mean_shift)):
                print('Delay Mean Shift: ' + str(mean_shift[i]))
                print('Model: ' + str(model_type))
                with models.CMCombined_Final_DifEffects(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.RegionVariationNoise = region_var_noise
                    model = vary_delay_mean_confirmed(model, mean_shift[i])
                    # delay_probs_conf_combined.append(model.DelayProbCases) 
                    model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_confirmed_combined_dif_' + str(i) + '.txt'
                save_traces(model, model_type, filename)

                with models.CMCombined_Final_DifEffects(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model.RegionVariationNoise = region_var_noise
                    model = vary_delay_mean_death(model, mean_shift[i])
                    # delay_probs_death_combined.append(model.DelayProbDeaths) 
                    model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_death_combined_dif_' + str(i) + '.txt'
                save_traces(model, model_type, filename)
        elif model_type == 'combined_no_noise':
            # for combined model vary confirmed and deaths delay
            for i in range(len(mean_shift)):
                print('Delay Mean Shift: ' + str(mean_shift[i]))
                print('Model: ' + str(model_type))
                with models.CMCombined_Final_NoNoise(data) as model:
                    model = vary_delay_mean_confirmed(model, mean_shift[i])
                    # delay_probs_conf_combined.append(model.DelayProbCases) 
                    model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_confirmed_combined_no_noise_' + str(i) + '.txt'
                save_traces(model, model_type, filename)
                with models.CMCombined_Final_NoNoise(data) as model:
                    model = vary_delay_mean_death(model, mean_shift[i])
                    # delay_probs_death_combined.append(model.DelayProbDeaths) 
                    model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_death_combined_no_noise_' + str(i) + '.txt'
                save_traces(model, model_type, filename)
        elif model_type == 'combined_additive':
            # for combined model vary confirmed and deaths delay
            for i in range(len(mean_shift)):
                print('Delay Mean Shift: ' + str(mean_shift[i]))
                print('Model: ' + str(model_type))
                with models.CMCombined_Additive(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model = vary_delay_mean_confirmed(model, mean_shift[i])
                    # delay_probs_conf_combined.append(model.DelayProbCases) 
                    model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_confirmed_combined_additive_' + str(i) + '.txt'
                save_traces(model, model_type, filename)

                with models.CMCombined_Additive(data) as model:
                    if daily_growth_noise is not None:
                        model.DailyGrowthNoise = daily_growth_noise
                    model = vary_delay_mean_death(model, mean_shift[i])
                    # delay_probs_death_combined.append(model.DelayProbDeaths) 
                    model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_death_combined_additive_' + str(i) + '.txt'
                save_traces(model, model_type, filename)
        else:
            # for other models there is only one delay mean
            for i in range(len(mean_shift)):
                print('Delay Mean Shift: ' + str(mean_shift[i]))
                print('Model: ' + str(model_type))
                if model_type == 'active':
                    with models.CMActive_Final(data) as model:
                        # plt.plot(model.DelayProb, color='k', label='default')
                        if daily_growth_noise is not None:
                            model.DailyGrowthNoise = daily_growth_noise
                        model = vary_delay_mean(model, mean_shift[i], model_type)
                        delay_probs_active.append(model.DelayProb)
                        model.build_model()
                if model_type == 'death':
                    with models.CMDeath_Final(data) as model:
                        # plt.plot(model.DelayProb, color='k', label='default')
                        if daily_growth_noise is not None:
                            model.DailyGrowthNoise = daily_growth_noise
                        model = vary_delay_mean(model, mean_shift[i], model_type)
                        delay_probs_death.append(model.DelayProb)
                        model.build_model()
                if model_type == 'combined_icl_no_noise':
                    with models.CMCombined_ICL_NoNoise(data) as model:
                        # plt.plot(model.DelayProb, color='k', label='default')
                        model = vary_delay_mean_death(model, mean_shift[i])
                        delay_probs_death.append(model.DelayProbDeaths)
                        model.build_model()
                model.run(2000, tune=500, chains=4, cores=4)
                filename = out_dir + '/delay_mean_' + model_type + '_' + str(i) + '.txt'
                save_traces(model, model_type, filename)
