### threading
import os

os.environ['THEANO_FLAGS'] = 'OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

### Initial imports
import logging
import copy
import numpy as np
import pymc3 as pm
import seaborn as sns

sns.set_style('ticks')

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from epimodel.preprocessing.data_preprocessor import preprocess_data
from epimodel import EpidemiologicalParameters, DefaultModel
from epimodel.pymc3_models.legacy import CMCombined_FinalLegacy, CMCombined_FinalLegacyFixedDispersion, \
    CMCombined_FinalLegacyLognorm, CMCombined_FinalLegacyAltSize, CMCombined_FinalLegacyLessNoise
from epimodel.pymc3_models.models import DefaultModelFixedDispersion, DefaultModelLognorm, DefaultModelPoissonOutput

import argparse
import pickle
import time

argparser = argparse.ArgumentParser()
argparser.add_argument('--exp', dest='exp', type=int)
args = argparser.parse_args()

if __name__ == '__main__':

    GI_MEAN_OLD = 6.67
    GI_SD_OLD = 2.37

    DelayProbCases = np.array([0., 0.0252817, 0.03717965, 0.05181224, 0.06274125,
                               0.06961334, 0.07277174, 0.07292397, 0.07077184, 0.06694868,
                               0.06209945, 0.05659917, 0.0508999, 0.0452042, 0.03976573,
                               0.03470891, 0.0299895, 0.02577721, 0.02199923, 0.01871723,
                               0.01577148, 0.01326564, 0.01110783, 0.00928827, 0.0077231,
                               0.00641162, 0.00530572, 0.00437895, 0.00358801, 0.00295791,
                               0.0024217, 0.00197484])

    DelayProbCases_OLD = DelayProbCases.reshape((1, DelayProbCases.size))

    DelayProbDeaths = np.array([0.00000000e+00, 2.24600347e-06, 3.90382088e-05, 2.34307085e-04,
                                7.83555003e-04, 1.91221622e-03, 3.78718437e-03, 6.45923913e-03,
                                9.94265709e-03, 1.40610714e-02, 1.86527920e-02, 2.34311421e-02,
                                2.81965055e-02, 3.27668001e-02, 3.68031574e-02, 4.03026198e-02,
                                4.30521951e-02, 4.50637136e-02, 4.63315047e-02, 4.68794406e-02,
                                4.67334059e-02, 4.59561441e-02, 4.47164503e-02, 4.29327455e-02,
                                4.08614522e-02, 3.85082076e-02, 3.60294203e-02, 3.34601703e-02,
                                3.08064505e-02, 2.81766028e-02, 2.56165924e-02, 2.31354369e-02,
                                2.07837267e-02, 1.86074383e-02, 1.65505661e-02, 1.46527043e-02,
                                1.29409383e-02, 1.13695920e-02, 9.93233881e-03, 8.66063386e-03,
                                7.53805464e-03, 6.51560047e-03, 5.63512264e-03, 4.84296166e-03,
                                4.14793478e-03, 3.56267297e-03, 3.03480656e-03, 2.59406730e-03,
                                2.19519042e-03, 1.85454286e-03, 1.58333238e-03, 1.33002321e-03,
                                1.11716435e-03, 9.35360376e-04, 7.87780158e-04, 6.58601602e-04,
                                5.48147154e-04, 4.58151351e-04, 3.85878963e-04, 3.21623249e-04,
                                2.66129174e-04, 2.21364768e-04, 1.80736566e-04, 1.52350196e-04])
    DelayProbDeaths_OLD = DelayProbDeaths.reshape((1, DelayProbDeaths.size))

    ep = EpidemiologicalParameters()
    DelayProbCases_NEW, DelayProbDeaths_NEW = ep.generate_reporting_and_fatality_delays(with_noise=False)
    GI_MEAN_NEW, GI_SD_NEW = ep.generate_gi(with_noise=False)

    data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30')
    data.mask_reopenings()
    exp_num = args.exp

    print(f'running exp {exp_num}')

    if exp_num == 0:
        with CMCombined_FinalLegacy(data) as model:
            model.build_model()

    elif exp_num == 1:
        with CMCombined_FinalLegacy(data) as model:
            model.build_model(serial_interval_mean=GI_MEAN_NEW, serial_interval_sigma=GI_SD_NEW)

    elif exp_num == 2:
        with CMCombined_FinalLegacy(data) as model:
            model.DelayProbCases = DelayProbCases_NEW
            model.build_model(serial_interval_mean=GI_MEAN_NEW, serial_interval_sigma=GI_SD_NEW)

    elif exp_num == 3:
        with CMCombined_FinalLegacy(data) as model:
            model.DelayProbCases = DelayProbCases_NEW
            model.DelayProbDeaths = DelayProbDeaths_NEW
            model.build_model(serial_interval_mean=GI_MEAN_NEW, serial_interval_sigma=GI_SD_NEW)

    elif exp_num == 4:
        with DefaultModel(data) as model:
            model.build_model(fatality_delay=DelayProbDeaths_NEW, reporting_delay=DelayProbCases_NEW,
                              generation_interval_mean=GI_MEAN_OLD, generation_interval_sigma=GI_SD_OLD,
                              cm_prior='normal', cm_prior_scale=0.2)

    elif exp_num == 5:
        with DefaultModel(data) as model:
            model.build_model(fatality_delay=DelayProbDeaths_OLD, reporting_delay=DelayProbCases_NEW,
                              generation_interval_mean=GI_MEAN_OLD, generation_interval_sigma=GI_SD_OLD,
                              cm_prior='normal', cm_prior_scale=0.2)

    elif exp_num == 6:
        with DefaultModel(data) as model:
            model.build_model(fatality_delay=DelayProbDeaths_OLD, reporting_delay=DelayProbCases_OLD,
                              generation_interval_mean=GI_MEAN_OLD, generation_interval_sigma=GI_SD_OLD,
                              cm_prior='normal', cm_prior_scale=0.2)

    elif exp_num == 7:
        with DefaultModelFixedDispersion(data) as model:
            model.build_model(fatality_delay=DelayProbDeaths_OLD, reporting_delay=DelayProbCases_OLD,
                              generation_interval_mean=GI_MEAN_OLD, generation_interval_sigma=GI_SD_OLD,
                              cm_prior='normal', cm_prior_scale=0.2)

    elif exp_num == 8:
        data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30',
                               mask_zero_cases=True, mask_zero_deaths=True)
        data.mask_reopenings()
        with DefaultModelLognorm(data) as model:
            model.build_model(fatality_delay=DelayProbDeaths_OLD, reporting_delay=DelayProbCases_OLD,
                              generation_interval_mean=GI_MEAN_OLD, generation_interval_sigma=GI_SD_OLD,
                              cm_prior='normal', cm_prior_scale=0.2)
    elif exp_num == 9:
        with CMCombined_FinalLegacyFixedDispersion(data) as model:
            model.build_model()

    elif exp_num == 10:
        data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30',
                               mask_zero_cases=True, mask_zero_deaths=True)
        data.mask_reopenings()
        with CMCombined_FinalLegacyLognorm(data) as model:
            model.build_model()

    elif exp_num == 11:
        with CMCombined_FinalLegacyAltSize(data) as model:
            model.build_model()

    elif exp_num == 12:
        data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30',
                               smoothing=1)
        data.mask_reopenings()
        with CMCombined_FinalLegacy(data) as model:
            model.build_model()

    elif exp_num == 13:
        data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30',
                               schools_unis='one_and')
        data.mask_reopenings()
        with CMCombined_FinalLegacy(data) as model:
            model.build_model()

    elif exp_num == 14:
        with CMCombined_FinalLegacyLessNoise(data) as model:
            model.build_model()

    elif exp_num == 15:
        with DefaultModel(data) as model:
            model.build_model(fatality_delay=DelayProbDeaths_NEW, reporting_delay=DelayProbCases_NEW,
                              generation_interval_mean=GI_MEAN_NEW, generation_interval_sigma=GI_SD_NEW,
                              cm_prior='normal', cm_prior_scale=0.2)

    elif exp_num == 16:
        with DefaultModelPoissonOutput(data) as model:
            model.build_model(fatality_delay=DelayProbDeaths_NEW, reporting_delay=DelayProbCases_NEW,
                              generation_interval_mean=GI_MEAN_NEW, generation_interval_sigma=GI_SD_NEW,
                              cm_prior='normal', cm_prior_scale=0.2)

    elif exp_num == 17:
        with DefaultModelPoissonOutput(data) as model:
            model.build_model(fatality_delay=DelayProbDeaths_NEW, reporting_delay=DelayProbCases_NEW,
                              generation_interval_mean=GI_MEAN_NEW, generation_interval_sigma=GI_SD_NEW,
                              cm_prior='reparam', cm_prior_scale=0.2)

    elif exp_num == 18:
        with DefaultModel(data) as model:
            model.build_model(fatality_delay=DelayProbDeaths_NEW, reporting_delay=DelayProbCases_NEW,
                              generation_interval_mean=GI_MEAN_NEW, generation_interval_sigma=GI_SD_NEW,
                              cm_prior='reparam', cm_prior_scale=0.2)

    elif exp_num == 19:
        data = preprocess_data('notebooks/double-entry-data/double_entry_final.csv', last_day='2020-05-30',
                               smoothing=1)
        data.mask_reopenings()
        with DefaultModelFixedDispersion(data) as model:
            model.build_model(fatality_delay=DelayProbDeaths_NEW, reporting_delay=DelayProbCases_NEW,
                              generation_interval_mean=GI_MEAN_NEW, generation_interval_sigma=GI_SD_NEW,
                              cm_prior='normal', cm_prior_scale=0.2, disp=0.2)

    time_start = time.time()
    with model.model:
        model.trace = pm.sample(1000, tune=500, cores=4, chains=4, max_treedepth=14, target_accept=0.95)
    time_end = time.time()

    model.trace.time_elapsed = time_end - time_start

    out_dir = 'bug_exps'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pickle.dump(model.trace, open(f'{out_dir}/exp_{exp_num}.pkl', 'wb'))
