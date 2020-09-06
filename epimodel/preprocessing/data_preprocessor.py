"""
Data Preprocessing

Contains preprocess_data function
"""
import copy

import numpy as np
import pandas as pd
import scipy.signal as ss
import theano

from .preprocessed_data import PreprocessedData


def preprocess_data(data_path, last_day=None, schools_unis='two_separate', drop_features=None, min_confirmed=100,
                    min_deaths=10, smoothing=1, mask_zero_deaths=False, mask_zero_cases=False):
    """
    Preprocess data .csv file, in our post-merge format, with different options.

    :param data_path: Path of .csv file to process.
    :param last_day: Last day of window to analysis to use e.g. str '2020-05-30'. If None (default), go to the last day
                     in the .csv file.
    :param schools_unis: how to process schools and unis. Options are:
                            - two_xor. One xor feature, one and feature.
                            - two_separate. One schools feature, one university feature.
                            - one_tiered. One tiered feature. 0 if none active, 0.5 if either active, 1 if both active.
                            - one_and. One feature, 1 if both active.
    :param drop_features: list of strs, names of NPI features to drop. Defaults to all NPIs not collected by the
                            EpidemicForecasting.org team.
    :param min_confirmed: confirmed cases threshold, below which new (daily) cases are ignored. 
    :param min_deaths: deaths threshold, below which new (daily) deaths are ignored. 
    :param smoothing: number of days over which to smooth. This should be an odd number. If 1, no smoothing occurs.
    :param mask_zero_deaths: bool, whether to ignore (i.e., mask) days with zero deaths.
    :param mask_zero_cases: bool, whether to ignore (i.e., mask) days with zero cases.
    :return: PreprocessedData object.
    """

    # load data from our csv
    df = pd.read_csv(data_path, parse_dates=["Date"], infer_datetime_format=True).set_index(
        ["Country Code", "Date"])

    # handle custom last day of analysis
    if last_day is None:
        Ds = list(df.index.levels[1])
    else:
        Ds = list(df.index.levels[1])
        last_ts = pd.to_datetime(last_day, utc=True)
        Ds = Ds[:(1 + Ds.index(last_ts))]

    nDs = len(Ds)

    all_rs = list([r for r, _ in df.index])
    regions = list(df.index.levels[0])
    locations = [all_rs.index(r) for r in regions]
    sorted_regions = [r for l, r in sorted(zip(locations, regions))]
    nRs = len(sorted_regions)
    region_names = copy.deepcopy(sorted_regions)
    region_full_names = df.loc[region_names]["Region Name"]

    if drop_features is None:
        # note: these features are taken from the OxCGRT Dataset.
        drop_features = ['Travel Screen/Quarantine', 'Travel Bans', 'Public Transport Limited',
                         'Internal Movement Limited', 'Public Information Campaigns', 'Symptomatic Testing']

    for f in drop_features:
        print(f'Dropping NPI {f}')
        df = df.drop(f, axis=1)

    # pull data
    CMs = list(df.columns[4:])
    nCMs = len(CMs)

    ActiveCMs = np.zeros((nRs, nCMs, nDs))
    Confirmed = np.zeros((nRs, nDs))
    Deaths = np.zeros((nRs, nDs))
    Active = np.zeros((nRs, nDs))
    NewDeaths = np.zeros((nRs, nDs))
    NewCases = np.zeros((nRs, nDs))

    for r_i, r in enumerate(sorted_regions):
        region_names[r_i] = df.loc[(r, Ds[0])]['Region Name']
        for d_i, d in enumerate(Ds):
            Confirmed[r_i, d_i] = df.loc[(r, d)]['Confirmed']
            Deaths[r_i, d_i] = df.loc[(r, d)]['Deaths']
            Active[r_i, d_i] = df.loc[(r, d)]['Active']

            ActiveCMs[r_i, :, :] = df.loc[r].loc[Ds][CMs].values.T

    # compute new (daily) cases, after using thresholds
    Confirmed[Confirmed < min_confirmed] = np.nan
    Deaths[Deaths < min_deaths] = np.nan
    NewCases[:, 1:] = (Confirmed[:, 1:] - Confirmed[:, :-1])
    NewDeaths[:, 1:] = (Deaths[:, 1:] - Deaths[:, :-1])
    NewDeaths[NewDeaths < 0] = 0
    NewCases[NewCases < 0] = 0

    NewCases[np.isnan(NewCases)] = 0
    NewDeaths[np.isnan(NewDeaths)] = 0

    if smoothing != 1:
        print('Smoothing')

        # bulk smooth
        SmoothedNewCases = np.around(
            ss.convolve2d(NewCases, 1 / smoothing * np.ones(shape=(1, smoothing)), boundary='symm',
                          mode='same'))
        SmoothedNewDeaths = np.around(
            ss.convolve2d(NewDeaths, 1 / smoothing * np.ones(shape=(1, smoothing)), boundary="symm",
                          mode='same'))

        # correct for specific regions
        for r in range(nRs):
            # if the country has too few deaths, ignore
            if Deaths[r, -1] < 50:
                print(f'Note: did not smooth deaths in {region_names[r]}')
                SmoothedNewDeaths[r, :] = NewDeaths[r, :]

        NewCases = SmoothedNewCases
        NewDeaths = SmoothedNewDeaths

    print('Masking invalid values')
    if mask_zero_deaths:
        NewDeaths[NewDeaths < 1] = np.nan
    else:
        NewDeaths[NewDeaths < 0] = np.nan

    if mask_zero_cases:
        NewCases[NewCases < 1] = np.nan
    else:
        NewCases[NewCases < 0] = np.nan

    Confirmed = np.ma.masked_invalid(Confirmed.astype(theano.config.floatX))
    Active = np.ma.masked_invalid(Active.astype(theano.config.floatX))
    Deaths = np.ma.masked_invalid(Deaths.astype(theano.config.floatX))
    NewDeaths = np.ma.masked_invalid(NewDeaths.astype(theano.config.floatX))
    NewCases = np.ma.masked_invalid(NewCases.astype(theano.config.floatX))

    # handle schools and universities
    if schools_unis == 'two_xor':
        school_index = CMs.index('School Closure')
        university_index = CMs.index('University Closure')

        ActiveCMs_final = copy.deepcopy(ActiveCMs)
        ActiveCMs_final[:, school_index, :] = np.logical_and(ActiveCMs[:, university_index, :],
                                                             ActiveCMs[:, school_index, :])
        ActiveCMs_final[:, university_index, :] = np.logical_xor(ActiveCMs[:, university_index, :],
                                                                 ActiveCMs[:, school_index, :])
        ActiveCMs = ActiveCMs_final
        CMs[school_index] = 'School and University Closure'
        CMs[university_index] = 'Schools xor University Closure'
    elif schools_unis == 'one_tiered':
        school_index = CMs.index('School Closure')
        university_index = CMs.index('University Closure')

        ActiveCMs_final = copy.deepcopy(ActiveCMs)
        ActiveCMs_final[:, school_index, :] = np.logical_and(ActiveCMs[:, university_index, :],
                                                             ActiveCMs[:, school_index, :]) + 0.5 * np.logical_xor(
            ActiveCMs[:, university_index, :],
            ActiveCMs[:, school_index, :])

        ActiveCMs = np.delete(ActiveCMs_final, university_index, axis=1)
        CMs.remove('University Closure')
    elif schools_unis == 'two_separate':
        # don't need to do anything for this!
        pass
    elif schools_unis == 'one_and':
        school_index = CMs.index('School Closure')
        university_index = CMs.index('University Closure')
        ActiveCMs_final = copy.deepcopy(ActiveCMs)
        ActiveCMs_final[:, school_index, :] = np.logical_and(ActiveCMs[:, university_index, :],
                                                             ActiveCMs[:, school_index, :])
        ActiveCMs = np.delete(ActiveCMs_final, university_index, axis=1)
        CMs[school_index] = 'School and University Closure'
        CMs.remove('University Closure')

    return PreprocessedData(Active,
                            Confirmed,
                            ActiveCMs,
                            CMs,
                            sorted_regions,
                            Ds,
                            Deaths,
                            NewDeaths,
                            NewCases,
                            region_full_names)
