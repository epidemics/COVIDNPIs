"""
:code:`data_merger.py`

This file contains the merge_data function, which takes data from different sources and parses it into our own format.
If you are a user of EpidemicForecasting.org, you will not need to use this function.
"""
import os
import copy

import numpy as np
import pandas as pd
import theano



def _merge_data(data_base_path, region_info, oxcgrt_feature_filter, selected_features_oxcgrt, selected_features_epi,
                ordered_features, output_name='data_final.csv', start_date='2020-02-10', end_date='2020-05-30',
                episet_fname='double_entry_final.csv', oxcgrt_fname='OxCGRT_16620.csv',
                johnhop_fname='johns-hopkins.csv'):
    """
    Merge data into our format.

    This takes data from the Johns-Hopkins Dataset, the EpidemicForecasting.org dataset, the OxCGRT government tracker
    data (https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker).

    Please see notebooks/reproductions/data_merger.ipynb for an example using this file.

    :param data_base_path: path where data csvs are stored.
    :param region_info: array of ('Region Name', 'EpiForecasting Region Code', 'OxCGRT Region Code') tuples
    :param oxcgrt_feature_filter: OxCGRT Filter. List of (Name, [(Index, [Values]]) tuples. NPI name is activated when
                                                for all index, OxCGRT index NPI has a value from Values. e.g., for
                                                ('example', [(0, [1, 5]), (2, [3, 4])]), the example NPI would be active
                                                when OxCGRT NPI 0 has value 1 or 5, and NPI 2 has value 3 or 4. Indices
                                                relate to the selected_features_oxcgrt list.
    :param selected_features_oxcgrt: list of feature names to draw from the OxCGRT dataset.
    :param selected_features_epi: dictionary of features to draw from the epidemicforecasting.org dataset. keys are
                                    names in the csv, values are what to call these values in the merged dataset.
    :param ordered_features: final list of NPIs, using the OxCGRT filter and the names provided in the values of
                             selected_features_epi
    :param output_name: output csv filename
    :param start_date: window of analysis start date
    :param end_date: window of analysis end date
    :param episet_fname: filename for the EpidemicForecasting.org dataset.
    :param oxcgrt_fname: filename for the OxCGRT dataset.
    :param johnhop_fname: filename for the JohnsHopkins dataset.
    """
    Ds = pd.date_range(start=start_date, end=end_date, tz='utc')

    # EpidemicForecasting.org Dataset
    epi_cmset = pd.read_csv(os.path.join(data_base_path, episet_fname), skiprows=1).set_index('Code')
    for c in epi_cmset.columns:
        if 'Code 3' in c or 'comment' in c or 'Oxford' in c or 'Unnamed' in c or 'Person' in c or 'sources' in c \
                or 'oxford' in c or 'Sources' in c:
            del epi_cmset[c]

    region_names = list([x for x, _, _ in region_info])
    regions_epi = list([x for _, x, _ in region_info])
    regions_oxcgrt = list([x for _, _, x in region_info])

    nRs = len(region_names)
    nDs = len(Ds)
    nCMs_epi = len(selected_features_epi.values())
    ActiveCMs_epi = np.zeros((nRs, nCMs_epi, nDs))

    def get_feature_index(col):
        for i, c in enumerate(selected_features_epi.values()):
            if c in col:
                return i

    columns = epi_cmset.iloc[0].index.tolist()
    for r, ccode in enumerate(regions_epi):
        row = epi_cmset.loc[ccode]
        for col in columns:
            if 'Start date' in col:
                f_i = get_feature_index(col)
                on_date = row.loc[col].strip()
                if not on_date == 'no' and not on_date == 'No' and not on_date == 'needs checking' and not on_date == 'Na':
                    on_date = pd.to_datetime(on_date, dayfirst=True)
                    if on_date in Ds:
                        on_loc = Ds.get_loc(on_date)
                        ActiveCMs_epi[r, f_i, on_loc:] = 1

    for r, ccode in enumerate(regions_epi):
        row = epi_cmset.loc[ccode]
        for col in columns:
            if 'End date' in col:
                f_i = get_feature_index(col)
                if not pd.isna(row.loc[col]):
                    off_date = row.loc[col].strip()
                    if not off_date == 'no' and not off_date == 'No' and not off_date == 'After 30 May' and not 'TBD' in off_date and not off_date == 'Na':
                        off_date = pd.to_datetime(off_date, dayfirst=True)
                        if off_date in Ds:
                            off_loc = Ds.get_loc(off_date)
                            ActiveCMs_epi[r, f_i, off_loc:] = 0

    logger_str = '\nCountermeasures: EpidemicForecasting.org           min   ... mean  ... max   ... unique'
    for i, cm in enumerate(selected_features_epi.values()):
        logger_str = f'{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs_epi[:, i, :]):.3f} ... {np.mean(ActiveCMs_epi[:, i, :]):.3f} ... {np.max(ActiveCMs_epi[:, i, :]):.3f} ... {np.unique(ActiveCMs_epi[:, i, :])[:5]}'
    print(logger_str)

    # OxCGRT NPIs
    print('Load OXCGRT')
    unique_missing_countries = []

    def oxcgrt_to_epimodel_index(ind):
        try:
            return regions_epi[regions_oxcgrt.index(ind)]
        except ValueError:
            if ind not in unique_missing_countries:
                unique_missing_countries.append(ind)
            return ind

    data_oxcgrt = pd.read_csv(os.path.join(data_base_path, oxcgrt_fname), index_col='CountryCode')

    columns_to_drop = ['CountryName', 'Date', 'ConfirmedCases', 'ConfirmedDeaths',
                       'StringencyIndex', 'StringencyIndexForDisplay']

    dti = pd.DatetimeIndex(pd.to_datetime(data_oxcgrt['Date'], utc=True, format='%Y%m%d'))
    epi_codes = [oxcgrt_to_epimodel_index(cc) for cc in data_oxcgrt.index.array]
    print(f'Warning: Missing {unique_missing_countries} from epidemicforecasting.org DB which are in OxCGRT')
    data_oxcgrt.index = pd.MultiIndex.from_arrays([epi_codes, dti])

    for col in columns_to_drop:
        del data_oxcgrt[col]

    data_oxcgrt.sort_index()

    regions_epi_filtered = copy.deepcopy(regions_epi)
    regions_epi_filtered.remove('LV')
    regions_epi_filtered.remove('MT')
    data_oxcgrt_filtered = data_oxcgrt.loc[regions_epi_filtered, selected_features_oxcgrt]

    values_to_stack = []
    Ds_l = list(Ds)

    # return data_oxcgrt_filtered

    for c in regions_epi:
        if c in data_oxcgrt_filtered.index:
            v = np.zeros((len(selected_features_oxcgrt), len(Ds)))

            if data_oxcgrt_filtered.loc[c].index[0] in Ds_l:
                x_0 = list(Ds).index(data_oxcgrt_filtered.loc[c].index[0])
            else:
                x_0 = 0

            print(c)
            v[:, x_0:] = data_oxcgrt_filtered.loc[c].loc[Ds[x_0:]].T
            values_to_stack.append(v)
        else:
            print(f'Missing {c} from OxCGRT. Assuming features are 0')
            values_to_stack.append(np.zeros_like(values_to_stack[-1]))

    # this has NaNs in!
    ActiveCMs_temp = np.stack(values_to_stack)
    nRs, _, nDs = ActiveCMs_temp.shape
    nCMs_oxcgrt = len(oxcgrt_feature_filter)
    ActiveCMs_oxcgrt = np.zeros((nRs, nCMs_oxcgrt, nDs))
    oxcgrt_derived_cm_names = [n for n, _ in oxcgrt_feature_filter]

    for r_indx in range(nRs):
        for feature_indx, (_, feature_filter) in enumerate(oxcgrt_feature_filter):
            nConditions = len(feature_filter)
            condition_mat = np.zeros((nConditions, nDs))
            for condition, (row, poss_values) in enumerate(feature_filter):
                row_vals = ActiveCMs_temp[r_indx, row, :]
                # check if feature has any of its possible values
                for value in poss_values:
                    condition_mat[condition, :] += (row_vals == value)
                # if it has any of them, this condition is satisfied
                condition_mat[condition, :] = condition_mat[condition, :] > 0
                # deal with missing data. nan * 0 = nan. Anything else is zero
                condition_mat[condition, :] += (row_vals * 0)
                # we need all conditions to be satisfied, hence a product
            ActiveCMs_oxcgrt[r_indx, feature_indx, :] = (np.prod(condition_mat, axis=0) > 0) + 0 * (
                np.prod(condition_mat, axis=0))

    # now forward fill in missing data!
    for r in range(nRs):
        for c in range(nCMs_oxcgrt):
            for d in range(nDs):
                # if it starts off nan, assume that its zero
                if d == 0 and np.isnan(ActiveCMs_oxcgrt[r, c, d]):
                    ActiveCMs_oxcgrt[r, c, d] = 0
                elif np.isnan(ActiveCMs_oxcgrt[r, c, d]):
                    # if the value is nan, assume it takes the value of the previous day
                    ActiveCMs_oxcgrt[r, c, d] = ActiveCMs_oxcgrt[r, c, d - 1]

    logger_str = '\nCountermeasures: OxCGRT           min   ... mean  ... max   ... unique'
    for i, cm in enumerate(oxcgrt_derived_cm_names):
        logger_str = f'{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs_oxcgrt[:, i, :]):.3f} ... {np.mean(ActiveCMs_oxcgrt[:, i, :]):.3f} ... {np.max(ActiveCMs_oxcgrt[:, i, :]):.3f} ... {np.unique(ActiveCMs_oxcgrt[:, i, :])[:5]}'
    print(logger_str)

    # merge NPIs into one ndarray
    nCMs = len(ordered_features)
    ActiveCMs = np.zeros((nRs, nCMs, nDs))

    for r in range(nRs):
        for f_indx, f in enumerate(ordered_features):
            if f in selected_features_epi.values():
                ActiveCMs[r, f_indx, :] = ActiveCMs_epi[r, list(selected_features_epi.values()).index(f), :]
            else:
                ActiveCMs[r, f_indx, :] = ActiveCMs_oxcgrt[r, oxcgrt_derived_cm_names.index(f), :]

    # [country, CM, day] Which CMs are active, and to what extent
    ActiveCMs = ActiveCMs.astype(theano.config.floatX)
    logger_str = '\nCountermeasures: Combined           min   ... mean  ... max   ... unique'
    for i, cm in enumerate(ordered_features):
        logger_str = f'{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs[:, i, :]):.3f} ... {np.mean(ActiveCMs[:, i, :]):.3f} ... {np.max(ActiveCMs[:, i, :]):.3f} ... {np.unique(ActiveCMs[:, i, :])[:5]}'
    print(logger_str)

    # John hopkins Cases and Deaths Stuff
    johnhop_ds = pd.read_csv(os.path.join(data_base_path, johnhop_fname), index_col=['Code', 'Date'],
                             parse_dates=['Date'], infer_datetime_format=True)
    Confirmed = np.stack([johnhop_ds['Confirmed'].loc[(fc, Ds)] for fc in regions_epi])
    Active = np.stack([johnhop_ds['Active'].loc[(fc, Ds)] for fc in regions_epi])
    Deaths = np.stack([johnhop_ds['Deaths'].loc[(fc, Ds)] for fc in regions_epi])

    columns = ['Country Code', 'Date', 'Region Name', 'Confirmed', 'Active', 'Deaths', *ordered_features]
    df = pd.DataFrame(columns=columns)
    for r_indx, r in enumerate(regions_epi):
        for d_indx, d in enumerate(Ds):
            rows, columns = df.shape
            country_name = region_names[regions_epi.index(r)]
            feature_list = []
            for i in range(len(ordered_features)):
                feature_list.append(ActiveCMs[r_indx, i, d_indx])
            df.loc[rows] = [r, d, country_name, Confirmed[r_indx, d_indx], Active[r_indx, d_indx],
                            Deaths[r_indx, d_indx], *feature_list]

    # save to new csv file!
    df = df.set_index(['Country Code', 'Date'])
    df.to_csv(output_name)
    print(f'Saved final CSV, {output_name}')
