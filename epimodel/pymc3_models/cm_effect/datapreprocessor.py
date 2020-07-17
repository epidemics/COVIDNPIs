import copy

import pandas as pd
import numpy as np
import theano
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as ss

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fp2 = FontProperties(fname=r"../../fonts/Font Awesome 5 Free-Solid-900.otf")


class DataMerger():
    def __init__(self, params_dict=None, *args, **kwargs):
        self.start_date = "2020-2-10"
        self.end_date = "2020-04-05"

        # rather than being final this will probability want to be min max
        self.min_final_num_active_cases = 100
        self.min_num_active_mask = 10
        self.min_num_confirmed_mask = 10

        self.episet_fname = "countermeasures_pretty120520.csv"
        # self.oxcgrt_fname = "OxCGRT_latest.csv"
        self.oxcgrt_fname = "OxCGRT_16620.csv"
        self.johnhop_fname = "johns-hopkins.csv"

        # load parameters, first from dictionary and then from kwargs
        if params_dict is not None:
            for key, value in params_dict.items():
                setattr(self, key, value)

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def merge_data(
            self,
            data_base_path,
            region_info,
            oxcgrt_feature_info,
            selected_features_oxcgrt,
            selected_features_epi,
            ordered_features,
            output_name="data_final.csv"
    ):
        # at the moment only features from the 0-1 countermeasures dataset
        Ds = pd.date_range(start=self.start_date, end=self.end_date, tz="utc")
        self.Ds = Ds

        # OUR STUFF
        epi_cmset = pd.read_csv(os.path.join(data_base_path, self.episet_fname)).rename(columns=selected_features_epi,
                                                                                        errors="raise").set_index(
            "Code")

        region_names = list([x for x, _, _ in region_info])
        regions_epi = list([x for _, x, _ in region_info])
        regions_oxcgrt = list([x for _, _, x in region_info])

        nRs = len(region_names)
        nDs = len(Ds)
        nCMs_epi = len(selected_features_epi.values())
        ActiveCMs_epi = np.zeros((nRs, nCMs_epi, nDs))

        for r, ccode in enumerate(regions_epi):
            for f, feature_name in enumerate(selected_features_epi.values()):
                on_date = epi_cmset.loc[ccode][feature_name].strip()
                if not on_date == "no" and not on_date == "No":
                    on_date = pd.to_datetime(on_date, dayfirst=True)
                    on_loc = Ds.get_loc(on_date)
                    ActiveCMs_epi[r, f, on_loc:] = 1

        logger_str = "\nCountermeasures: EpidemicForecasting.org           min   ... mean  ... max   ... unique"
        for i, cm in enumerate(selected_features_epi.values()):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs_epi[:, i, :]):.3f} ... {np.mean(ActiveCMs_epi[:, i, :]):.3f} ... {np.max(ActiveCMs_epi[:, i, :]):.3f} ... {np.unique(ActiveCMs_epi[:, i, :])[:5]}"
        logger.info(logger_str)

        # OXCGRT STUFF
        logger.info("Load OXCGRT")
        unique_missing_countries = []

        def oxcgrt_to_epimodel_index(ind):
            try:
                return regions_epi[regions_oxcgrt.index(ind)]
            except ValueError:
                if ind not in unique_missing_countries:
                    unique_missing_countries.append(ind)
                return ind

        data_oxcgrt = pd.read_csv(os.path.join(data_base_path, self.oxcgrt_fname), index_col="CountryCode")

        columns_to_drop = ["CountryName", "Date", "ConfirmedCases", "ConfirmedDeaths",
                           "StringencyIndex", "StringencyIndexForDisplay"]

        dti = pd.DatetimeIndex(pd.to_datetime(data_oxcgrt["Date"], utc=True, format="%Y%m%d"))
        epi_codes = [oxcgrt_to_epimodel_index(cc) for cc in data_oxcgrt.index.array]
        logger.warning(f"Missing {unique_missing_countries} from epidemicforecasting.org DB which are in OxCGRT")
        data_oxcgrt.index = pd.MultiIndex.from_arrays([epi_codes, dti])

        for col in columns_to_drop:
            del data_oxcgrt[col]

        data_oxcgrt.sort_index()

        data_oxcgrt_filtered = data_oxcgrt.loc[regions_epi, selected_features_oxcgrt]

        values_to_stack = []
        Ds_l = list(Ds)

        for c in regions_epi:
            if c in data_oxcgrt_filtered.index:
                v = np.zeros((len(selected_features_oxcgrt), len(Ds)))

                if data_oxcgrt_filtered.loc[c].index[0] in Ds_l:
                    x_0 = list(Ds).index(data_oxcgrt_filtered.loc[c].index[0])
                else:
                    x_0 = 0

                v[:, x_0:] = data_oxcgrt_filtered.loc[c].loc[Ds[x_0:]].T
                values_to_stack.append(v)
            else:
                logger.info(f"Missing {c} from OXCGRT. Assuming features are 0")
                values_to_stack.append(np.zeros_like(values_to_stack[-1]))

        # this has NaNs in!
        ActiveCMs_temp = np.stack(values_to_stack)
        nRs, _, nDs = ActiveCMs_temp.shape
        nCMs_oxcgrt = len(oxcgrt_feature_info)
        ActiveCMs_oxcgrt = np.zeros((nRs, nCMs_oxcgrt, nDs))
        oxcgrt_derived_cm_names = [n for n, _ in oxcgrt_feature_info]

        for r_indx in range(nRs):
            for feature_indx, (_, feature_filter) in enumerate(oxcgrt_feature_info):
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

        logger_str = "\nCountermeasures: OxCGRT           min   ... mean  ... max   ... unique"
        for i, cm in enumerate(oxcgrt_derived_cm_names):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs_oxcgrt[:, i, :]):.3f} ... {np.mean(ActiveCMs_oxcgrt[:, i, :]):.3f} ... {np.max(ActiveCMs_oxcgrt[:, i, :]):.3f} ... {np.unique(ActiveCMs_oxcgrt[:, i, :])[:5]}"
        logger.info(logger_str)

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
        logger_str = "\nCountermeasures: Combined           min   ... mean  ... max   ... unique"
        for i, cm in enumerate(ordered_features):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs[:, i, :]):.3f} ... {np.mean(ActiveCMs[:, i, :]):.3f} ... {np.max(ActiveCMs[:, i, :]):.3f} ... {np.unique(ActiveCMs[:, i, :])[:5]}"
        logger.info(logger_str)

        # Johnhopkins Stuff
        # TODO - this is out of date. to fix.
        johnhop_ds = pd.read_csv(os.path.join(data_base_path, self.johnhop_fname), index_col=["Code", "Date"],
                                 parse_dates=["Date"], infer_datetime_format=True)
        Confirmed = np.stack([johnhop_ds["Confirmed"].loc[(fc, Ds)] for fc in regions_epi])
        Active = np.stack([johnhop_ds["Active"].loc[(fc, Ds)] for fc in regions_epi])
        Deaths = np.stack([johnhop_ds["Deaths"].loc[(fc, Ds)] for fc in regions_epi])

        columns = ["Country Code", "Date", "Region Name", "Confirmed", "Active", "Deaths", *ordered_features]
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
        df = df.set_index(["Country Code", "Date"])
        df.to_csv(output_name)
        logger.info("Saved final CSV")


class DataMergerDoubleEntry():
    def __init__(self, params_dict=None, *args, **kwargs):
        self.start_date = "2020-2-10"
        self.end_date = "2020-05-30"

        # rather than being final this will probability want to be min max
        self.min_final_num_active_cases = 100
        self.min_num_active_mask = 10
        self.min_num_confirmed_mask = 10

        self.episet_fname = "double_entry_final.csv"
        # self.oxcgrt_fname = "OxCGRT_latest.csv"
        self.oxcgrt_fname = "OxCGRT_16620.csv"
        self.johnhop_fname = "johns-hopkins.csv"

        # load parameters, first from dictionary and then from kwargs
        if params_dict is not None:
            for key, value in params_dict.items():
                setattr(self, key, value)

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def merge_data(
            self,
            data_base_path,
            region_info,
            oxcgrt_feature_info,
            selected_features_oxcgrt,
            selected_features_epi,
            ordered_features,
            output_name="data_final.csv"
    ):
        # at the moment only features from the 0-1 countermeasures dataset
        Ds = pd.date_range(start=self.start_date, end=self.end_date, tz="utc")
        self.Ds = Ds

        # OUR STUFF

        epi_cmset = pd.read_csv(os.path.join(data_base_path, self.episet_fname), skiprows=1).set_index("Code")
        for c in epi_cmset.columns:
            if "Code 3" in c or "comment" in c or "Oxford" in c or "Unnamed" in c or "Person" in c or "sources" in c or "oxford" in c:
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
                if "Start date" in col:
                    f_i = get_feature_index(col)
                    on_date = row.loc[col].strip()
                    if not on_date == "no" and not on_date == "No" and not on_date == "needs checking" and not on_date == "Na":
                        on_date = pd.to_datetime(on_date, dayfirst=True)
                        if on_date in Ds:
                            on_loc = Ds.get_loc(on_date)
                            ActiveCMs_epi[r, f_i, on_loc:] = 1

        for r, ccode in enumerate(regions_epi):
            row = epi_cmset.loc[ccode]
            for col in columns:
                if "End date" in col:
                    f_i = get_feature_index(col)
                    if not pd.isna(row.loc[col]):
                        off_date = row.loc[col].strip()
                        if not off_date == "no" and not off_date == "No" and not off_date == "After 30 May" and not "TBD" in off_date and not off_date == "Na":
                            off_date = pd.to_datetime(off_date, dayfirst=True)
                            if off_date in Ds:
                                off_loc = Ds.get_loc(off_date)
                                ActiveCMs_epi[r, f_i, off_loc:] = 0

        logger_str = "\nCountermeasures: EpidemicForecasting.org           min   ... mean  ... max   ... unique"
        for i, cm in enumerate(selected_features_epi.values()):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs_epi[:, i, :]):.3f} ... {np.mean(ActiveCMs_epi[:, i, :]):.3f} ... {np.max(ActiveCMs_epi[:, i, :]):.3f} ... {np.unique(ActiveCMs_epi[:, i, :])[:5]}"
        logger.info(logger_str)

        # OXCGRT STUFF
        logger.info("Load OXCGRT")
        unique_missing_countries = []

        def oxcgrt_to_epimodel_index(ind):
            try:
                return regions_epi[regions_oxcgrt.index(ind)]
            except ValueError:
                if ind not in unique_missing_countries:
                    unique_missing_countries.append(ind)
                return ind

        data_oxcgrt = pd.read_csv(os.path.join(data_base_path, self.oxcgrt_fname), index_col="CountryCode")

        columns_to_drop = ["CountryName", "Date", "ConfirmedCases", "ConfirmedDeaths",
                           "StringencyIndex", "StringencyIndexForDisplay"]

        dti = pd.DatetimeIndex(pd.to_datetime(data_oxcgrt["Date"], utc=True, format="%Y%m%d"))
        epi_codes = [oxcgrt_to_epimodel_index(cc) for cc in data_oxcgrt.index.array]
        logger.warning(f"Missing {unique_missing_countries} from epidemicforecasting.org DB which are in OxCGRT")
        data_oxcgrt.index = pd.MultiIndex.from_arrays([epi_codes, dti])

        for col in columns_to_drop:
            del data_oxcgrt[col]

        data_oxcgrt.sort_index()

        data_oxcgrt_filtered = data_oxcgrt.loc[regions_epi, selected_features_oxcgrt]

        values_to_stack = []
        Ds_l = list(Ds)

        for c in regions_epi:
            if c in data_oxcgrt_filtered.index:
                v = np.zeros((len(selected_features_oxcgrt), len(Ds)))

                if data_oxcgrt_filtered.loc[c].index[0] in Ds_l:
                    x_0 = list(Ds).index(data_oxcgrt_filtered.loc[c].index[0])
                else:
                    x_0 = 0

                v[:, x_0:] = data_oxcgrt_filtered.loc[c].loc[Ds[x_0:]].T
                values_to_stack.append(v)
            else:
                logger.info(f"Missing {c} from OXCGRT. Assuming features are 0")
                values_to_stack.append(np.zeros_like(values_to_stack[-1]))

        # this has NaNs in!
        ActiveCMs_temp = np.stack(values_to_stack)
        nRs, _, nDs = ActiveCMs_temp.shape
        nCMs_oxcgrt = len(oxcgrt_feature_info)
        ActiveCMs_oxcgrt = np.zeros((nRs, nCMs_oxcgrt, nDs))
        oxcgrt_derived_cm_names = [n for n, _ in oxcgrt_feature_info]

        for r_indx in range(nRs):
            for feature_indx, (_, feature_filter) in enumerate(oxcgrt_feature_info):
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

        logger_str = "\nCountermeasures: OxCGRT           min   ... mean  ... max   ... unique"
        for i, cm in enumerate(oxcgrt_derived_cm_names):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs_oxcgrt[:, i, :]):.3f} ... {np.mean(ActiveCMs_oxcgrt[:, i, :]):.3f} ... {np.max(ActiveCMs_oxcgrt[:, i, :]):.3f} ... {np.unique(ActiveCMs_oxcgrt[:, i, :])[:5]}"
        logger.info(logger_str)

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
        logger_str = "\nCountermeasures: Combined           min   ... mean  ... max   ... unique"
        for i, cm in enumerate(ordered_features):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs[:, i, :]):.3f} ... {np.mean(ActiveCMs[:, i, :]):.3f} ... {np.max(ActiveCMs[:, i, :]):.3f} ... {np.unique(ActiveCMs[:, i, :])[:5]}"
        logger.info(logger_str)

        # Johnhopkins Stuff
        # TODO - this is out of date. to fix.
        johnhop_ds = pd.read_csv(os.path.join(data_base_path, self.johnhop_fname), index_col=["Code", "Date"],
                                 parse_dates=["Date"], infer_datetime_format=True)
        Confirmed = np.stack([johnhop_ds["Confirmed"].loc[(fc, Ds)] for fc in regions_epi])
        Active = np.stack([johnhop_ds["Active"].loc[(fc, Ds)] for fc in regions_epi])
        Deaths = np.stack([johnhop_ds["Deaths"].loc[(fc, Ds)] for fc in regions_epi])

        columns = ["Country Code", "Date", "Region Name", "Confirmed", "Active", "Deaths", *ordered_features]
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
        df = df.set_index(["Country Code", "Date"])
        df.to_csv(output_name)
        logger.info("Saved final CSV")

class DataMergerDoubleEntryWithMobility():
    def __init__(self, params_dict=None, *args, **kwargs):
        self.start_date = "2020-2-10"
        self.end_date = "2020-05-30"

        # rather than being final this will probability want to be min max
        self.min_final_num_active_cases = 100
        self.min_num_active_mask = 10
        self.min_num_confirmed_mask = 10

        self.episet_fname = "double_entry_final.csv"
        # self.oxcgrt_fname = "OxCGRT_latest.csv"
        self.oxcgrt_fname = "OxCGRT_16620.csv"
        self.johnhop_fname = "johns-hopkins.csv"
        self.mobility_fname = "google_mobility_june23.csv"

        # load parameters, first from dictionary and then from kwargs
        if params_dict is not None:
            for key, value in params_dict.items():
                setattr(self, key, value)

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def merge_data(
            self,
            data_base_path,
            region_info,
            oxcgrt_feature_info,
            selected_features_oxcgrt,
            selected_features_epi,
            ordered_features,
            output_name="data_final.csv"
    ):
        # at the moment only features from the 0-1 countermeasures dataset
        Ds = pd.date_range(start=self.start_date, end=self.end_date, tz="utc")
        self.Ds = Ds

        # OUR STUFF

        epi_cmset = pd.read_csv(os.path.join(data_base_path, self.episet_fname), skiprows=1).set_index("Code")
        for c in epi_cmset.columns:
            if "Code 3" in c or "comment" in c or "Oxford" in c or "Unnamed" in c or "Person" in c or "sources" in c or "oxford" in c:
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
                if "Start date" in col:
                    f_i = get_feature_index(col)
                    on_date = row.loc[col].strip()
                    if not on_date == "no" and not on_date == "No" and not on_date == "needs checking" and not on_date == "Na":
                        on_date = pd.to_datetime(on_date, dayfirst=True)
                        if on_date in Ds:
                            on_loc = Ds.get_loc(on_date)
                            ActiveCMs_epi[r, f_i, on_loc:] = 1

        for r, ccode in enumerate(regions_epi):
            row = epi_cmset.loc[ccode]
            for col in columns:
                if "End date" in col:
                    f_i = get_feature_index(col)
                    if not pd.isna(row.loc[col]):
                        off_date = row.loc[col].strip()
                        if not off_date == "no" and not off_date == "No" and not off_date == "After 30 May" and not "TBD" in off_date and not off_date == "Na":
                            off_date = pd.to_datetime(off_date, dayfirst=True)
                            if off_date in Ds:
                                off_loc = Ds.get_loc(off_date)
                                ActiveCMs_epi[r, f_i, off_loc:] = 0

        logger_str = "\nCountermeasures: EpidemicForecasting.org           min   ... mean  ... max   ... unique"
        for i, cm in enumerate(selected_features_epi.values()):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs_epi[:, i, :]):.3f} ... {np.mean(ActiveCMs_epi[:, i, :]):.3f} ... {np.max(ActiveCMs_epi[:, i, :]):.3f} ... {np.unique(ActiveCMs_epi[:, i, :])[:5]}"
        logger.info(logger_str)

        # OXCGRT STUFF
        logger.info("Load OXCGRT")
        unique_missing_countries = []

        def oxcgrt_to_epimodel_index(ind):
            try:
                return regions_epi[regions_oxcgrt.index(ind)]
            except ValueError:
                if ind not in unique_missing_countries:
                    unique_missing_countries.append(ind)
                return ind

        data_oxcgrt = pd.read_csv(os.path.join(data_base_path, self.oxcgrt_fname), index_col="CountryCode")

        columns_to_drop = ["CountryName", "Date", "ConfirmedCases", "ConfirmedDeaths",
                           "StringencyIndex", "StringencyIndexForDisplay"]

        dti = pd.DatetimeIndex(pd.to_datetime(data_oxcgrt["Date"], utc=True, format="%Y%m%d"))
        epi_codes = [oxcgrt_to_epimodel_index(cc) for cc in data_oxcgrt.index.array]
        logger.warning(f"Missing {unique_missing_countries} from epidemicforecasting.org DB which are in OxCGRT")
        data_oxcgrt.index = pd.MultiIndex.from_arrays([epi_codes, dti])

        for col in columns_to_drop:
            del data_oxcgrt[col]

        data_oxcgrt.sort_index()

        data_oxcgrt_filtered = data_oxcgrt.loc[regions_epi, selected_features_oxcgrt]

        values_to_stack = []
        Ds_l = list(Ds)

        for c in regions_epi:
            if c in data_oxcgrt_filtered.index:
                v = np.zeros((len(selected_features_oxcgrt), len(Ds)))

                if data_oxcgrt_filtered.loc[c].index[0] in Ds_l:
                    x_0 = list(Ds).index(data_oxcgrt_filtered.loc[c].index[0])
                else:
                    x_0 = 0

                v[:, x_0:] = data_oxcgrt_filtered.loc[c].loc[Ds[x_0:]].T
                values_to_stack.append(v)
            else:
                logger.info(f"Missing {c} from OXCGRT. Assuming features are 0")
                values_to_stack.append(np.zeros_like(values_to_stack[-1]))

        # this has NaNs in!
        ActiveCMs_temp = np.stack(values_to_stack)
        nRs, _, nDs = ActiveCMs_temp.shape
        nCMs_oxcgrt = len(oxcgrt_feature_info)
        ActiveCMs_oxcgrt = np.zeros((nRs, nCMs_oxcgrt, nDs))
        oxcgrt_derived_cm_names = [n for n, _ in oxcgrt_feature_info]

        for r_indx in range(nRs):
            for feature_indx, (_, feature_filter) in enumerate(oxcgrt_feature_info):
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

        logger_str = "\nCountermeasures: OxCGRT           min   ... mean  ... max   ... unique"
        for i, cm in enumerate(oxcgrt_derived_cm_names):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs_oxcgrt[:, i, :]):.3f} ... {np.mean(ActiveCMs_oxcgrt[:, i, :]):.3f} ... {np.max(ActiveCMs_oxcgrt[:, i, :]):.3f} ... {np.unique(ActiveCMs_oxcgrt[:, i, :])[:5]}"
        logger.info(logger_str)

        # mobility_data
        df = pd.read_csv("../../data/google_mobility_june23.csv", infer_datetime_format=True).set_index(["Country Code", "Date"])
        del df["Unnamed: 0"]

        nCMs = len(df.columns)
        cols_mob = df.columns

        ActiveCMs_mob = np.zeros((nRs, nCMs, nDs))
        for r_i, r in enumerate(regions_epi):
            if r in df.index:
                for d_i, d in enumerate(Ds):
                    if str(d) in df.index.get_level_values(1):
                        for cm in range(nCMs):
                            ActiveCMs_mob[r_i, cm, d_i] = df.loc[r, str(d)][cm]

        nCMs = len(ordered_features)
        ActiveCMs = np.zeros((nRs, nCMs, nDs))

        for r in range(nRs):
            for f_indx, f in enumerate(ordered_features):
                if f in selected_features_epi.values():
                    ActiveCMs[r, f_indx, :] = ActiveCMs_epi[r, list(selected_features_epi.values()).index(f), :]
                elif "Mobility" in f:
                    ActiveCMs[r, f_indx, :] = ActiveCMs_mob[r, list(cols_mob).index(f), :]
                else:
                    ActiveCMs[r, f_indx, :] = ActiveCMs_oxcgrt[r, oxcgrt_derived_cm_names.index(f), :]

        # [country, CM, day] Which CMs are active, and to what extent
        ActiveCMs = ActiveCMs.astype(theano.config.floatX)
        logger_str = "\nCountermeasures: Combined           min   ... mean  ... max   ... unique"
        for i, cm in enumerate(ordered_features):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs[:, i, :]):.3f} ... {np.mean(ActiveCMs[:, i, :]):.3f} ... {np.max(ActiveCMs[:, i, :]):.3f} ... {np.unique(ActiveCMs[:, i, :])[:5]}"
        logger.info(logger_str)

        # Johnhopkins Stuff
        johnhop_ds = pd.read_csv(os.path.join(data_base_path, self.johnhop_fname), index_col=["Code", "Date"],
                                 parse_dates=["Date"], infer_datetime_format=True)
        Confirmed = np.stack([johnhop_ds["Confirmed"].loc[(fc, Ds)] for fc in regions_epi])
        Active = np.stack([johnhop_ds["Active"].loc[(fc, Ds)] for fc in regions_epi])
        Deaths = np.stack([johnhop_ds["Deaths"].loc[(fc, Ds)] for fc in regions_epi])

        columns = ["Country Code", "Date", "Region Name", "Confirmed", "Active", "Deaths", *ordered_features]
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
        df = df.set_index(["Country Code", "Date"])
        df.to_csv(output_name)
        logger.info("Saved final CSV")

class DataPreprocessor():
    def __init__(self, *args, **kwargs):
        self.min_confirmed = 100
        self.min_deaths = 10

        self.mask_zero_deaths = False
        self.mask_zero_cases = False

        self.smooth = True
        self.N_smooth = 5

        self.drop_features = ["Travel Screen/Quarantine", "Travel Bans", "Public Transport Limited",
                              "Internal Movement Limited",
                              "Public Information Campaigns", "Symptomatic Testing"]

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def generate_params_dict(self):
        return {
            "min_final_num_active_cases": self.min_final_num_active_cases,
            "confirmed_mask": self.min_num_active_mask,
        }

    def preprocess_data(self, data_path, last_day=None, schools_unis="default"):
        # load data
        df = pd.read_csv(data_path, parse_dates=["Date"], infer_datetime_format=True).set_index(
            ["Country Code", "Date"])

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

        for f in self.drop_features:
            logger.info(f"Dropping {f}")
            df = df.drop(f, axis=1)

        CMs = list(df.columns[4:])
        nCMs = len(CMs)

        ActiveCMs = np.zeros((nRs, nCMs, nDs))
        Confirmed = np.zeros((nRs, nDs))
        Deaths = np.zeros((nRs, nDs))
        Active = np.zeros((nRs, nDs))
        NewDeaths = np.zeros((nRs, nDs))
        NewCases = np.zeros((nRs, nDs))

        for r_i, r in enumerate(sorted_regions):
            region_names[r_i] = df.loc[(r, Ds[0])]["Region Name"]
            for d_i, d in enumerate(Ds):
                Confirmed[r_i, d_i] = df.loc[(r, d)]["Confirmed"]
                Deaths[r_i, d_i] = df.loc[(r, d)]["Deaths"]
                Active[r_i, d_i] = df.loc[(r, d)]["Active"]

                ActiveCMs[r_i, :, :] = df.loc[r].loc[Ds][CMs].values.T

        # preprocess data
        Confirmed[Confirmed < self.min_confirmed] = np.nan
        Deaths[Deaths < self.min_deaths] = np.nan
        NewCases[:, 1:] = (Confirmed[:, 1:] - Confirmed[:, :-1])
        NewDeaths[:, 1:] = (Deaths[:, 1:] - Deaths[:, :-1])
        NewDeaths[NewDeaths < 0] = 0
        NewCases[NewCases < 0] = 0

        NewCases[np.isnan(NewCases)] = 0
        NewDeaths[np.isnan(NewDeaths)] = 0

        logger.info("Performing Smoothing")
        if self.smooth:
            SmoothedNewCases = np.around(
                ss.convolve2d(NewCases, 1 / self.N_smooth * np.ones(shape=(1, self.N_smooth)), boundary="symm",
                              mode="same"))
            SmoothedNewDeaths = np.around(
                ss.convolve2d(NewDeaths, 1 / self.N_smooth * np.ones(shape=(1, self.N_smooth)), boundary="symm",
                              mode="same"))
            for r in range(nRs):
                # if the country has too few deaths, ignore
                if Deaths[r, -1] < 50:
                    logger.info(f"Skipping smoothing {region_names[r]}")
                    SmoothedNewDeaths[r, :] = NewDeaths[r, :]

                # plt.figure(dpi=250)
                # plt.title(region_names[r])
                # plt.plot(SmoothedNewDeaths[r, :], color="tab:red")
                # plt.plot(NewDeaths[r, :], "--", color="tab:red")
                # plt.plot(SmoothedNewCases[r, :], color="tab:blue")
                # plt.plot(NewCases[r, :], "--", color="tab:blue")
                # plt.yscale("log")
                # print(f"{region_names[r]} Death Difference {np.sum(SmoothedNewDeaths[r, :]) - np.sum(NewDeaths[r, :])} Smoothed Cases Dif {np.sum(SmoothedNewCases[r, :]) - np.sum(NewCases[r, :])}")

            NewCases = SmoothedNewCases
            NewDeaths = SmoothedNewDeaths

        logger.info("Performing Masking")
        if self.mask_zero_deaths:
            NewDeaths[NewDeaths < 1] = np.nan
        else:
            NewDeaths[NewDeaths < 0] = np.nan

        if self.mask_zero_cases:
            NewCases[NewCases < 1] = np.nan
        else:
            NewCases[NewCases < 0] = np.nan

        Confirmed = np.ma.masked_invalid(Confirmed.astype(theano.config.floatX))
        Active = np.ma.masked_invalid(Active.astype(theano.config.floatX))
        Deaths = np.ma.masked_invalid(Deaths.astype(theano.config.floatX))
        NewDeaths = np.ma.masked_invalid(NewDeaths.astype(theano.config.floatX))
        NewCases = np.ma.masked_invalid(NewCases.astype(theano.config.floatX))

        if schools_unis == "xor":
            school_index = CMs.index("School Closure")
            university_index = CMs.index("University Closure")

            ActiveCMs_final = copy.deepcopy(ActiveCMs)
            ActiveCMs_final[:, school_index, :] = np.logical_and(ActiveCMs[:, university_index, :],
                                                                 ActiveCMs[:, school_index, :])
            ActiveCMs_final[:, university_index, :] = np.logical_xor(ActiveCMs[:, university_index, :],
                                                                     ActiveCMs[:, school_index, :])
            ActiveCMs = ActiveCMs_final
            CMs[school_index] = "School and University Closure"
            CMs[university_index] = "Schools xor University Closure"
        elif schools_unis == "single":
            school_index = CMs.index("School Closure")
            university_index = CMs.index("University Closure")

            ActiveCMs_final = copy.deepcopy(ActiveCMs)
            ActiveCMs_final[:, school_index, :] = np.logical_and(ActiveCMs[:, university_index, :],
                                                                 ActiveCMs[:, school_index, :]) + 0.5 * np.logical_xor(
                ActiveCMs[:, university_index, :],
                ActiveCMs[:, school_index, :])

            ActiveCMs = np.delete(ActiveCMs_final, university_index, axis=1)
            CMs.remove("University Closure")

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

class ICLDataPreprocessor(DataPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def preprocess_data(self, data_path, ICL_data_path, days_max=None):
        # load data
        data = super().preprocess_data(data_path, days_max)

        ICL_dict = {
            "AT": "Austria",
            "BE": "Belgium",
            "DK": "Denmark",
            "FR": "France",
            "DE": "Germany",
            "IT": "Italy",
            "NO": "Norway",
            "ES": "Spain",
            "SE": "Sweden",
            "CH": "Switzerland",
            "GB": "United_Kingdom",
            "GR": "Greece",
            "PT": "Portugal",
            "NL": "Netherlands",
        }

        def country_to_code(c):
            for k, v in ICL_dict.items():
                if c == v:
                    return k

        Rs = list(ICL_dict.keys())
        RNames = [ICL_dict[k] for k in Rs]
        ICL_c_i = [data.Rs.index(r) for r in Rs]
        nRs = len(ICL_c_i)
        data.reduce_regions_from_index(ICL_c_i)

        df = pd.read_csv(ICL_data_path, parse_dates=["Date effective"], infer_datetime_format=True)

        CMs = ["Schools + Universities", "Self-isolating if ill", "Public events", "Lockdown",
               "Social distancing encouraged", "Any"]
        nCMs = len(CMs)

        _, _, nDs = data.ActiveCMs.shape

        Ds = data.Ds
        ActiveCMs = np.zeros((nRs, nCMs, nDs))

        for x in df.index:
            row = df.iloc[x]
            c = row["Country"]
            cm = row["Type"]
            on_date = row["Date effective"].date()
            on_date = pd.to_datetime(on_date, utc=True)
            on_loc = Ds.index(on_date)

            if cm in CMs[:-1]:
                r_i = Rs.index(country_to_code(c))
                f_i = CMs.index(cm)
                ActiveCMs[r_i, f_i, on_loc:] = 1
                ActiveCMs[r_i, len(CMs) - 1, on_loc:] = 1

        return PreprocessedData(data.Active,
                                data.Confirmed,
                                ActiveCMs,
                                CMs,
                                Rs,
                                Ds,
                                data.Deaths,
                                data.NewDeaths,
                                data.NewCases,
                                RNames
                                )

class PreprocessedData(object):
    def __init__(self,
                 Active,
                 Confirmed,
                 ActiveCMs,
                 CMs,
                 Rs,
                 Ds,
                 Deaths,
                 NewDeaths,
                 NewCases,
                 RNames):
        super().__init__()
        self.Active = Active
        self.Confirmed = Confirmed
        self.Deaths = Deaths
        self.ActiveCMs = ActiveCMs
        self.Rs = Rs
        self.CMs = CMs
        self.Ds = Ds
        self.NewDeaths = NewDeaths
        self.NewCases = NewCases
        self.RNames = RNames

        # for i, c in enumerate(self.CMs):
        #     if c == "Stay Home Order":
        #         self.CMs[i] = "Stay Home Order (with exemptions)"

    def reduce_regions_from_index(self, reduced_regions_indx):
        self.Active = self.Active[reduced_regions_indx, :]
        self.Confirmed = self.Confirmed[reduced_regions_indx, :]
        self.Deaths = self.Deaths[reduced_regions_indx, :]
        self.NewDeaths = self.NewDeaths[reduced_regions_indx, :]
        self.NewCases = self.NewCases[reduced_regions_indx, :]
        self.ActiveCMs = self.ActiveCMs[reduced_regions_indx, :, :]

    def filter_region_min_deaths(self, min_num_deaths=100):
        reduced_regions = []
        reduced_regions_indx = []
        for indx, r in enumerate(self.Rs):
            if self.Deaths.data[indx, -1] < min_num_deaths:
                print(f"Region {r} removed since it has {self.Deaths[indx, -1]} deaths on the last day")
            elif np.isnan(self.Deaths.data[indx, -1]):
                print(f"Region {r} removed since it has {self.Deaths[indx, -1]} deaths on the last day")
            else:
                reduced_regions.append(r)
                reduced_regions_indx.append(indx)

        self.Rs = reduced_regions
        self.reduce_regions_from_index(reduced_regions_indx)

    def filter_regions(self, regions_to_remove):
        reduced_regions = []
        reduced_regions_indx = []
        for indx, r in enumerate(self.Rs):
            if r in regions_to_remove:
                pass
            else:
                reduced_regions_indx.append(indx)
                reduced_regions.append(r)

        self.Rs = reduced_regions
        _, nCMs, nDs = self.ActiveCMs.shape
        self.reduce_regions_from_index(reduced_regions_indx)

    def ignore_feature(self, f_i):
        self.ActiveCMs[:, f_i, :] = 0

    def ignore_early_features(self):
        for r in range(len(self.Rs)):
            for f_i, f in enumerate(self.CMs):
                if f_i == 0:
                    if np.sum(self.ActiveCMs[r, f_i, :]) > 0:
                        # i.e., if the feature is turned on.
                        nz = np.nonzero(self.ActiveCMs[r, f_i, :])[0]
                        # if the first day that the feature is on corresponds to a masked day. this is conservative
                        if np.isnan(self.Confirmed.data[r, nz[0]]):
                            self.ActiveCMs[r, f_i, :] = 0
                            print(f"Region {self.Rs[r]} has feature {f} removed, since it is too early")

    def coactivation_plot(self, cm_plot_style, newfig=True, skip_yticks=False):
        if newfig:
            plt.figure(figsize=(2, 3), dpi=300)

        nRs, nCMs, nDs = self.ActiveCMs.shape
        plt.title("Frequency $i$ Active Given $j$ Active", fontsize=8)
        ax = plt.gca()
        mat = np.zeros((nCMs, nCMs))
        for cm in range(nCMs):
            mask = self.ActiveCMs[:, cm, :] * (self.NewDeaths.mask == False)
            for cm2 in range(nCMs):
                mat[cm, cm2] = np.sum(mask * self.ActiveCMs[:, cm2, :]) / np.sum(mask)
        im = plt.imshow(mat * 100, vmin=25, vmax=100, cmap="viridis", aspect="auto")
        ax.tick_params(axis="both", which="major", labelsize=8)

        plt.xticks(
            np.arange(len(self.CMs)),
            [f"{cm_plot_style[i][0]}" for i, f in enumerate(self.CMs)],
            fontproperties=fp2,
        )

        for i, ticklabel in enumerate(ax.get_xticklabels()):
            ticklabel.set_color(cm_plot_style[i][1])

        plt.yticks(
            np.arange(len(self.CMs)),
            [f"{f}     " if not skip_yticks else "    " for f in self.CMs]
        )

        plt.xlabel("$i$", fontsize=8)
        plt.ylabel("$j$", fontsize=8)

        x_min, x_max = plt.xlim()
        x_r = x_max - x_min
        for i, (ticklabel, tickloc) in enumerate(zip(ax.get_yticklabels(), ax.get_yticks())):
            ticklabel.set_color(cm_plot_style[i][1])
            plt.text(-0.16 * x_r, tickloc, cm_plot_style[i][0], horizontalalignment='center',
                     verticalalignment='center',
                     fontproperties=fp2, fontsize=7, color=cm_plot_style[i][1])

        plt.xticks(fontsize=7)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbr = plt.colorbar(im, cax=cax, format=PercentFormatter())
        ax = plt.gca()
        ax.tick_params(axis="both", which="major", labelsize=6)
        cbr.set_ticks([25, 50, 75, 100])
        # sns.despine()

    def cumulative_days_plot(self, cm_plot_style, newfig=True, skip_yticks=False):
        if newfig:
            plt.figure(figsize=(3, 3), dpi=300)

        nRs, nCMs, nDs = self.ActiveCMs.shape

        ax = plt.gca()
        mask = np.reshape((self.NewDeaths.mask == False), (nRs, 1, nDs))
        days_active = np.sum(np.sum(self.ActiveCMs * np.repeat(mask, nCMs, axis=1), axis=0), axis=1)
        plt.barh(-np.arange(nCMs), days_active)

        plt.yticks(
            -np.arange(len(self.CMs)),
            [f"{f}     " if not skip_yticks else "    " for f in self.CMs]
        )

        x_min, x_max = plt.xlim()
        x_r = x_max - x_min
        for i, (ticklabel, tickloc) in enumerate(zip(ax.get_yticklabels(), ax.get_yticks())):
            ticklabel.set_color(cm_plot_style[i][1])
            plt.text(-0.09 * x_r, tickloc, cm_plot_style[i][0], horizontalalignment='center',
                     verticalalignment='center',
                     fontproperties=fp2, fontsize=7, color=cm_plot_style[i][1])

        plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000], fontsize=6)
        # ax.tick_params(axis="both", which="major", labelsize=10)
        plt.title("Total Days Active", fontsize=8)
        plt.xlabel("Days", fontsize=8)
        plt.ylim([-len(self.CMs) + 0.5, 0.5])

    def summary_plot(self, cm_plot_style):
        plt.figure(figsize=(10, 3), dpi=300)
        plt.subplot(1, 2, 1)
        self.coactivation_plot(cm_plot_style, False)
        plt.subplot(1, 2, 2)
        self.cumulative_days_plot(cm_plot_style, False)
        plt.tight_layout()
        plt.savefig("FigureCA.pdf", bbox_inches='tight')
        # sns.despine()

    def alt_summary_plot(self, cm_plot_style):
        plt.figure(figsize=(9, 3.5), dpi=300)
        plt.subplot(1, 2, 1)
        self.alt_coactivation_plot(cm_plot_style, False)
        plt.subplot(1, 2, 2)
        self.alt_cumulative_days_plot(cm_plot_style, False)
        plt.tight_layout()
        plt.savefig("Data.pdf", bbox_inches='tight')
        # sns.despine()

    def alt_coactivation_plot(self, cm_plot_style, newfig=True, skip_yticks=False):
        if newfig:
            plt.figure(figsize=(2, 3), dpi=300)

        nRs, nCMs, nDs = self.ActiveCMs.shape
        plt.title("Frequency$[\phi_{i} = 1 | \phi_j = 1]$", fontsize=8)
        ax = plt.gca()
        mat = np.zeros((nCMs, nCMs))
        for cm in range(nCMs):
            mask = self.ActiveCMs[:, cm, :]
            for cm2 in range(nCMs):
                mat[cm, cm2] = np.sum(mask * self.ActiveCMs[:, cm2, :]) / np.sum(mask)
        im = plt.imshow(mat * 100, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.tick_params(axis="both", which="major", labelsize=8)

        plt.xticks(
            np.arange(len(self.CMs)),
            self.CMs, ha="left", rotation=-25
        )

        plt.yticks(
            np.arange(len(self.CMs)),
            self.CMs
        )

        plt.xlabel("NPI $i$", fontsize=8)
        plt.ylabel("NPI $j$", fontsize=8)

        x_min, x_max = plt.xlim()
        x_r = x_max - x_min

        plt.xticks(fontsize=8)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, format=PercentFormatter())
        ax = plt.gca()
        ax.tick_params(axis="both", which="major", labelsize=8)

    def alt_cumulative_days_plot(self, cm_plot_style, newfig=True, skip_yticks=False):
        if newfig:
            plt.figure(figsize=(3, 3), dpi=300)

        nRs, nCMs, nDs = self.ActiveCMs.shape

        ax = plt.gca()
        days_active = np.sum(np.sum(self.ActiveCMs, axis=0), axis=1)
        plt.barh(-np.arange(nCMs), days_active)

        plt.yticks(
            -np.arange(len(self.CMs)),
            self.CMs
        )

        x_min, x_max = plt.xlim()
        x_r = x_max - x_min

        plt.xticks([0, 500, 1000, 1500, 2000], fontsize=6)
        ax.tick_params(axis="both", which="major", labelsize=8)
        plt.title("Total Days Active", fontsize=8)
        plt.xlabel("Days", fontsize=8)
        plt.ylim([-len(self.CMs) + 0.5, 0.5])

    def mask_reopenings(self, d_min=90, n_extra=0, max_cms=None):

        if max_cms is not None:
            n_max = max_cms
        else:
            n_max = len(self.CMs)

        total_cms = self.ActiveCMs[:, :n_max, :]
        diff_cms = np.zeros_like(total_cms)
        diff_cms[:, :, 1:] = total_cms[:, :, 1:] - total_cms[:, :, :-1]
        rs, ds = np.nonzero(np.any(diff_cms < 0, axis=1))
        nnz = rs.size

        for nz_i in range(nnz):
            if (ds[nz_i] + 3) > d_min and ds[nz_i] + 3 < len(self.Ds):
                print(f"Masking {self.Rs[rs[nz_i]]} from {self.Ds[ds[nz_i] + 3]}")
                self.NewCases[rs[nz_i], ds[nz_i] + 3 - n_extra:].mask = True
                self.NewDeaths[rs[nz_i], ds[nz_i] + 12 - n_extra:].mask = True

    def mask_region_ends(self, ndays=20):
        for rg in self.Rs:
            i = self.Rs.index(rg)
            self.Active.mask[i, -ndays:] = True
            self.Confirmed.mask[i, -ndays:] = True
            self.Deaths.mask[i, -ndays:] = True
            self.NewDeaths.mask[i, -ndays:] = True
            self.NewCases.mask[i, -ndays:] = True
