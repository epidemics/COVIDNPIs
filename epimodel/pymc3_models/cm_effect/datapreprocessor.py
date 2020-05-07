import copy

import pandas as pd
import numpy as np
import theano
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as ss

from mpl_toolkits.axes_grid1 import make_axes_locatable

from epimodel import read_csv, RegionDataset

from epimodel.regions import Level

import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dataset_summary_plot(features, ActiveCMs):
    nRs, nCMs, nDs = ActiveCMs.shape

    plt.figure(figsize=(10, 5), dpi=300)
    plt.subplot(1, 2, 1)
    plt.title("Co-activation")
    ax = plt.gca()
    mat = np.zeros((nCMs, nCMs))
    for cm in range(nCMs):
        mask = ActiveCMs[:, cm, :]
        for cm2 in range(nCMs):
            mat[cm, cm2] = np.sum(mask * ActiveCMs[:, cm2, :]) / np.sum(mask)
    im = plt.imshow(mat * 100, vmin=0, vmax=100, cmap="inferno", aspect="auto")
    ax.tick_params(axis="both", which="major", labelsize=8)

    plt.xticks(
        np.arange(len(features)),
        [f for f in features],
        rotation=90
    )

    plt.yticks(
        np.arange(len(features)),
        [f for f in features],
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    days_active = np.sum(np.sum(ActiveCMs, axis=0), axis=1)
    plt.barh(-np.arange(nCMs), days_active)

    plt.yticks(
        -np.arange(len(features)),
        [f for f in features],
    )

    ax.tick_params(axis="both", which="major", labelsize=8)
    plt.title("Days NPI Used")
    plt.xlabel("Days")

    plt.tight_layout()
    sns.despine()


class DataMerger():
    def __init__(self, params_dict=None, *args, **kwargs):
        self.start_date = "2020-2-10"
        self.end_date = "2020-04-05"

        # rather than being final this will probability want to be min max
        self.min_final_num_active_cases = 100
        self.min_num_active_mask = 10
        self.min_num_confirmed_mask = 10

        self.episet_fname = "countermeasures-model-boolean_Gat3Bus2SchCurHespMa.csv"
        self.oxcgrt_fname = "OxCGRT_latest.csv"
        self.epicheck_fname = "Hspec_Bus_Sah_Gath_doublecheck.csv"
        self.mask_override_fname = "mask_override.csv"
        self.johnhop_fname = "johns-hopkins.csv"
        self.regions_fname = "regions.csv"

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
            epifor_check_cols
    ):
        # at the moment only features from the 0-1 countermeasures dataset
        Ds = pd.date_range(start=self.start_date, end=self.end_date, tz="utc")
        self.Ds = Ds

        region_ds = RegionDataset.load(os.path.join(data_base_path, self.regions_fname))
        johnhop_ds = read_csv(os.path.join(data_base_path, self.johnhop_fname))
        mask_override_ds = pd.read_csv(os.path.join(data_base_path, self.mask_override_fname))
        epi_cmset = read_csv(os.path.join(data_base_path, self.episet_fname))
        epi_cmset = epi_cmset.rename(columns=selected_features_epi, errors="raise")

        region_names = list([x for x, _, _ in region_info])
        regions_epi = list([x for _, x, _ in region_info])
        regions_oxcgrt = list([x for _, _, x in region_info])

        # country filtering
        filtered_countries = []
        for cc in regions_epi:
            c = region_ds[cc]
            if (
                    c.Level == Level.country
                    and c.Code in johnhop_ds.index
                    and c.Code in epi_cmset.index
            ):
                if (
                        johnhop_ds.loc[(c.Code, Ds[-1]), "Active"]
                        > self.min_final_num_active_cases
                ):
                    filtered_countries.append(c.Code)

        # epidemic forecasting.org dataset
        print("Loading from epidemicforecasting.org")
        for feat in selected_features_epi.values():
            if not feat in epi_cmset:
                logger.warning(f"{feat} is missing in the original epidemicforecasting.org DB")
                epi_cmset[feat] = 0

        sd = epi_cmset.loc[filtered_countries, selected_features_epi.values()]

        # overwrite epidemic forecasting data with dataset checks if they exist

        logger.info("Updating from epidemicforecasting.org double-check data")
        epicheck = pd.read_csv(os.path.join(data_base_path, self.epicheck_fname), skiprows=[1]).rename(
            columns=epifor_check_cols).set_index('Code')
        epicheck = epicheck.loc[epicheck.index.isin(filtered_countries)]

        for col in epifor_check_cols.values():
            logger.info(f" updating {col}")
            epicheck[col] = epicheck[col].str.lower().replace("no", '01-01-2021')
            epicheck[col] = pd.to_datetime(epicheck[col].str.replace('.', '-'), format="%d-%m-%Y")
            for ccode in epicheck.index:
                switch_date = epicheck.loc[ccode, col]
                if not pd.isna(switch_date):
                    dates_off = pd.date_range(self.start_date, switch_date)
                    dates_on = pd.date_range(switch_date, self.end_date)
                    sd.loc[(ccode, dates_off), col] = 0
                    sd.loc[(ccode, dates_on), col] = 1

        logger.info("Mask Override")
        nRs_over, _ = mask_override_ds.shape
        for r in range(nRs_over):
            switch_date = mask_override_ds.iloc[r, 2].strip()
            ccode = mask_override_ds.iloc[r, 1].strip()
            # print(f"{ccode} actually had masks from {switch_date}")
            # print(self.start_date)
            # print(self.end_date)
            dates_off = pd.date_range(self.start_date, switch_date)
            dates_on = pd.date_range(switch_date, self.end_date)
            # print(dates_off)
            # print(dates_on)
            sd.loc[(ccode, dates_off), "Mask Wearing"] = 0
            sd.loc[(ccode, dates_on), "Mask Wearing"] = 1

        ActiveCMs_epi = np.stack([sd.loc[c].loc[Ds].T for c in filtered_countries])
        logger_str = "\nCountermeasures: EpidemicForecasting.org           min   ... mean  ... max   ... unique"
        for i, cm in enumerate(selected_features_epi.values()):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs_epi[:, i, :]):.3f} ... {np.mean(ActiveCMs_epi[:, i, :]):.3f} ... {np.max(ActiveCMs_epi[:, i, :]):.3f} ... {np.unique(ActiveCMs_epi[:, i, :])[:5]}"
        logger.info(logger_str)

        logger.info("Load OXCGRT")
        unique_missing_countries = []

        # OxCGRT dataset
        def oxcgrt_to_epimodel_index(ind):
            try:
                return regions_epi[regions_oxcgrt.index(ind)]
            except ValueError:
                if ind not in unique_missing_countries:
                    unique_missing_countries.append(ind)
                return ind

        date_column = "Date"
        data_oxcgrt = pd.read_csv(os.path.join(data_base_path, self.oxcgrt_fname), index_col="CountryCode")

        columns_to_drop = ["CountryName", "Date", "ConfirmedCases", "ConfirmedDeaths",
                           "StringencyIndex", "StringencyIndexForDisplay",
                           "LegacyStringencyIndex", "LegacyStringencyIndexForDisplay"]

        dti = pd.DatetimeIndex(pd.to_datetime(data_oxcgrt[date_column], utc=True, format="%Y%m%d"))
        epi_codes = [oxcgrt_to_epimodel_index(cc) for cc in data_oxcgrt.index.array]
        logger.warning(f"Missing {unique_missing_countries} from epidemicforecasting.org DB which are in OxCGRT")
        data_oxcgrt.index = pd.MultiIndex.from_arrays([epi_codes, dti])

        for col in columns_to_drop:
            del data_oxcgrt[col]

        data_oxcgrt.sort_index()

        data_oxcgrt_filtered = data_oxcgrt.loc[regions_epi, selected_features_oxcgrt]

        values_to_stack = []
        for c in regions_epi:
            if c in data_oxcgrt_filtered.index:
                values_to_stack.append(data_oxcgrt_filtered.loc[c].loc[Ds].T)
            else:
                logger.info(f"Missing {c} from OXCGRT. Assuming features are 0")
                values_to_stack.append(np.zeros_like(values_to_stack[-1]))

        ActiveCMs_temp = np.stack(values_to_stack)
        nRs, _, nDs = ActiveCMs_temp.shape
        nCMs = len(oxcgrt_feature_info)
        ActiveCMs_oxcgrt = np.zeros((nRs, nCMs, nDs))
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
                # we need all conditions to be satisfied, hence a product
                ActiveCMs_oxcgrt[r_indx, feature_indx, :] = np.prod(condition_mat, axis=0) > 0

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

        dataset_summary_plot(ordered_features, ActiveCMs)

        Confirmed = np.stack([johnhop_ds["Confirmed"].loc[(fc, Ds)] for fc in filtered_countries])
        Active = np.stack([johnhop_ds["Active"].loc[(fc, Ds)] for fc in filtered_countries])
        Deaths = np.stack([johnhop_ds["Deaths"].loc[(fc, Ds)] for fc in filtered_countries])

        columns = ["Country Code", "Date", "Region Name", "Confirmed", "Active", "Deaths", *ordered_features]
        df = pd.DataFrame(columns=columns)
        for r_indx, r in enumerate(filtered_countries):
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
        df.to_csv("data_final.csv")
        logger.info("Saved final CSV")


class DataPreprocessor():
    def __init__(self, *args, **kwargs):
        self.min_confirmed = 100
        self.min_deaths = 10

        self.mask_zero_deaths = False
        self.mask_zero_cases = False

        self.smooth = True
        self.N_smooth = 5

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def generate_params_dict(self):
        return {
            "min_final_num_active_cases": self.min_final_num_active_cases,
            "confirmed_mask": self.min_num_active_mask,
        }

    def preprocess_data(self, data_path):
        # load data
        df = pd.read_csv(data_path, parse_dates=["Date"], infer_datetime_format=True).set_index(
            ["Country Code", "Date"])
        Ds = list(df.index.levels[1])
        nDs = len(Ds)

        all_rs = list([r for r, _ in df.index])
        regions = list(df.index.levels[0])
        locations = [all_rs.index(r) for r in regions]
        sorted_regions = [r for l, r in sorted(zip(locations, regions))]
        nRs = len(sorted_regions)
        region_names = copy.deepcopy(sorted_regions)

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
        NewCases[:, 1:] = Confirmed[:, 1:] - Confirmed[:, :-1]
        NewDeaths[:, 1:] = Deaths[:, 1:] - Deaths[:, :-1]
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

        dataset_summary_plot(CMs, ActiveCMs)

        Confirmed = np.ma.masked_invalid(Confirmed.astype(theano.config.floatX))
        Active = np.ma.masked_invalid(Active.astype(theano.config.floatX))
        Deaths = np.ma.masked_invalid(Deaths.astype(theano.config.floatX))
        NewDeaths = np.ma.masked_invalid(NewDeaths.astype(theano.config.floatX))
        NewCases = np.ma.masked_invalid(NewCases.astype(theano.config.floatX))
        return PreprocessedData(Active,
                                Confirmed,
                                ActiveCMs,
                                CMs,
                                sorted_regions,
                                Ds,
                                Deaths,
                                NewDeaths,
                                NewCases)

class DataPreprocessorLegacy1(object):
    def __init__(self, params_dict=None, *args, **kwargs):

        self.start_date = "2020-2-10"
        self.end_date = "2020-04-05"

        # rather than being final this will probability want to be min max
        self.min_final_num_active_cases = 100
        self.min_num_active_mask = 10
        self.min_num_confirmed_mask = 10

        # load parameters, first from dictionary and then from kwargs
        if params_dict is not None:
            for key, value in params_dict.items():
                setattr(self, key, value)

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def generate_params_dict(self):
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "min_final_num_active_cases": self.min_final_num_active_cases,
            "min_num_active_mask": self.min_num_active_mask,
            "min_num_confirmed_mask": self.min_num_confirmed_mask,
        }

    def preprocess_data(
            self,
            data_base_path,
            countries,
            selected_features,
            selected_cm_set="countermeasures-model-0to1-split.csv",
    ):
        # at the moment only features from the 0-1 countermeasures dataset
        Ds = pd.date_range(start=self.start_date, end=self.end_date, tz="utc")
        nDs = len(Ds)

        region_ds = RegionDataset.load(os.path.join(data_base_path, "regions.csv"))
        johnhop_ds = read_csv(os.path.join(data_base_path, "johns-hopkins.csv"))

        self.johnhop_ds = johnhop_ds

        cm_set_dirs = [
            "countermeasures-features.csv",
            "countermeasures-model-boolean_Gat3Bus2SchCurHespMa.csv",
        ]

        cm_sets = {
            cm_set_file: read_csv(os.path.join(data_base_path, cm_set_file))
            for cm_set_file in cm_set_dirs
        }
        for n, v in cm_sets.items():
            logger.debug(f"\nCMS {n} columns:\n{v.columns!r}")

        selected_CMs = selected_features
        CM_dataset = cm_sets[selected_cm_set]
        nCMs = len(selected_features)

        self.CM_dataset = CM_dataset

        filtered_countries = []
        for cc in set(countries):
            c = region_ds[cc]
            if (
                    c.Level == Level.country
                    and c.Code in johnhop_ds.index
                    and c.Code in CM_dataset.index
            ):
                if (
                        johnhop_ds.loc[(c.Code, Ds[-1]), "Active"]
                        > self.min_final_num_active_cases
                ):
                    filtered_countries.append(c.Code)
        nCs = len(filtered_countries)
        # note that it is essential to sort these values to get the correct corresponances from the john hopkins dataset
        filtered_countries.sort()
        sd = CM_dataset.loc[filtered_countries, selected_CMs]
        if "Mask wearing" in selected_CMs:
            sd["Mask wearing"] *= 0.01

        logger_str = "\nCountermeasures                               min   ... mean  ... max   ... unique"
        for i, cm in enumerate(selected_CMs):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {sd[cm].min().min():.3f} ... {sd[cm].mean().mean():.3f} ... {sd[cm].max().max():.3f} ... {np.unique(sd[cm])[:5]}"

        logger.info(logger_str)
        ActiveCMs = np.stack([sd.loc[c].loc[Ds].T for c in filtered_countries])
        assert ActiveCMs.shape == (nCs, nCMs, nDs)
        # [country, CM, day] Which CMs are active, and to what extent
        ActiveCMs = ActiveCMs.astype(theano.config.floatX)

        dataset_summary_plot(selected_features, ActiveCMs)

        Confirmed = (
            johnhop_ds["Confirmed"]
                .loc[(tuple(filtered_countries), Ds)]
                .unstack(1)
                .values
        )
        assert Confirmed.shape == (nCs, nDs)
        Confirmed[Confirmed < self.min_num_confirmed_mask] = np.nan
        Confirmed = np.ma.masked_invalid(Confirmed.astype(theano.config.floatX))

        # Active cases, masking values smaller than 10
        Active = (
            johnhop_ds["Active"].loc[(tuple(filtered_countries), Ds)].unstack(1).values
        )
        assert Active.shape == (nCs, nDs)
        Active[Active < self.min_num_active_mask] = np.nan
        # [country, day]
        Active = np.ma.masked_invalid(Active.astype(theano.config.floatX))

        logger.info(
            f"Data Preprocessing Complete using:\n\n{json.dumps(self.generate_params_dict(), indent=4)}\n"
            f"Selected {len(filtered_countries)} Regions: f{filtered_countries}"
        )

        Deaths = (
            johnhop_ds["Deaths"].loc[(tuple(filtered_countries), Ds)].unstack(1).values
        )
        assert Active.shape == (nCs, nDs)
        Deaths[Deaths < 10] = np.nan
        # [country, day]
        Deaths = np.ma.masked_invalid(Deaths.astype(theano.config.floatX))

        NewDeaths = np.zeros(shape=Deaths.shape)
        NewDeaths[:, 1:] = Deaths[:, 1:] - Deaths[:, :-1]
        NewDeaths[np.isnan(NewDeaths)] = 0
        NewDeaths[NewDeaths < 0] = np.nan
        NewDeaths = np.ma.masked_invalid(NewDeaths.astype(theano.config.floatX))
        NewDeaths = NewDeaths.astype(int)

        loaded_data = PreprocessedData(
            Active,
            Confirmed,
            ActiveCMs,
            selected_features,
            filtered_countries,
            Ds,
            Deaths,
            NewDeaths
        )

        return loaded_data


class DataPreprocessorLegacy2(DataPreprocessorLegacy1):

    def __init__(self, params_dict=None, *args, **kwargs):
        super().__init__(params_dict, *args, **kwargs)
        self.episet_fname = "countermeasures-model-boolean_Gat3Bus2SchCurHespMa.csv"
        self.oxcgrt_fname = "OxCGRT_latest.csv"
        self.epicheck_fname = "Hspec_Bus_Sah_Gath_doublecheck.csv"

    def preprocess_data(
            self,
            data_base_path,
            region_info,
            oxcgrt_feature_info,
            selected_features_oxcgrt,
            selected_features_epi,
            ordered_features,
            oxford_to_epi_features,
            epifor_end_date="2020-04-21",
            mask_zero_deaths=True,
    ):
        # at the moment only features from the 0-1 countermeasures dataset
        Ds = pd.date_range(start=self.start_date, end=self.end_date, tz="utc")
        self.Ds = Ds

        region_ds = RegionDataset.load(os.path.join(data_base_path, "regions.csv"))
        johnhop_ds = read_csv(os.path.join(data_base_path, "johns-hopkins.csv"))
        self.johnhop_ds = johnhop_ds

        epi_cmset = read_csv(os.path.join(data_base_path, self.episet_fname))

        regions_epi = list([x for _, x, _ in region_info])
        regions_oxcgrt = list([x for _, _, x in region_info])

        # country filtering
        filtered_countries = []
        for cc in regions_epi:
            c = region_ds[cc]
            if (
                    c.Level == Level.country
                    and c.Code in johnhop_ds.index
                    and c.Code in epi_cmset.index
            ):
                if (
                        johnhop_ds.loc[(c.Code, Ds[-1]), "Active"]
                        > self.min_final_num_active_cases
                ):
                    filtered_countries.append(c.Code)
        nCs = len(filtered_countries)

        # note that it is essential to sort these values to get the correct corresponances from the john hopkins dataset
        # filtered_countries.sort()

        # epidemic forecasting.org dataset
        for feat in selected_features_epi:
            if not feat in epi_cmset:
                epi_cmset[feat] = 0
        sd = epi_cmset.loc[filtered_countries, selected_features_epi]

        # overwrite epidemic forecasting data with dataset checks if they exist

        check_cols= {'Events above 10 people banned authoritative':'Gatherings limited to 10',
                     'Events above 100 people banned authoritative':'Gatherings limited to 100',
                     'Events above 1000 people banned authoritative':'Gatherings limited to 1000',
                     'Stay at home authoritative':'General curfew',
                     'Risky businesses closed authoritative':'Business suspended - some',
                     'All non-essential businesses closed authoritative':'Business suspended - many',
                     'Infection control in healthcare system - strong authoritative':'Healthcare infection control - strong',
                     'Infection control in healthcare system - weak authoritative':'Healthcare infection control - basic'}

        epicheck = pd.read_csv(os.path.join(data_base_path,self.epicheck_fname),skiprows=[1]).rename(columns=check_cols).set_index('Code')

        epicheck = epicheck.loc[epicheck.index.isin(filtered_countries)]

        for col in check_cols.values():
            epicheck[col] = epicheck[col].str.lower().replace("no",'01-01-2021')
            epicheck[col] = pd.to_datetime(epicheck[col].str.replace('.','-'),format="%d-%m-%Y")
            for ccode in epicheck.index:
                switch_date = epicheck.loc[ccode,col]
                if not pd.isna(switch_date):
                    dates_off = pd.date_range(self.start_date,switch_date)
                    dates_on = pd.date_range(switch_date,self.end_date)
                    sd.loc[(ccode,dates_off),col] = 0
                    sd.loc[(ccode,dates_on),col] = 1


        logger_str = "\nCountermeasures: Epidemic Forecasting              min   ... mean  ... max   ... unique"
        for i, cm in enumerate(selected_features_epi):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {sd[cm].min().min():.3f} ... {sd[cm].mean().mean():.3f} ... {sd[cm].max().max():.3f} ... {np.unique(sd[cm])[:5]}"
        logger.info(logger_str)

        ActiveCMs_epi = np.stack([sd.loc[c].loc[Ds].T for c in filtered_countries])

        # OxCGRT dataset
        def oxcgrt_to_epimodel_index(ind):
            try:
                return regions_epi[regions_oxcgrt.index(ind)]
            except:
                return ind

        date_column = "Date"
        data_oxcgrt = pd.read_csv(os.path.join(data_base_path, self.oxcgrt_fname), index_col="CountryCode")

        columns_to_drop = ["CountryName", "Date", "ConfirmedCases", "ConfirmedDeaths",
                           "StringencyIndex", "StringencyIndexForDisplay",
                           "LegacyStringencyIndex", "LegacyStringencyIndexForDisplay"]

        dti = pd.DatetimeIndex(pd.to_datetime(data_oxcgrt[date_column], utc=True, format="%Y%m%d"))
        epi_codes = [oxcgrt_to_epimodel_index(cc) for cc in data_oxcgrt.index.array]
        data_oxcgrt.index = pd.MultiIndex.from_arrays([epi_codes, dti])

        for col in columns_to_drop:
            del data_oxcgrt[col]

        data_oxcgrt.sort_index()

        data_oxcgrt_filtered = data_oxcgrt.loc[regions_epi, selected_features_oxcgrt]
        ActiveCMs_temp = np.stack([data_oxcgrt_filtered.loc[c].loc[Ds].T for c in regions_epi])

        nRs, _, nDs = ActiveCMs_temp.shape
        nCMs = len(oxcgrt_feature_info)

        ActiveCMs_oxcgrt = np.zeros((nRs, nCMs, nDs))
        oxcgrt_derived_cm_names = [n for n, _ in oxcgrt_feature_info]

        for r_indx in range(nRs):
            for feature_indx, (_, feature_filter) in enumerate(oxcgrt_feature_info):
                nConditions = len(feature_filter)
                condition_mat = np.zeros((nConditions, nDs))
                for condition, (row, poss_values) in enumerate(feature_filter):
                    row_vals = ActiveCMs_temp[r_indx, row, :]
                    for value in poss_values:
                        condition_mat[condition, :] += (row_vals == value)
                    condition_mat[condition, :] = condition_mat[condition, :] > 0
                ActiveCMs_oxcgrt[r_indx, feature_indx, :] = np.prod(condition_mat, axis=0) > 0

        logger_str = "\nCountermeasures: OxCGRT           min   ... mean  ... max   ... unique"
        for i, cm in enumerate(oxcgrt_derived_cm_names):
            logger_str = f"{logger_str}\n{i + 1:2} {cm:42} {np.min(ActiveCMs_oxcgrt[i]):.3f} ... {np.mean(ActiveCMs_oxcgrt[i]):.3f} ... {np.max(ActiveCMs_oxcgrt[i]):.3f} ... {np.unique(ActiveCMs_oxcgrt[i])[:5]}"
        logger.info(logger_str)

        nCMs = len(ordered_features)
        ActiveCMs = np.zeros((nRs, nCMs, nDs))

        epi_date_range = np.arange((pd.Timestamp(epifor_end_date) - pd.Timestamp(self.start_date)).days)

        for r in range(nRs):
            for k, v in oxford_to_epi_features.items():
                edr_feat = epi_date_range
                ox_revert_index = np.where(ActiveCMs_oxcgrt[r,oxcgrt_derived_cm_names.index(k),:]<0)[0]
                if len(ox_revert_index):
                    edr_feat = epi_date_range[:ox_revert_index[0]]
                ActiveCMs_oxcgrt[r,oxcgrt_derived_cm_names.index(k),edr_feat] = ActiveCMs_epi[r,selected_features_epi.index(v),edr_feat]

        for r in range(nRs):
            for f_indx, f in enumerate(ordered_features):
                if f in selected_features_epi:
                    ActiveCMs[r, f_indx, :] = ActiveCMs_epi[r, selected_features_epi.index(f), :]
                else:
                    ActiveCMs[r, f_indx, :] = ActiveCMs_oxcgrt[r, oxcgrt_derived_cm_names.index(f), :]

        # [country, CM, day] Which CMs are active, and to what extent
        ActiveCMs = ActiveCMs.astype(theano.config.floatX)

        dataset_summary_plot(ordered_features, ActiveCMs)

        Confirmed = np.stack([johnhop_ds["Confirmed"].loc[(fc, Ds)] for fc in filtered_countries])

        assert Confirmed.shape == (nCs, nDs)
        Confirmed[Confirmed < self.min_num_confirmed_mask] = np.nan
        Confirmed = np.ma.masked_invalid(Confirmed.astype(theano.config.floatX))

        Active = np.stack([johnhop_ds["Active"].loc[(fc, Ds)] for fc in filtered_countries])

        assert Active.shape == (nCs, nDs)
        Active[Active < self.min_num_active_mask] = np.nan
        # [country, day]
        Active = np.ma.masked_invalid(Active.astype(theano.config.floatX))

        logger.info(
            f"Data Preprocessing Complete using:\n\n{json.dumps(self.generate_params_dict(), indent=4)}\n"
            f"Selected {filtered_countries} Regions: f{filtered_countries}"
        )

        Deaths = np.stack([johnhop_ds["Deaths"].loc[(fc, Ds)] for fc in filtered_countries])
        assert Deaths.shape == (nCs, nDs)
        Deaths[Deaths < 10] = np.nan
        # [country, day]
        Deaths = np.ma.masked_invalid(Deaths.astype(theano.config.floatX))

        NewDeaths = np.zeros(shape=Deaths.shape)
        NewDeaths[:, 1:] = Deaths[:, 1:] - Deaths[:, :-1]
        NewDeaths[np.isnan(NewDeaths)] = 0

        if mask_zero_deaths:
            NewDeaths[NewDeaths < 1] = np.nan
        else:
            # annoyingly negative deaths do appear!
            NewDeaths[NewDeaths < 0] = np.nan

        NewDeaths = np.ma.masked_invalid(NewDeaths.astype(theano.config.floatX))

        NewCases = np.zeros(shape=Confirmed.shape)
        NewCases[:, 1:] = Confirmed[:, 1:] - Confirmed[:, :-1]
        NewCases[np.isnan(NewCases)] = 0

        mask_zero_cases = False
        if mask_zero_cases:
            NewCases[NewCases < 1] = np.nan
        else:
            # annoyingly negative Cases do appear!
            NewCases[NewCases < 0] = np.nan

        NewCases = np.ma.masked_invalid(NewCases.astype(theano.config.floatX))

        loaded_data = PreprocessedData(
            Active,
            Confirmed,
            ActiveCMs,
            ordered_features,
            filtered_countries,
            Ds,
            Deaths,
            NewDeaths,
            NewCases,
        )

        return loaded_data

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
                 NewCases):
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
        _, nCMs, nDs = self.ActiveCMs.shape

        self.Active = self.Active[reduced_regions_indx, :]
        self.Confirmed = self.Confirmed[reduced_regions_indx, :]
        self.Deaths = self.Deaths[reduced_regions_indx, :]
        self.NewDeaths = self.NewDeaths[reduced_regions_indx, :]
        self.NewCases = self.NewCases[reduced_regions_indx, :]
        self.ActiveCMs = self.ActiveCMs[reduced_regions_indx, :, :]
