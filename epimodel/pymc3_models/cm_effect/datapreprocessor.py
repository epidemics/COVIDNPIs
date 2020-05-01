import pandas as pd
import numpy as np
import theano
import matplotlib.pyplot as plt
import seaborn as sns

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


class DataPreprocessor(object):
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


class DataPreprocessorV2(DataPreprocessor):

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
            epifor_end_date="2020-04-21"
    ):
        # at the moment only features from the 0-1 countermeasures dataset
        Ds = pd.date_range(start=self.start_date, end=self.end_date, tz="utc")
        nDs = len(Ds)

        region_ds = RegionDataset.load(os.path.join(data_base_path, "regions.csv"))
        johnhop_ds = read_csv(os.path.join(data_base_path, "johns-hopkins.csv"))

        self.johnhop_ds = johnhop_ds

        epi_cmset = read_csv(os.path.join(data_base_path, self.episet_fname))

        region_names = list([x for x, _, _ in region_info])
        regions_epi = list([x for _, x, _ in region_info])
        regions_oxcgrt = list([x for _, _, x in region_info])

        # country filtering
        filtered_countries = []
        for cc in set(regions_epi):
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
        filtered_countries.sort()

        # epidemic forecasting.org dataset
        sd = epi_cmset.loc[filtered_countries, selected_features_epi]

        # overwrite epidemic forecasting data with dataset checks if they exist

        check_cols= {'Events above 10 people banned authoritative':'Gatherings limited to 10',
                     'Events above 100 people banned authoritative':'Gatherings limited to 100',
                     'Events above 1000 people banned authoritative':'Gatherings limited to 1000',
                     'Stay at home authoritative':'General curfew',
                     'Risky businesses closed authoritative':'Business suspended - some',
                     'All non-essential businesses closed authoritative':'Business suspended - many',
                     'Hospital specialization level 2 authoritative':'Healthcare specialisation'}

        epicheck = pd.read_csv(os.path.join(data_base_path,self.epicheck_fname),skiprows=[1]).rename(columns=check_cols).set_index('Code')

        epicheck = epicheck.loc[epicheck.index.isin(filtered_countries)]

        for col in check_cols.values():
            epicheck[col] = epicheck[col].replace("no",'2021-01-01')
            epicheck[col] = epicheck[col].str.replace('.','-')
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
                ActiveCMs_oxcgrt[r,oxcgrt_derived_cm_names.index(k),epi_date_range] = ActiveCMs_epi[r,selected_features_epi.index(v),epi_date_range]

        for r in range(nRs):
            for f_indx, f in enumerate(ordered_features):            
                if f in selected_features_epi:
                    ActiveCMs[r, f_indx, :] = ActiveCMs_epi[r, selected_features_epi.index(f), :]
                else:
                    ActiveCMs[r, f_indx, :] = ActiveCMs_oxcgrt[r, oxcgrt_derived_cm_names.index(f), :]

        # [country, CM, day] Which CMs are active, and to what extent
        ActiveCMs = ActiveCMs.astype(theano.config.floatX)

        dataset_summary_plot(ordered_features, ActiveCMs)

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
            ordered_features,
            filtered_countries,
            Ds,
            Deaths,
            NewDeaths
        )

        return loaded_data


class PreprocessedData(object):
    def __init__(self, Active, Confirmed, ActiveCMs, CMs, Rs, Ds, Deaths, NewDeaths):
        super().__init__()
        self.Active = Active
        self.Confirmed = Confirmed
        self.Deaths = Deaths
        self.ActiveCMs = ActiveCMs
        self.Rs = Rs
        self.CMs = CMs
        self.Ds = Ds
        self.NewDeaths = NewDeaths
