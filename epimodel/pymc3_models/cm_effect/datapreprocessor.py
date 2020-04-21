import pandas as pd
import numpy as np
import theano
import matplotlib.pyplot as plt

from epimodel import read_csv, RegionDataset

from epimodel.regions import Level

import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor(object):
    def __init__(self, params_dict=None, **kwargs):

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

    def preprocess_data(self, data_base_path, countries, selected_features,
                        selected_cm_set="countermeasures-model-0to1-split.csv"):
        # at the moment only features from the 0-1 countermeasures dataset
        Ds = pd.date_range(start=self.start_date, end=self.end_date, tz="utc")
        nDs = len(Ds)

        region_ds = RegionDataset.load(os.path.join(data_base_path, "regions.csv"))
        johnhop_ds = read_csv(os.path.join(data_base_path, "johns-hopkins.csv"))

        self.johnhop_ds = johnhop_ds

        cm_set_dirs = [
            "countermeasures-features.csv",
            "countermeasures-model-0to1.csv",
            "countermeasures-selected-binary.csv",
            "countermeasures-model-0to1-split.csv"
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

        logger_str = (
            "\nCountermeasures                               min   ... mean  ... max   ... unique"
        )
        for i, cm in enumerate(selected_CMs):
            logger_str = f"{logger_str}\n{i+1:2} {cm:42} {sd[cm].min().min():.3f} ... {sd[cm].mean().mean():.3f} ... {sd[cm].max().max():.3f} ... {np.unique(sd[cm])[:5]}"

        logger.info(logger_str)
        ActiveCMs = np.stack([sd.loc[c].loc[Ds].T for c in filtered_countries])
        assert ActiveCMs.shape == (nCs, nCMs, nDs)
        # [country, CM, day] Which CMs are active, and to what extent
        ActiveCMs = ActiveCMs.astype(theano.config.floatX)

        plt.figure(figsize=(4, 3), dpi=300)
        plt.imshow(sd.corr())
        plt.colorbar()
        plt.title("Selected CM Correlation")
        ax = plt.gca()
        ax.tick_params(axis="both", which="major", labelsize=6)
        plt.xticks(np.arange(len(selected_features)), [f"$\\alpha_{{{i+1}}}$" for i in range(len(selected_features))])
        plt.yticks(np.arange(len(selected_features)), [f"$\\alpha_{{{i + 1}}}$" for i in range(len(selected_features))])
        plt.show()

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
            f"Selected {len(filtered_countries)} Regions: f{filtered_countries}")

        Deaths = (
            johnhop_ds["Deaths"].loc[(tuple(filtered_countries), Ds)].unstack(1).values
        )
        assert Active.shape == (nCs, nDs)
        Deaths[Deaths < 10] = np.nan
        # [country, day]
        Deaths = np.ma.masked_invalid(Deaths.astype(theano.config.floatX))

        loaded_data = PreprocessedData(
            Active, Confirmed, ActiveCMs, selected_features, filtered_countries, Ds, Deaths
        )

        return loaded_data


class PreprocessedData(object):
    def __init__(self, Active, Confirmed, ActiveCMs, CMs, Rs, Ds, Deaths):
        super().__init__()
        self.Active = Active
        self.Confirmed = Confirmed
        self.Deaths = Deaths
        self.ActiveCMs = ActiveCMs
        self.Rs = Rs
        self.CMs = CMs
        self.Ds = Ds
