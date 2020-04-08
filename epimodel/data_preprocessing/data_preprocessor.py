import pandas as pd
import numpy as np
import theano
import matplotlib.pyplot as plt

from epimodel import read_csv, RegionDataset

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
            "min_num_confirmed_mask": self.min_num_confirmed_mask
        }

    def preprocess_data(self, data_base_path, countries, cms01_cols):
        # at the moment only features from the 0-1 countermeasures dataset
        Ds = pd.date_range(start=self.start_date, end=self.end_date, tz='utc')
        nDs = len(Ds)

        region_ds = RegionDataset.load(os.path.join(data_base_path, "regions.csv"))
        johnhop_ds = read_csv(os.path.join(data_base_path, "johns-hopkins.csv"))

        cm_set_dirs = ['countermeasures-features.csv', 'countermeasures-model-0to1.csv',
                       'countermeasures-selected-binary.csv']

        cm_sets = {cm_set_file: read_csv(os.path.join(data_base_path, cm_set_file)) for cm_set_file in cm_set_dirs}
        for n, v in cm_sets.items():
            logger.debug(f"\nCMS {n} columns:\n{v.columns!r}")

        selected_CMs = cms01_cols
        CM_dataset = cm_sets['countermeasures-model-0to1.csv']
        nCMs = len(cms01_cols)

        filtered_countries = []
        for cc in set(countries):
            c = region_ds[cc]
            if c.Level == "country" and c.Code in johnhop_ds.index and c.Code in CM_dataset.index:
                if johnhop_ds.loc[(c.Code, Ds[-1]), "Active"] > self.min_final_num_active_cases:
                    filtered_countries.append(c.Code)
        nCs = len(filtered_countries)

        sd = CM_dataset.loc[filtered_countries, selected_CMs]
        if 'Mask wearing' in selected_CMs:
            sd['Mask wearing'] *= 0.01

        logger.info("\nCountermeasures                               min   .. mean  .. max")
        for i, cm in enumerate(selected_CMs):
            logger.info(
                f"{i:2} {cm:42} {sd[cm].min().min():.3f} .. {sd[cm].mean().mean():.3f} .. {sd[cm].max().max():.3f}")
        ActiveCMs = np.stack([sd.loc[c].loc[Ds].T for c in filtered_countries])
        assert ActiveCMs.shape == (nCs, nCMs, nDs)
        # [country, CM, day] Which CMs are active, and to what extent
        ActiveCMs = ActiveCMs.astype(theano.config.floatX)

        plt.figure(figsize=(4, 3), dpi=150)
        plt.imshow(sd.corr())
        plt.colorbar()
        plt.title("Selected CM Correlation")
        plt.show()

        dataset_size = (nCs, nCMs, nDs)

        Confirmed = johnhop_ds["Confirmed"].loc[(tuple(filtered_countries), Ds)].unstack(1).values
        assert Confirmed.shape == (nCs, nDs)
        Confirmed[Confirmed < self.min_num_confirmed_mask] = np.nan
        Confirmed = np.ma.masked_invalid(Confirmed.astype(theano.config.floatX))

        # Active cases, masking values smaller than 10
        Active = johnhop_ds["Active"].loc[(tuple(filtered_countries), Ds)].unstack(1).values
        assert Active.shape == (nCs, nDs)
        Active[Active < self.min_num_active_mask] = np.nan
        # [country, day]
        Active = np.ma.masked_invalid(Active.astype(theano.config.floatX))

        logger.info(f"Data Preprocessing Complete using:\n\n{json.dumps(self.generate_params_dict(), indent=4)}")

        return dataset_size, ActiveCMs, Active, Confirmed
