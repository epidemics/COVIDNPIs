from uuid import UUID
import numpy as np
import pandas as pd

import gspread
from oauth2client.client import GoogleCredentials

from tqdm import tqdm
import ergo
from epimodel import RegionDataset, Level, algorithms


def load_spreadsheet(sheet_url: str, tab_name: str):
    """ Export a DataFrame from a Google Sheets tab. The first row is
        used for column names, and index is set equal to the row number
        for easy cross-referencing. """
    client = gspread.authorize(GoogleCredentials.get_application_default())
    spreadsheet = client.open_by_url(sheet_url)
    values = spreadsheet.worksheet(tab_name).get_all_values()
    return pd.DataFrame(values[1:], columns=values[0], index=range(2, len(values) + 1))


class Exporter:
    FIELDS = [
        "Region",
        "Value",
        "Parameter",
        "Start date",
        "End date",
        "Type",
        "Class",
    ]

    def __init__(self, foretold_token=None, progress_bar=True, rds=None):
        self.foretold = ergo.Foretold(foretold_token) if foretold_token else None
        self.progress_bar = progress_bar
        self.rds = rds or RegionDataset.load(
            "epimodel/data/regions.csv", "epimodel/data/regions-gleam.csv"
        )
        algorithms.estimate_missing_populations(rds)

    def make_scenarios(self, sheet_url, tab_name):
        df = self.fetch_parameters_sheet(sheet_url, tab_name)

    def fetch_parameters_sheet(self, sheet_url, tab_name):
        df = load_spreadsheet(sheet_url, tab_name).replace({"": None})
        df = df[pd.notnull(df["Region"])][self.FIELDS].copy()
        df["Start date"] = df["Start date"].astype("datetime64[D]")
        df["End date"] = df["End date"].astype("datetime64[D]")
        df["Value"] = self._values_to_float(df["Value"])
        df["Region"] = df["Region"].apply(self._get_region_code)
        return df

    def _values_to_float(self, values: pd.Series):
        values = values.copy()
        uuid_filter = values.apply(self._is_uuid)
        values[uuid_filter] = [
            self._get_foretold_mean(uuid)
            for uuid in self._progress_bar(
                values[uuid_filter], desc="fetching Foretold distributions"
            )
        ]
        return values.astype("float")

    @staticmethod
    def _is_uuid(value):
        try:
            UUID(value, version=4)
            return True
        except ValueError:
            return False

    def _get_foretold_mean(self, uuid):
        question_distribution = self.foretold.get_question(uuid)
        # Sample the centers of 100 1%-wide quantiles
        qs = np.arange(0.005, 1.0, 0.01)
        ys = np.array([question_distribution.quantile(q) for q in qs])
        mean = np.sum(ys) / len(qs)
        return mean

    def _get_region_code(self, region):
        # try code first
        if region in self.rds:
            return region

        # If this fails, assume name. Match Gleam regions first.
        matches = self.rds.find_all_by_name(region, levels=Level)
        if not matches:
            raise Exception(f"No corresponding region found for {region!r}.")
        return matches[0]

    def _progress_bar(self, enum, desc=None):
        if self.progress_bar:
            return tqdm(enum, desc=desc)
        return enum
