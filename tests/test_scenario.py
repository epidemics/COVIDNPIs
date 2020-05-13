import unittest
import pytest
from . import PandasTestCase

import pandas as pd

import epimodel
from epimodel.utils import utc_date
import epimodel.gleam.scenario as sc


class TestConfigParser(PandasTestCase):
    def test_output_format(self):
        parser = sc.ConfigParser(
            rds=epimodel.RegionDataset.load(
                "../data/regions.csv", "../data/regions-gleam.csv"
            )
        )
        df = parser.get_config_from_csv("data/scenario_config.csv")  # self.config_csv)

        self.assert_array_equal(
            df.columns, sc.ConfigParser.FIELDS, "output columns do not match"
        )
        self.assertFalse(
            df.apply(lambda x: x == "").any().any(), "output contains empty strings"
        )
        self.assertFalse(
            pd.isnull(df["Parameter"]).any(), "output has null parameter names"
        )

        # Column types
        self.assert_dtype(df["Region"], "O")
        self.assertIsInstance(df["Region"].dropna().iloc[0], epimodel.Region)
        self.assert_dtype(df["Value"], "float")
        self.assert_dtype(df["Start date"], "datetime64[ns]")
        self.assert_dtype(df["End date"], "datetime64[ns]")
