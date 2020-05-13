import unittest
import pytest
from . import PandasTestCase

import pandas as pd

from epimodel import Region, RegionDataset
import epimodel.gleam.scenario as sc


@pytest.mark.usefixtures("ut_datadir", "ut_rds")
class TestConfigParser(PandasTestCase):
    def test_output_format(self):
        parser = sc.ConfigParser(rds=self.rds)
        df = parser.get_config_from_csv(self.datadir / "scenario_config.csv")

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
        self.assertIsInstance(df["Region"].dropna().iloc[0], Region)
        self.assert_dtype(df["Value"], "float")
        self.assert_dtype(df["Start date"], "datetime64[ns]")
        self.assert_dtype(df["End date"], "datetime64[ns]")
