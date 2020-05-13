import unittest
import pytest
from . import PandasTestCase

import pandas as pd
import numpy as np

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
            pd.isnull(df["Parameter"]).any(), "output has null parameter names"
        )

        # unsafe to compare numeric types to string b/c Python/Numpy disagreement
        # see https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
        non_numerics = [
            col for col in df.columns if not np.issubdtype(df[col].dtype, np.number)
        ]
        self.assertFalse(
            (df[non_numerics] == "").any().any(), "output contains empty strings"
        )

        # Column types
        self.assert_dtype(df["Region"], "object")
        self.assertIsInstance(df["Region"].dropna().iloc[0], Region)
        self.assert_dtype(df["Value"], "float")
        self.assert_dtype(df["Start date"], "M")
        self.assert_dtype(df["End date"], "M")
