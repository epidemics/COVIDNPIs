import pytest
from unittest.mock import Mock, patch
from . import PandasTestCase

import pandas as pd
import numpy as np

from epimodel import Region, RegionDataset
import epimodel.gleam.scenario as sc


@pytest.mark.usefixtures("ut_datadir", "ut_rds")
class TestConfigParser(PandasTestCase):
    @staticmethod
    def config_from_list(row):
        config = pd.DataFrame(columns=sc.ConfigParser.FIELDS)
        config.loc[2] = row
        return config

    def config_exception(self, **kwargs):
        config = self.config_from_list(
            [
                "PK",
                "0.35",
                "beta",
                "2020-04-14",
                "2021-05-01",
                "Countermeasure package",
                "Strong",
            ]
        )
        for k, v in kwargs.items():
            config.loc[:, k] = v
        return config

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

    def test_find_region_by_code(self):
        parser = sc.ConfigParser(rds=self.rds)
        df = parser.get_config(self.config_exception(Region="FR"))
        self.assertEqual(df["Region"].iloc[0], self.rds["FR"])

    def test_find_region_by_name(self):
        parser = sc.ConfigParser(rds=self.rds)
        df = parser.get_config(self.config_exception(Region="France"))
        self.assertEqual(df["Region"].iloc[0], self.rds["FR"])

    @patch("epimodel.gleam.scenario.ergo")
    def test_foretold_lookup(self, ergo):
        """
        If the Value field contains a valid UUID, it should be used to
        obtain a Foretold question distribution which is then averaged.
        """

        QUESTION_ID = "682befc6-3c19-48b0-98a0-bf52c5221c06"

        question = Mock()
        question.quantile = Mock(return_value=1)
        foretold = Mock()
        foretold.get_question = Mock(return_value=question)
        ergo.Foretold = Mock(return_value=foretold)

        parser = sc.ConfigParser(rds=self.rds, foretold_token="ABC", progress_bar=False)

        df = parser.get_config(self.config_exception(Value=QUESTION_ID))

        ergo.Foretold.assert_called_once_with("ABC")
        foretold.get_question.assert_called_once_with(QUESTION_ID)
        question.quantile.assert_called()

        self.assertEqual(df["Value"].iloc[0], 1)


class TestSimulationSet(PandasTestCase):
    def_gen_patcher = patch("epimodel.gleam.scenario.DefinitionGenerator", spec=True)

    def setUp(self):
        self.DefinitionGenerator = self.def_gen_patcher.start()
        self.output = Mock(side_effect=lambda x: x)
        self.DefinitionGenerator.definition_from_config = self.output

    def tearDown(self):
        self.def_gen_patcher.stop()

    def get_config(self):
        """
        Generates a simplified, invalid config that still has everything this class uses
        """
        return pd.DataFrame(
            [
                ["AC AD", "Countermeasure package", "A"],
                ["BC BD", "Countermeasure package", "B"],
                ["AC BC", "Background condition", "C"],
                ["AD BD", "Background condition", "D"],
                ["AC AD BC BD", None, None],
            ],
            columns=["present_in", "Type", "Class"],
        )

    def test_output(self):
        config = self.get_config()
        ss = sc.SimulationSet(config)

        for package_class in ["A", "B"]:
            for background_class in ["C", "D"]:
                pair = (package_class, background_class)
                self.assertIn(pair, ss)

                expected_output = config[config.present_in.str.contains("".join(pair))]
                self.assert_array_equal(ss[pair], expected_output)
                self.output.assert_called_with(expected_output)
