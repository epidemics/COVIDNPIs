import pytest
from unittest.mock import Mock, patch
from . import PandasTestCase

import pandas as pd
import numpy as np

from epimodel import Region, RegionDataset
import epimodel.gleam.scenario as sc


class ConfigTestCase(PandasTestCase):
    @staticmethod
    def config_from_rows(*rows, columns=sc.ConfigParser.FIELDS):
        return pd.DataFrame(rows, columns=columns)

    def config_row(self, row, **overrides):
        if row is None:
            row = [None for _ in sc.ConfigParser.FIELDS]
        config = self.config_from_rows(row)
        for k, v in overrides.items():
            config.loc[:, k] = v
        return config


@pytest.mark.usefixtures("ut_datadir", "ut_rds")
class TestConfigParser(ConfigTestCase):
    def config_exception(self, **overrides):
        return self.config_row(
            [
                "PK",
                "0.35",
                "beta",
                "2020-04-14",
                "2021-05-01",
                "Countermeasure package",
                "Strong",
            ],
            **overrides,
        )

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
        self.DefinitionGenerator.definition_from_config = Mock(side_effect=lambda x: x)

    def tearDown(self):
        self.def_gen_patcher.stop()

    def get_config(self):
        """
        Generates a simplified, invalid config that still has everything
        this class uses. Order is important because of the way the test
        filters this, but not important for SimulationSet itself.
        """
        return pd.DataFrame(
            [
                ["AC AD", "Countermeasure package", "A"],
                ["BC BD", "Countermeasure package", "B"],
                ["AC AD BC BD", "Countermeasure package", None],
                ["AC BC", "Background condition", "C"],
                ["AD BD", "Background condition", "D"],
                ["AC AD BC BD", None, None],  # None is assumed background
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


@pytest.mark.usefixtures("ut_rds")
class TestDefinitionGenerator(PandasTestCase):
    definition_patcher = patch("epimodel.gleam.scenario.GleamDefinition", spec=True)

    def setUp(self):
        self.GleamDefinition = self.definition_patcher.start()
        self.output = self.GleamDefinition.return_value

    def tearDown(self):
        self.definition_patcher.stop()

    def config_row(self, row):
        if row.get("Region") is not None:
            row["Region"] = self.rds[row["Region"]]
        if row.get("Start date") is not None:
            row["Start date"] = pd.to_datetime(row["Start date"])
        if row.get("End date") is not None:
            row["End date"] = pd.to_datetime(row["End date"])

        config = pd.DataFrame(columns=sc.ConfigParser.FIELDS)
        config.loc[0] = pd.Series(row)
        return config

    def test_run_dates(self):
        config = self.config_row(
            {
                "Parameter": "run dates",
                "Start date": "2020-04-14",
                "End date": "2021-05-01",
            }
        )
        sc.DefinitionGenerator(config)
        self.output.set_start_date.assert_called_once_with(config["Start date"][0])
        self.output.set_end_date.assert_called_once_with(config["End date"][0])

    def test_run_dates_only_start(self):
        config = self.config_row(
            {"Parameter": "run dates", "Start date": "2020-04-14",}
        )
        sc.DefinitionGenerator(config)
        self.output.set_start_date.assert_called_once_with(config["Start date"][0])
        self.output.set_end_date.assert_not_called()

    def test_run_dates_only_end(self):
        config = self.config_row({"Parameter": "run dates", "End date": "2021-05-01",})
        sc.DefinitionGenerator(config)
        self.output.set_start_date.assert_not_called()
        self.output.set_end_date.assert_called_once_with(config["End date"][0])

    # global parameters

    def test_name(self):
        value = "GLEAMviz test run"
        config = self.config_row({"Parameter": "name", "Value": value,})
        sc.DefinitionGenerator(config)
        self.output.set_name.assert_called_once_with(value)

    def test_id(self):
        value = "1234567.890"
        config = self.config_row({"Parameter": "id", "Value": value,})
        sc.DefinitionGenerator(config)
        self.output.set_id.assert_called_once_with(value)

    def test_duration(self):
        value = 180.0  # days
        config = self.config_row({"Parameter": "duration", "Value": value,})
        sc.DefinitionGenerator(config)
        self.output.set_duration.assert_called_once_with(value)

    def test_number_of_runs(self):
        value = 5
        config = self.config_row({"Parameter": "number of runs", "Value": value,})
        sc.DefinitionGenerator(config)
        self.output.set_run_count.assert_called_once_with(value)

    def test_airline_traffic(self):
        value = 0.3
        config = self.config_row({"Parameter": "airline traffic", "Value": value,})
        sc.DefinitionGenerator(config)
        self.output.set_airline_traffic.assert_called_once_with(value)

    def test_seasonality(self):
        value = 0.6
        config = self.config_row({"Parameter": "seasonality", "Value": value,})
        sc.DefinitionGenerator(config)
        self.output.set_seasonality.assert_called_once_with(value)

    def test_commuting_time(self):
        value = 7.5
        config = self.config_row({"Parameter": "commuting time", "Value": value,})
        sc.DefinitionGenerator(config)
        self.output.set_commuting_rate.assert_called_once_with(value)

    # global compartment variables

    def test_compartment_variable(self):
        value = 0.3
        config = self.config_row({"Parameter": "imu", "Value": value,})
        sc.DefinitionGenerator(config)
        self.output.set_compartment_variable.assert_called_once_with("imu", value)
        self.output.add_exception.assert_not_called()

    def test_partial_exception_fails_region(self):
        config = self.config_row({"Region": "FR", "Parameter": "imu", "Value": 0.3,})
        self.assertRaises(ValueError, sc.DefinitionGenerator, config)
        self.output.set_compartment_variable.assert_not_called()
        self.output.add_exception.assert_not_called()

    def test_partial_exception_fails_dates(self):
        config = self.config_row(
            {
                "Parameter": "imu",
                "Value": 0.3,
                "Start date": "2020-05-01",
                "End date": "2020-06-01",
            }
        )
        self.assertRaises(ValueError, sc.DefinitionGenerator, config)
        self.output.set_compartment_variable.assert_not_called()
        self.output.add_exception.assert_not_called()

    # exceptions

    def test_single_exception(self):
        config = self.config_row(
            {
                "Region": "FR",
                "Parameter": "imu",
                "Value": 0.3,
                "Start date": "2020-05-01",
                "End date": "2020-06-01",
            }
        )
        sc.DefinitionGenerator(config)
        self.output.add_exception.assert_called_with(
            tuple(config["Region"]),
            {"imu": 0.3},
            config["Start date"][0],
            config["End date"][0],
        )
        self.output.set_compartment_variable.assert_not_called()
