import pytest
from unittest.mock import Mock, patch
from . import PandasTestCase

import pandas as pd
import numpy as np

from epimodel import Region, RegionDataset
import epimodel.gleam.scenario as sc
from epimodel.gleam.definition import GleamDefinition


@pytest.mark.usefixtures("ut_datadir", "ut_rds")
class TestScenarioIntegration(PandasTestCase):
    timestamp_patcher = patch("pandas.Timestamp", autospec=True)

    def setUp(self):
        self.timestamp = pd.Timestamp('2020-05-01')
        self.Timestamp = self.timestamp_patcher.start()
        self.Timestamp.return_value = self.timestamp

    def tearDown(self):
        self.timestamp_patcher.stop()

    def test_integration(self):
        parser = sc.ConfigParser(rds=self.rds)
        df = parser.get_config_from_csv(self.datadir / "scenario/config.csv")
        simulations = sc.SimulationSet(df)
        for classes, definition in simulations:
            file_path = (
                "scenario/definitions/GLEAMviz_Test__%s__%s.xml" % classes
            ).replace(" ", "_")

            ### Uncomment the following line and run this test if you
            ### need to reset the files, but be sure to manually check
            ### the output afterwards to ensure it's correct.
            # definition.save(self.datadir / file_path)

            expected = GleamDefinition(self.datadir / file_path)
            definition.assert_equal(expected)


@pytest.mark.usefixtures("ut_datadir", "ut_rds")
class TestConfigParser(PandasTestCase):
    @staticmethod
    def config_from_rows(*rows):
        config = pd.DataFrame(rows, columns=sc.ConfigParser.FIELDS)
        return config

    def config_exception(self, **kwargs):
        config = self.config_from_rows(
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
        df = parser.get_config_from_csv(self.datadir / "scenario/config.csv")

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

    def test_converts_values_selectively(self):
        parser = sc.ConfigParser(rds=self.rds)
        df = parser.get_config(
            self.config_from_rows(
                ["", "test name", "name", "", "", "", ""],
                ["", "test.id", "id", "", "", "", ""],
                ["", "0.5", "beta", "", "", "", ""],
            )
        )
        self.assert_array_equal(df["Value"], pd.Series(["test name", "test.id", 0.5]))

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
    def_gen_patcher = patch("epimodel.gleam.scenario.DefinitionGenerator", autospec=True)

    def setUp(self):
        self.DefinitionGenerator = self.def_gen_patcher.start()
        self.DefinitionGenerator.definition_from_config = Mock(side_effect=lambda x, classes: x)

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
    definition_patcher = patch("epimodel.gleam.scenario.GleamDefinition", autospec=True)

    def setUp(self):
        self.GleamDefinition = self.definition_patcher.start()
        self.output = self.GleamDefinition.return_value

    def tearDown(self):
        self.definition_patcher.stop()

    def config_rows(self, *rows):
        return pd.concat([self.config_row(row) for row in rows]).reset_index()

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

    def config_exception(
        self, variables: dict, regions=["FR"], start="2020-05-01", end="2020-06-01"
    ):
        return self.config_rows(
            *(
                {
                    "Region": region,
                    "Value": v,
                    "Parameter": k,
                    "Start date": start,
                    "End date": end,
                }
                for region in regions
                for k, v in variables.items()
            )
        )

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

    def test_duplicate_parameter_fails(self):
        config = pd.concat(
            [
                self.config_row({"Parameter": "id", "Value": "abc",}),
                self.config_row({"Parameter": "id", "Value": "123",}),
            ]
        )
        self.assertRaises(ValueError, sc.DefinitionGenerator, config)
        self.output.set_id.assert_not_called()

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
        config = self.config_exception({"imu": 0.3})
        sc.DefinitionGenerator(config)
        self.output.add_exception.assert_called_once_with(
            (config["Region"][0],),
            {"imu": 0.3},
            config["Start date"][0],
            config["End date"][0],
        )
        self.output.set_compartment_variable.assert_not_called()

    def test_multivariate_exception(self):
        config = self.config_exception({"beta": 0.8, "imu": 0.3})
        sc.DefinitionGenerator(config)
        self.output.add_exception.assert_called_once_with(
            (config["Region"][0],),
            {"beta": 0.8, "imu": 0.3},
            config["Start date"][0],
            config["End date"][0],
        )
        self.output.set_compartment_variable.assert_not_called()

    def test_multi_region_exception(self):
        config = self.config_exception({"imu": 0.3}, ["FR", "PK"])
        sc.DefinitionGenerator(config)
        self.output.add_exception.assert_called_once_with(
            (self.rds["FR"], self.rds["PK"]),
            {"imu": 0.3},
            config["Start date"][0],
            config["End date"][0],
        )
        self.output.set_compartment_variable.assert_not_called()

    def test_multivariate_multi_region_exception(self):
        config = self.config_exception({"beta": 0.8, "imu": 0.3}, ["FR", "PK"])
        sc.DefinitionGenerator(config)
        self.output.add_exception.assert_called_once_with(
            (self.rds["FR"], self.rds["PK"]),
            {"beta": 0.8, "imu": 0.3},
            config["Start date"][0],
            config["End date"][0],
        )
        self.output.set_compartment_variable.assert_not_called()

    def test_multiple_exceptions(self):
        config = pd.concat(
            [
                self.config_exception({"beta": 0.8, "imu": 0.3}, ["FR", "PK"]),
                self.config_exception({"beta": 0.5, "imu": 0.2}, ["DE", "GB"]),
            ]
        ).reset_index()
        sc.DefinitionGenerator(config)
        self.output.add_exception.assert_any_call(
            (self.rds["FR"], self.rds["PK"]),
            {"beta": 0.8, "imu": 0.3},
            config["Start date"][0],
            config["End date"][0],
        )
        self.output.add_exception.assert_any_call(
            (self.rds["DE"], self.rds["GB"]),
            {"beta": 0.5, "imu": 0.2},
            config["Start date"][0],
            config["End date"][0],
        )
        self.output.set_compartment_variable.assert_not_called()

    def test_multiple_exception_dates(self):
        config = pd.concat(
            [
                self.config_exception({"mu": 0.5}, ["FR"], "2020-05-01", "2020-06-01"),
                self.config_exception({"mu": 0.5}, ["FR"], "2020-06-01", "2020-07-01"),
            ]
        ).reset_index()
        sc.DefinitionGenerator(config)
        self.output.add_exception.assert_any_call(
            (self.rds["FR"],),
            {"mu": 0.5},
            config["Start date"][0],
            config["End date"][0],
        )
        self.output.add_exception.assert_any_call(
            (self.rds["FR"],),
            {"mu": 0.5},
            config["Start date"][1],
            config["End date"][1],
        )
        self.output.set_compartment_variable.assert_not_called()

    # multipliers

    def test_multiplier_affects_one_parameter(self):
        config = pd.concat(
            [
                self.config_rows(
                    {"Value": 2.0, "Parameter": "beta multiplier"},
                    {"Value": 0.5, "Parameter": "mu"},
                    {"Value": 0.5, "Parameter": "beta"},
                ),
                self.config_exception({"mu": 1.0, "beta": 1.0}, ["FR"]),
            ]
        ).reset_index()
        sc.DefinitionGenerator(config)
        self.output.set_compartment_variable.assert_any_call("mu", 0.5)
        self.output.set_compartment_variable.assert_any_call("beta", 1.0)
        self.output.add_exception.assert_any_call(
            (self.rds["FR"],),
            {"mu": 1.0, "beta": 2.0},
            config["Start date"][3],
            config["End date"][3],
        )

    def test_invalid_multiplier(self):
        config = self.config_rows(
            {"Value": 2.0, "Parameter": "duration multiplier"},
            {"Value": 90.0, "Parameter": "duration"},
        )
        self.assertRaises(ValueError, sc.DefinitionGenerator, config)
