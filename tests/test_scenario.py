import pytest
from unittest.mock import Mock, patch, call
from . import PandasTestCase

from glob import glob
import yaml
import pandas as pd
import numpy as np

from epimodel import Region, Level
from epimodel.gleam import Batch, GleamDefinition
import epimodel.gleam.scenario as sc


@pytest.mark.usefixtures("ut_datadir", "ut_rds", "ut_tmp_path")
class TestScenarioIntegration(PandasTestCase):
    timestamp_patcher = patch("pandas.Timestamp.utcnow", autospec=True)

    def setUp(self):
        # patch timestamp to be constant for consistent test results
        self.timestamp = pd.Timestamp("2020-05-01", tz="UTC")
        self.utcnow = self.timestamp_patcher.start()
        self.utcnow.return_value = self.timestamp

        self.xml_patcher = patch(
            "epimodel.gleam.definition.GleamDefinition.DEFAULT_XML_FILE",
            self.datadir / "default_gleam_definition.xml",
        )
        self.xml_patcher.start()

    def tearDown(self):
        self.timestamp_patcher.stop()
        self.xml_patcher.stop()

    def test_integration(self):
        with open(self.datadir / "scenario/config.yaml", "r") as fp:
            config = yaml.safe_load(fp)["scenarios"]

        parser = sc.InputParser(rds=self.rds)
        params = parser.parse_parameters_df(
            pd.read_csv(self.datadir / config["parameters"])
        )
        estimates = parser.parse_estimates_df(
            pd.read_csv(self.datadir / config["estimates"])
        )

        # check created definitions
        simulations = sc.SimulationSet(config, params, estimates)
        for classes, def_builder in simulations:
            dir = self.datadir / "scenario/definitions"

            ### Uncomment the following lines and run this test if you
            ### need to reset the files, but be sure to manually check
            ### the output afterwards to ensure it's correct.
            # def_builder.save_to_dir(dir)
            # with open(dir / def_builder.filename, 'a') as fp: fp.write("\n")

            expected = GleamDefinition(dir / def_builder.filename)
            def_builder.definition.assert_equal(expected)

        # check Batch integration
        batch = Batch.new(dir=self.tmp_path)
        simulations.add_to_batch(batch)

        batch.export_definitions_to_gleam(self.tmp_path)
        batch.close()

        # ensure Batch outputs correctly
        for _, def_builder in simulations:
            def_builder.save_to_dir(self.tmp_path)
            id_str = def_builder.definition.get_id_str()

            test_path = self.tmp_path / def_builder.filename
            batch_path = self.tmp_path / f"{id_str}.gvh5/definition.xml"

            with open(test_path, "r") as tfp, open(batch_path, "r") as bfp:
                self.assertEqual(tfp.read(), bfp.read())


@pytest.mark.usefixtures("ut_datadir", "ut_rds")
class TestInputParser(PandasTestCase):
    @staticmethod
    def estimates_input_from_rows(*rows):
        rows = [row + [None, None] for row in rows]
        estimates = pd.DataFrame.from_records(
            rows, columns=["Name", "Infectious_mean", "Beta1", "Beta2"]
        )
        return estimates

    @staticmethod
    def params_input_from_rows(*rows):
        params = pd.DataFrame.from_records(
            rows, columns=sc.InputParser.PARAMETER_FIELDS
        )
        return params

    def exception_params_input(self, **kwargs):
        params = self.params_input_from_rows(
            ["PK", "0.35", "beta", "2020-04-14", "2021-05-01", "group", "Strong",]
        )
        for k, v in kwargs.items():
            params.loc[:, k] = v
        return params

    def test_estimates_output_format(self):
        parser = sc.InputParser(rds=self.rds)
        df = parser.parse_estimates_df(pd.read_csv(self.datadir / "estimates.csv"))

        self.assert_dtype(df["Region"], "object")
        self.assertIsInstance(df["Region"].iloc[0], Region)
        self.assert_dtype(df["Infectious"], "float")

    def test_estimates_distribute_down(self):
        """
        Estimates should get spread down to regions at the gleam_basin level
        """
        raw_estimates = self.estimates_input_from_rows(["BE", "100"], ["MA", "100"])
        parser = sc.InputParser(rds=self.rds)
        df = parser.parse_estimates_df(raw_estimates)

        for region in df["Region"]:
            self.assertEqual(region.Level, Level.gleam_basin)
        self.assert_almost_equal(df.Infectious.sum(), 200)

    def test_params_output_format(self):
        parser = sc.InputParser(rds=self.rds)
        df = parser.parse_parameters_df(
            pd.read_csv(self.datadir / "scenario/parameters.csv")
        )

        self.assert_array_equal(
            df.columns, sc.InputParser.PARAMETER_FIELDS, "output columns do not match"
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
        parser = sc.InputParser(rds=self.rds)
        df = parser.parse_parameters_df(self.exception_params_input(Region="FR"))
        self.assertEqual(df["Region"].iloc[0], self.rds["FR"])

    def test_find_region_by_name(self):
        parser = sc.InputParser(rds=self.rds)
        df = parser.parse_parameters_df(self.exception_params_input(Region="France"))
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

        parser = sc.InputParser(rds=self.rds, foretold_token="ABC", progress_bar=False)

        df = parser.parse_parameters_df(self.exception_params_input(Value=QUESTION_ID))

        ergo.Foretold.assert_called_once_with("ABC")
        foretold.get_question.assert_called_once_with(QUESTION_ID)
        question.quantile.assert_called()

        self.assertEqual(df["Value"].iloc[0], 1)


@pytest.mark.usefixtures("ut_rds")
class TestSimulationSet(PandasTestCase):
    def_builder_patcher = patch(
        "epimodel.gleam.scenario.DefinitionBuilder", autospec=True
    )

    def setUp(self):
        self.DefinitionBuilder = self.def_builder_patcher.start()
        self.DefinitionBuilder.side_effect = self._return_inputs

    def tearDown(self):
        self.def_builder_patcher.stop()

    def _return_inputs(self, parameters, estimates, id, name, classes):
        """
        return inputs instead of DefinitionBuilder instance for easy
        inspection of test results
        """
        return parameters, estimates, id, name, classes

    def get_estimates(self, *rows):
        return pd.DataFrame.from_records(rows, columns=sc.InputParser.ESTIMATE_FIELDS)

    def get_params(self):
        """
        Generates a simplified, invalid config that still has everything
        this class uses. Order is important because of the way the test
        filters this, but not important for SimulationSet itself.
        """
        return pd.DataFrame(
            [
                ["AC AD", "group", "A"],
                ["BC BD", "group", "B"],
                ["AC AD BC BD", "group", None],
                ["AC BC", "trace", "C"],
                ["AC AD BC BD", None, None],
            ],
            columns=["present_in", "Type", "Class"],
        )

    def test_params_output(self):
        config = {
            "name": "Test Params",
            "groups": [{"name": "A"}, {"name": "B"},],
            "traces": [{"name": "C"}, {"name": "D"},],
            "compartment_multipliers": {"Infectious": 1.0,},
            "compartments_max_fraction": 1.0,
        }
        params = self.get_params()
        estimates = self.get_estimates([self.rds["G-MXP"], 100])

        ss = sc.SimulationSet(config, params, estimates)

        # None is assumed trace
        params["Type"].fillna("trace", inplace=True)

        ids = set()

        for group in ["A", "B"]:
            for trace in ["C", "D"]:
                pair = (group, trace)
                self.assertIn(pair, ss)

                expected_params = params[params.present_in.str.contains("".join(pair))]

                out_params, out_estimates, id, name, classes = ss[pair]
                self.assert_array_equal(out_params, expected_params)
                self.assert_array_equal(out_estimates, estimates)
                self.assertEqual(classes, pair)
                self.assertEqual(name, config["name"])
                self.assertIsInstance(id, int)
                ids.add(id)
        self.assertEqual(len(ids), 4, "not all ids are unique")

    def test_prepare_estimates(self):
        config = {
            "name": "Test Exposed",
            "groups": [{"name": "A"},],
            "traces": [{"name": "C"},],
            "compartment_multipliers": {"Infectious": 1.0, "Exposed": 1.8,},
            "compartments_max_fraction": 1.0,
        }

        # mock each gleam_basin in Belgium
        regions = [Mock(autospec=region) for region in self.rds["BE"].children]
        regions[0].Population = np.nan  # ensure null population is handled
        regions[1].Population = 1000
        regions[2].Population = 10000

        params = self.get_params()
        params = params[params.present_in.str.contains("AC")]
        estimates = self.get_estimates(*([region, 100] for region in regions))

        ss = sc.SimulationSet(config, params, estimates)

        expected_estimates = estimates.copy()
        expected_estimates["Exposed"] = 180
        out_estimates = ss[("A", "C")][1]
        self.assert_array_equal(out_estimates, expected_estimates)

    def test_prepare_estimates_max_fraction(self):
        config = {
            "name": "Test Exposed",
            "groups": [{"name": "A"},],
            "traces": [{"name": "C"},],
            "compartment_multipliers": {"Infectious": 1.0, "Exposed": 9.0,},
            "compartments_max_fraction": 0.5,
        }
        region = Mock(autospec=self.rds["G-MLA"])
        region.Population = 100

        params = self.get_params()
        params = params[params.present_in.str.contains("AC")]
        estimates = self.get_estimates([region, 10])

        ss = sc.SimulationSet(config, params, estimates)

        # total estimate == population
        # compartments_max_fraction == 0.5 * population
        # so output will be half of input
        expected_estimates = self.get_estimates([region, 5])
        expected_estimates["Exposed"] = 45

        out_estimates = ss[("A", "C")][1]
        self.assert_array_equal(out_estimates, expected_estimates)


@pytest.mark.usefixtures("ut_rds")
class TestDefinitionBuilder(PandasTestCase):
    definition_patcher = patch("epimodel.gleam.scenario.GleamDefinition", autospec=True)

    def setUp(self):
        self.GleamDefinition = self.definition_patcher.start()
        self.output = self.GleamDefinition.return_value

    def tearDown(self):
        self.definition_patcher.stop()

    def estimates_rows(self, *rows):
        return pd.DataFrame.from_records(rows)

    def params_rows(self, *rows):
        return pd.concat([self.params_row(row) for row in rows]).reset_index()

    def params_row(self, row):
        if row.get("Region") is not None:
            row["Region"] = self.rds[row["Region"]]
        if row.get("Start date") is not None:
            row["Start date"] = pd.to_datetime(row["Start date"])
        if row.get("End date") is not None:
            row["End date"] = pd.to_datetime(row["End date"])

        params = pd.DataFrame(columns=sc.InputParser.PARAMETER_FIELDS)
        if row:
            params.loc[0] = pd.Series(row)
        return params

    def params_exception(
        self, variables: dict, regions=["FR"], start="2020-05-01", end="2020-06-01"
    ):
        return self.params_rows(
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

    @staticmethod
    def init_def_builder(params=None, estimates=None, id=None, name=None, classes=None):
        if params is None:
            params = pd.DataFrame(columns=sc.InputParser.PARAMETER_FIELDS)
        if estimates is None:
            estimates = pd.DataFrame(columns=["Region"])
        if id is None:
            id = 123
        if name is None:
            name = "Testing"
        if classes is None:
            classes = ("A", "B")
        return sc.DefinitionBuilder(params, estimates, id, name, classes)

    # estimates

    def test_set_seeds(self):
        estimates = self.estimates_rows(
            {"Region": self.rds["G-MLA"], "Exposed": 100, "Infectious": 10}
        )
        self.init_def_builder(estimates=estimates)
        self.output.set_seeds.assert_called_once_with(estimates)

    def test_set_multiple_seeds(self):
        estimates = self.estimates_rows(
            {"Region": self.rds["G-KGL"], "Exposed": 100, "Infectious": 10},
            {"Region": self.rds["G-KME"], "Exposed": 50, "Infectious": 5},
        )
        self.init_def_builder(estimates=estimates)
        self.output.set_seeds.assert_called_once_with(estimates)

    # configuration

    def test_id(self):
        def_builder = self.init_def_builder(id=123)
        self.output.set_id.assert_called_once_with(123)

        self.output.get_id_str.return_value = "123.574"
        self.assertEqual(def_builder.filename, "123.574.xml")

    def test_name(self):
        self.init_def_builder(name="Test", classes=("A", "B"))
        self.output.set_name.assert_called_with("Test (A + B)")

    # run dates

    def test_run_dates(self):
        params = self.params_row(
            {
                "Parameter": "run dates",
                "Start date": "2020-04-14",
                "End date": "2021-05-01",
            }
        )
        self.init_def_builder(params)
        self.output.set_start_date.assert_called_once_with(params["Start date"][0])
        self.output.set_end_date.assert_called_once_with(params["End date"][0])

    def test_run_dates_only_start(self):
        params = self.params_row(
            {"Parameter": "run dates", "Start date": "2020-04-14",}
        )
        self.init_def_builder(params)
        self.output.set_start_date.assert_called_once_with(params["Start date"][0])
        self.output.set_end_date.assert_not_called()

    def test_run_dates_only_end(self):
        params = self.params_row({"Parameter": "run dates", "End date": "2021-05-01",})
        self.init_def_builder(params)
        self.output.set_start_date.assert_not_called()
        self.output.set_end_date.assert_called_once_with(params["End date"][0])

    # global parameters

    def test_name(self):
        value = "GLEAMviz test"
        params = self.params_row({"Parameter": "name", "Value": value,})
        self.init_def_builder(params)
        self.output.set_name.assert_called_with(value)

    def test_duration(self):
        value = 180.0  # days
        params = self.params_row({"Parameter": "duration", "Value": value,})
        self.init_def_builder(params)
        self.output.set_duration.assert_called_once_with(value)

    def test_number_of_runs(self):
        value = 5
        params = self.params_row({"Parameter": "number of runs", "Value": value,})
        self.init_def_builder(params)
        self.output.set_run_count.assert_called_once_with(value)

    def test_airline_traffic(self):
        value = 0.3
        params = self.params_row({"Parameter": "airline traffic", "Value": value,})
        self.init_def_builder(params)
        self.output.set_airline_traffic.assert_called_once_with(value)

    def test_seasonality(self):
        value = 0.6
        params = self.params_row({"Parameter": "seasonality", "Value": value,})
        self.init_def_builder(params)
        self.output.set_seasonality.assert_called_once_with(value)

    def test_commuting_time(self):
        value = 7.5
        params = self.params_row({"Parameter": "commuting time", "Value": value,})
        self.init_def_builder(params)
        self.output.set_commuting_rate.assert_called_once_with(value)

    def test_duplicate_parameter_fails(self):
        params = pd.concat(
            [
                self.params_row({"Parameter": "duration", "Value": 90,}),
                self.params_row({"Parameter": "duration", "Value": 180,}),
            ]
        )
        self.assertRaises(ValueError, self.init_def_builder, params)
        self.output.set_duration.assert_not_called()

    # global compartment variables

    def test_compartment_variable(self):
        value = 0.3
        params = self.params_row({"Parameter": "imu", "Value": value,})
        self.init_def_builder(params)
        self.output.set_compartment_variable.assert_called_once_with("imu", value)
        self.output.add_exception.assert_not_called()

    def test_partial_exception_fails_region(self):
        params = self.params_row({"Region": "FR", "Parameter": "imu", "Value": 0.3,})
        self.assertRaises(ValueError, self.init_def_builder, params)
        self.output.set_compartment_variable.assert_not_called()
        self.output.add_exception.assert_not_called()

    def test_partial_exception_fails_dates(self):
        params = self.params_row(
            {
                "Parameter": "imu",
                "Value": 0.3,
                "Start date": "2020-05-01",
                "End date": "2020-06-01",
            }
        )
        self.assertRaises(ValueError, self.init_def_builder, params)
        self.output.set_compartment_variable.assert_not_called()
        self.output.add_exception.assert_not_called()

    # exceptions

    def test_single_exception(self):
        params = self.params_exception({"imu": 0.3})
        self.init_def_builder(params)
        self.output.add_exception.assert_called_once_with(
            (params["Region"][0],),
            {"imu": 0.3},
            params["Start date"][0],
            params["End date"][0],
        )
        self.output.set_compartment_variable.assert_not_called()

    def test_multivariate_exception(self):
        params = self.params_exception({"beta": 0.8, "imu": 0.3})
        self.init_def_builder(params)
        self.output.add_exception.assert_called_once_with(
            (params["Region"][0],),
            {"beta": 0.8, "imu": 0.3},
            params["Start date"][0],
            params["End date"][0],
        )
        self.output.set_compartment_variable.assert_not_called()

    def test_multi_region_exception(self):
        params = self.params_exception({"imu": 0.3}, ["FR", "PK"])
        self.init_def_builder(params)
        self.output.add_exception.assert_called_once_with(
            (self.rds["FR"], self.rds["PK"]),
            {"imu": 0.3},
            params["Start date"][0],
            params["End date"][0],
        )
        self.output.set_compartment_variable.assert_not_called()

    def test_multivariate_multi_region_exception(self):
        params = self.params_exception({"beta": 0.8, "imu": 0.3}, ["FR", "PK"])
        self.init_def_builder(params)
        self.output.add_exception.assert_called_once_with(
            (self.rds["FR"], self.rds["PK"]),
            {"beta": 0.8, "imu": 0.3},
            params["Start date"][0],
            params["End date"][0],
        )
        self.output.set_compartment_variable.assert_not_called()

    def test_multiple_exceptions(self):
        params = pd.concat(
            [
                self.params_exception({"beta": 0.8, "imu": 0.3}, ["FR", "PK"]),
                self.params_exception({"beta": 0.5, "imu": 0.2}, ["DE", "GB"]),
            ]
        ).reset_index()
        self.init_def_builder(params)
        self.output.add_exception.assert_any_call(
            (self.rds["FR"], self.rds["PK"]),
            {"beta": 0.8, "imu": 0.3},
            params["Start date"][0],
            params["End date"][0],
        )
        self.output.add_exception.assert_any_call(
            (self.rds["DE"], self.rds["GB"]),
            {"beta": 0.5, "imu": 0.2},
            params["Start date"][0],
            params["End date"][0],
        )
        self.output.set_compartment_variable.assert_not_called()

    def test_multiple_exception_dates(self):
        params = pd.concat(
            [
                self.params_exception({"mu": 0.5}, ["FR"], "2020-05-01", "2020-06-01"),
                self.params_exception({"mu": 0.5}, ["FR"], "2020-06-01", "2020-07-01"),
            ]
        ).reset_index()
        self.init_def_builder(params)
        self.output.add_exception.assert_any_call(
            (self.rds["FR"],),
            {"mu": 0.5},
            params["Start date"][0],
            params["End date"][0],
        )
        self.output.add_exception.assert_any_call(
            (self.rds["FR"],),
            {"mu": 0.5},
            params["Start date"][1],
            params["End date"][1],
        )
        self.output.set_compartment_variable.assert_not_called()

    # multipliers

    def test_multiplier_affects_one_parameter(self):
        params = pd.concat(
            [
                self.params_rows(
                    {"Value": 2.0, "Parameter": "beta multiplier"},
                    {"Value": 0.5, "Parameter": "mu"},
                    {"Value": 0.5, "Parameter": "beta"},
                ),
                self.params_exception({"mu": 1.0, "beta": 1.0}, ["FR"]),
            ]
        ).reset_index()
        self.init_def_builder(params)
        self.output.set_compartment_variable.assert_any_call("mu", 0.5)
        self.output.set_compartment_variable.assert_any_call("beta", 1.0)
        self.output.add_exception.assert_any_call(
            (self.rds["FR"],),
            {"mu": 1.0, "beta": 2.0},
            params["Start date"][3],
            params["End date"][3],
        )

    def test_invalid_multiplier(self):
        params = self.params_rows(
            {"Value": 2.0, "Parameter": "duration multiplier"},
            {"Value": 90.0, "Parameter": "duration"},
        )
        self.assertRaises(ValueError, self.init_def_builder, params)
