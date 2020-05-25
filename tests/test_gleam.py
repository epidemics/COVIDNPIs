import pytest
from unittest.mock import Mock, NonCallableMagicMock
from . import PandasTestCase

import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET
import epimodel
from epimodel.utils import utc_date
from epimodel.gleam.definition import GleamDefinition


def test_batch_new_open(tmp_path):
    b = epimodel.gleam.Batch.new(path=tmp_path / "batch.hdf")
    path = b.path
    b.close()
    b2 = epimodel.gleam.Batch.open(path)
    b2.close()


@pytest.mark.usefixtures("ut_datadir", "ut_rds")
class TestGleamDefinition(PandasTestCase):
    def setUp(self):
        self.defn = GleamDefinition(self.datadir / "test_definition.xml")

    def assert_xml_equal(self, node1, node2, path="/"):
        if isinstance(node1, str):
            node1 = ET.fromstring(node1)
        if isinstance(node2, str):
            node2 = ET.fromstring(node2)
        self.defn.etree_assert_equal(node1, node2, path)

    def get_estimates(self):
        return pd.DataFrame(
            {"Infectious": [1, 2, 3], "Exposed": [4, 5, np.nan]},
            index=["G-MLA", "G-KGL", "G-KME"],
        )

    def test_load_definition(self):
        self.assertEqual(self.defn.get_id(), 1585188102568)
        self.assert_approx_equal(self.defn.get_variable("beta"), 1.01)
        self.assertEqual(self.defn.get_traffic_occupancy(), 20)
        self.assertEqual(self.defn.get_start_date(), utc_date("2020-03-25"))
        self.assert_approx_equal(self.defn.get_seasonality(), 0.85)

    def test_load_from_xml_string(self):
        with open(self.datadir / "test_definition.xml") as xml_file:
            defn = GleamDefinition.from_xml_string(xml_file.read())
        self.assertEqual(defn.get_id(), 1585188102568)
        self.assert_approx_equal(defn.get_variable("beta"), 1.01)
        self.assertEqual(defn.get_traffic_occupancy(), 20)
        self.assertEqual(defn.get_start_date(), utc_date("2020-03-25"))
        self.assert_approx_equal(defn.get_seasonality(), 0.85)

    def test_set_start_duration(self):
        self.defn.set_start_date("2020-10-11")
        self.defn.set_duration(11)
        self.assertEqual(self.defn.get_start_date(), utc_date("2020-10-11"))
        self.assertEqual(self.defn.get_end_date(), utc_date("2020-10-22"))
        self.assertEqual(self.defn.get_duration(), 11)

    def test_set_start_end(self):
        self.defn.set_start_date("2020-10-11")
        self.defn.set_end_date("2020-10-22")
        self.assertEqual(self.defn.get_start_date(), utc_date("2020-10-11"))
        self.assertEqual(self.defn.get_end_date(), utc_date("2020-10-22"))
        self.assertEqual(self.defn.get_duration(), 11)

    def test_add_exceptions(self):
        self.defn.clear_exceptions()
        self.defn.add_exception([self.rds["CZ"], self.rds["G-AAA"]], {})
        self.defn.add_exception([self.rds["W-EU"]], {"beta": 1e-10})
        self.assert_xml_equal(
            self.defn.exceptions_node,
            """
            <exceptions>
                <exception basins="710" continents="" countries="55" from="2020-03-25" hemispheres="" regions="" till="2021-03-10" />
                <exception basins="" continents="2" countries="" from="2020-03-25" hemispheres="" regions="" till="2021-03-10">
                    <variable name="beta" value="1e-10" />
                </exception>
            </exceptions>
            """,
        )

    def test_add_seed(self):
        self.defn.clear_seeds()
        self.defn.add_seed(self.rds["G-AAA"], {"Infectious": 100, "Recovered": 200})
        self.assert_xml_equal(
            self.defn.seeds_node,
            """
            <seeds>
                <seed city="710" compartment="Infectious" number="100" />
                <seed city="710" compartment="Recovered" number="200" />
            </seeds>
            """,
        )

    def test_set_seeds(self):
        self.defn.set_seeds(self.get_estimates(), self.rds)
        self.assert_xml_equal(
            self.defn.seeds_node,
            """
            <seeds>
                <seed city="1543" compartment="Infectious" number="2" />
                <seed city="1543" compartment="Exposed" number="5" />
                <seed city="1544" compartment="Infectious" number="3" />
                <seed city="655" compartment="Infectious" number="1" />
                <seed city="655" compartment="Exposed" number="4" />
            </seeds>
            """,
        )

    def test_set_initial_compartments(self):
        self.assertRaises(
            AssertionError,
            self.defn.set_initial_compartments,
            {"Exposed": 60, "Infectious": 39.9},
        )
        self.assertRaises(
            AssertionError,
            self.defn.set_initial_compartments,
            {"Exposed": 60, "Infectious": 40.1},
        )
        self.defn.set_initial_compartments({"Exposed": 60, "Infectious": 40})
        self.assert_xml_equal(
            self.defn.initial_compartments_node,
            """
            <initialCompartments>
                <initialCompartment compartment="Exposed" fraction="60.0" />
                <initialCompartment compartment="Infectious" fraction="40.0" />
            </initialCompartments>
            """,
        )

    def test_set_initial_compartments_from_estimates_rounds(self):
        estimates = self.get_estimates()
        fake_populations = pd.Series([300, 300, 300], index=estimates.index)
        rds = NonCallableMagicMock()
        rds.data.loc.__getitem__.return_value = fake_populations

        # these numbers will round to 99% total, so this test ensures
        # the function handles that possibility
        estimates["Susceptible"] = 130
        estimates["Infectious"] = 70
        estimates["Exposed"] = 100

        output = Mock()
        self.defn.set_initial_compartments = output
        self.defn.set_initial_compartments_from_estimates(estimates, rds)

        output.assert_called_once_with(
            {
                "Susceptible": pytest.approx(43.4),
                "Infectious": pytest.approx(23.3),
                "Exposed": pytest.approx(33.3),
            }
        )

    def test_set_initial_compartments_from_estimates_fills_susceptible(self):
        """
        It should assume all unaccounted population is Susceptible
        """
        estimates = self.get_estimates()
        fake_populations = pd.Series([500, 500, 500], index=estimates.index)
        rds = NonCallableMagicMock()
        rds.data.loc.__getitem__.return_value = fake_populations

        estimates["Infectious"] = 100
        estimates["Exposed"] = 100

        output = Mock()
        self.defn.set_initial_compartments = output
        self.defn.set_initial_compartments_from_estimates(estimates, rds)

        output.assert_called_once_with(
            {
                "Susceptible": pytest.approx(60),
                "Infectious": pytest.approx(20),
                "Exposed": pytest.approx(20),
            }
        )
