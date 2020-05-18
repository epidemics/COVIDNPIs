import pytest
from . import PandasTestCase

import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET
import epimodel
from epimodel.utils import utc_date
from epimodel.gleam.definition import GleamDefinition


def test_batch_new_open(tmp_path):
    b = epimodel.gleam.Batch.new(dir=tmp_path)
    path = b.path
    b.close()
    b2 = epimodel.gleam.Batch.open(path)
    b2.close()


def test_set_seeds_add_export_sims(regions_gleam, datadir, tmp_path):
    d = GleamDefinition(datadir / "test_definition.xml")
    b = epimodel.gleam.Batch.new(dir=tmp_path)

    i_df = pd.DataFrame(
        {"Infectious": [1, 2, 3], "Exposed": [4, 5, np.nan]},
        index=["G-MLA", "G-KGL", "G-KME"],
    )
    d.set_seeds(i_df, regions_gleam)

    b.set_simulations([(d, "Name1", "MEDIUM", "WEAK_WEAK")])
    b.set_initial_compartments(i_df)
    path = b.path
    b.close()

    b2 = epimodel.gleam.Batch.open(path)
    assert len(b2.hdf["simulations"]) == 1
    assert len(b2.hdf["initial_compartments"]) == 3
    b2.export_definitions_to_gleam(tmp_path)
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
            {"Exposed": 60, "Infectious": 39},
        )
        self.assertRaises(
            AssertionError,
            self.defn.set_initial_compartments,
            {"Exposed": 60, "Infectious": 41},
        )
        self.defn.set_initial_compartments({"Exposed": 60, "Infectious": 40})
        self.assert_xml_equal(
            self.defn.initial_compartments_node,
            """
            <initialCompartments>
                <initialCompartment compartment="Exposed" fraction="60" />
                <initialCompartment compartment="Infectious" fraction="40" />
            </initialCompartments>
            """,
        )

    def test_set_initial_compartments_from_estimates(self):
        estimates = self.get_estimates()
        estimates["Infectious"] = 70
        estimates["Exposed"] = 100
        estimates["Recovered"] = 130
        self.defn.set_initial_compartments_from_estimates(estimates)
        self.assert_xml_equal(
            self.defn.initial_compartments_node,
            """
            <initialCompartments>
                <initialCompartment compartment="Infectious" fraction="23" />
                <initialCompartment compartment="Exposed" fraction="33" />
                <initialCompartment compartment="Recovered" fraction="44" />
            </initialCompartments>
            """,
        )
