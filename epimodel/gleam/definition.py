import io
import copy
from datetime import datetime, date
import logging
import xml.etree.ElementTree as ET
from typing import Iterable, Union
from pathlib import Path

import pandas as pd

from ..regions import Level, RegionDataset, Region
from ..utils import utc_date

log = logging.getLogger(__name__)


class GleamDefinition:
    GLEAM_ID_SUFFIX = ".574"  # Magic? Or arbtrary?
    DEFAULT_XML_FILE = (
        Path(__file__).parent.parent.parent / "data/default_gleam_definition.xml"
    )

    def __init__(self, xml_file=None):
        """
        Load gleam `definition.xml` from a file (path or a file-like object).
        """
        ET.register_namespace("", "http://www.gleamviz.org/xmlns/gleamviz_v4_0")
        self.ns = {"gv": "http://www.gleamviz.org/xmlns/gleamviz_v4_0"}
        self.tree = ET.parse(xml_file or self.DEFAULT_XML_FILE)
        self.root = self.tree.getroot()

        if not self.get_timestamp_str():
            self.set_timestamp()

    def copy(self):
        return copy.deepcopy(self)

    def find_all(self, query):
        return self.root.findall(query, namespaces=self.ns)

    def find_one(self, query):
        x = self.root.findall(query, namespaces=self.ns)
        if not len(x) == 1:
            raise Exception(
                "Expected one XML object at query {!r}, found {!r}".format(query, x)
            )
        return x[0]

    def save(self, file):
        self.tree.write(file)  # , default_namespace=self.ns['gv'])
        log.debug(f"Written Gleam definition to {file!r}")

    def to_xml_string(self):
        b = io.BytesIO()
        self.save(b)
        return b.getvalue().decode("ascii")

    ### Testing methids

    def assert_equal(self, other):
        assert isinstance(other, self.__class__)
        self._etree_assert_eq(self.root, other.root)

    def _etree_assert_eq(self, e1: ET.Element, e2: ET.Element, path="/"):
        """
        Recursive equality assertion, based on:
        https://stackoverflow.com/questions/7905380/testing-equivalence-of-xml-etree-elementtree
        """
        try:
            tag = self._strip_tag(e1)
            assert tag == self._strip_tag(e2)
            assert (e1.text or "").strip() == (e2.text or "").strip()
            assert (e1.tail or "").strip() == (e2.tail or "").strip()
            assert e1.attrib == e2.attrib
            assert len(e1) == len(e2)
        except AssertionError:
            # useful for debugging mismatches
            # import pdb; pdb.set_trace()
            raise AssertionError(
                f"{e1!r} != {e2!r} at path {path}\n\n%s\n\n%s"
                % (
                    ET.tostring(e1, encoding="utf-8", method="xml"),
                    ET.tostring(e2, encoding="utf-8", method="xml"),
                )
            )
        for c1, c2 in zip(e1, e2):
            self._etree_assert_eq(c1, c2, path=f"{path}{tag}/")

    def _strip_tag(self, element):
        return element.tag.replace("{%s}" % self.ns["gv"], "")

    ### Main nodes

    @property
    def definition_node(self) -> ET.Element:
        return self.find_one("./gv:definition")

    @property
    def parameter_node(self) -> ET.Element:
        return self.find_one("./gv:definition/gv:parameters")

    def variable_node(self, name: str) -> ET.Element:
        return self.find_one(
            f'./gv:definition/gv:compartmentalModel/gv:variables/gv:variable[@name="{name}"]'
        )

    @property
    def initial_compartments_node(self):
        return self.find_one("./gv:definition/gv:initialCompartments")

    @property
    def exceptions_node(self) -> ET.Element:
        return self.find_one("./gv:definition/gv:exceptions")

    @property
    def seeds_node(self) -> ET.Element:
        return self.find_one("./gv:definition/gv:seeds")

    @property
    def timestamp_node(self) -> ET.Element:
        return self.find_one("./gv:metadata/gv:creationDate")

    ### Exceptions

    def clear_exceptions(self):
        """Remove all exceptions from the XML."""
        enode = self.exceptions_node
        enode.clear()
        enode.tail = "\n  "

    def format_exceptions(self):
        self._format_list_node(self.exceptions_node)

    def add_exception(
        self,
        regions: Iterable[Region],
        variables: dict,
        start=None,
        end=None,
        format=False,
    ):
        """
        Add a single exception restricted to `regions` and given dates.

        `variables` is a dictionary `{variable_name: value}`.
        Default `start` is the simulation start, default `end` is the simulation end.
        NB: This is not changed if you change the simulation start/end later!
        """
        attrs = dict(basins="", continents="", countries="", hemispheres="", regions="")
        if start is None:
            start = self.get_start_date()
        if end is None:
            end = self.get_end_date()
        attrs["from"] = utc_date(start).date().isoformat()
        attrs["till"] = utc_date(end).date().isoformat()
        for r in regions:
            if pd.isnull(r.GleamID) or r.GleamID == "":
                raise ValueError(f"{r!r} does not correspond to a Gleam region.")
            tn = {
                Level.gleam_basin: "basins",
                Level.continent: "continents",
                Level.country: "countries",
                Level.subregion: "regions",
            }[r.Level]
            attrs[tn] = (attrs[tn] + f" {r.GleamID}").strip()
        ex = ET.SubElement(self.exceptions_node, "exception", attrs)
        for vn, vv in variables.items():
            ET.SubElement(ex, "variable", dict(name=str(vn), value=str(vv)))

        if format:
            self.format_exceptions()

    ### Seed compartments

    def clear_seeds(self):
        seeds_node = self.seeds_node
        seeds_node.clear()
        seeds_node.tail = "\n    "

    def format_seeds(self):
        self._format_list_node(self.seeds_node)

    def set_seeds(self, estimates: pd.DataFrame, rds=None):
        """
        Estimates must be a DataFrame with a `Region` column containing
        Region objects. All other columns are compartment names
        containing the total number of people in that compartment for
        that region.

        Alternately, you can leave out the the `Region` column and key
        the DataFrame by region code instead, but you must also pass an
        `rds` argument.

        All regions must be Gleam basins.
        """
        estimates = estimates.copy()

        if "Region" not in estimates:
            estimates["Region"] = [rds[code] for code in estimates.index]

        # sort by GleamID for deterministic order
        estimates["GleamID"] = estimates["Region"].apply(lambda reg: reg.GleamID)
        estimates.sort_values(by="GleamID", inplace=True)
        seeds = pd.DataFrame(
            {
                "region": estimates["Region"],
                "compartments": estimates.drop(columns=["Region", "GleamID"]).apply(
                    lambda row: {
                        compartment: population
                        for compartment, population in row.iteritems()
                        if pd.notnull(population)
                    },
                    axis=1,
                ),
            }
        )

        self.clear_seeds()
        for _, seed in seeds.iterrows():
            self.add_seed(*seed)
        self.format_seeds()

    def add_seed(self, region: Region, compartments: dict, format=False):
        """
        Add seed populations for a gleam_basin.
        """
        assert region.Level == Level.gleam_basin
        assert pd.notnull(region.GleamID)

        for compartment, size in compartments.items():
            ET.SubElement(
                self.seeds_node,
                "seed",
                {
                    "number": str(int(size)),
                    "compartment": compartment,
                    "city": str(region.GleamID),
                },
            )

        if format:
            self.format_seeds()

    ### Initial Compartments

    def clear_initial_compartments(self):
        cnode = self.initial_compartments_node
        cnode.clear()
        cnode.tail = "\n    "

    def format_initial_compartments(self):
        self._format_list_node(self.initial_compartments_node)

    def set_initial_compartments_from_estimates(self, estimates):
        """
        Determines compartment proportions from same DataFrame format as
        used in set_seeds and uses this to set_initial_compartments.
        """
        if "Region" in estimates:
            estimates = estimates.drop(columns=["Region"])
        compartments = estimates.sum()
        total_compartments = compartments.sum()
        compartments = (compartments / compartments.sum() * 100).dropna().apply(round)

        # ensure everything adds up to 100
        while compartments.sum() < 100:
            compartments[compartments.idxmax()] += 1
        while compartments.sum() > 100:
            compartments[compartments.idxmax()] -= 1

        self.set_initial_compartments(compartments.to_dict())

    def set_initial_compartments(self, compartments):
        assert sum(int(v) for v in compartments.values()) == 100

        self.clear_initial_compartments()
        ic_node = self.initial_compartments_node

        for compartment, percentage in compartments.items():
            ET.SubElement(
                ic_node,
                "initialCompartment",
                {"compartment": compartment, "fraction": str(int(percentage))},
            )

        self.format_initial_compartments()

    ### Formatting

    def _format_list_node(self, node: ET.Element, depth=2):
        short_tail = f"\n{'  ' * depth}"
        long_tail = f"{short_tail}  "
        node.text = long_tail
        for child in node:
            child.tail = long_tail
        last_child = node.find("./*[last()]")
        if last_child is not None:
            last_child.tail = short_tail

    ### General attributes

    def get_name(self):
        return self.definition_node.attrib["name"]

    def set_name(self, val: str):
        assert isinstance(val, str)
        self.definition_node.attrib["name"] = val

    def set_default_name(self, comment=None):
        self.set_name(
            f"{self.timestamp_node.text} {comment} "
            f"Seas={self.get_seasonality()} TrOcc={self.get_traffic_occupancy()} "
            f"beta={self.get_variable('beta')}"
        )

    def get_id(self) -> int:
        return int(self.get_id_str().replace(self.GLEAM_ID_SUFFIX, ""))

    def get_id_str(self) -> str:
        return self.definition_node.get("id")

    def set_id(self, val: int):
        assert isinstance(val, int)
        return self.definition_node.set("id", f"{val}{self.GLEAM_ID_SUFFIX}")

    def get_timestamp(self) -> datetime:
        return pd.Timestamp(self.get_timestamp_str(), tz="UTC")

    def get_timestamp_str(self) -> str:
        return self.timestamp_node.text

    def set_timestamp(self, timestamp=None):
        if timestamp is None:
            timestamp = pd.Timestamp.utcnow()
        else:
            timestamp = pd.Timestamp(timestamp, tz="UTC")
        self.timestamp_node.text = timestamp.strftime("%Y-%m-%dT%T")

    def get_start_date(self) -> datetime:
        return utc_date(self.get_start_date_str())

    def get_start_date_str(self) -> str:
        return self.parameter_node.get("startDate")

    def set_start_date(self, date: Union[str, date, datetime]):
        self.parameter_node.set("startDate", utc_date(date).date().isoformat())

    def get_duration(self) -> int:
        """Return the number of days to simulate."""
        return int(self.parameter_node.get("duration"))

    def set_duration(self, duration: Union[int, float]):
        """Set duration in days."""
        assert isinstance(duration, (int, float))
        self.parameter_node.set("duration", str(round(duration)))

    def get_run_count(self) -> int:
        """Set number of simulations to run."""
        return int(self.parameter_node.get("runCount"))

    def set_run_count(self, run_count: Union[int, float]):
        """Set number of simulations to run."""
        assert isinstance(run_count, (int, float))
        self.parameter_node.set("runCount", str(int(run_count)))

    def get_end_date(self) -> datetime:
        return self.get_start_date() + pd.DateOffset(self.get_duration())

    def set_end_date(self, date: Union[str, date, datetime]):
        """ Note: this must be set *after* start_date
            or it may change unexpectedly """
        self.set_duration(
            (utc_date(date) - self.get_start_date()) / pd.Timedelta(days=1)
        )

    ### Global Parameters

    def get_seasonality(self) -> float:
        return float(self.parameter_node.get("seasonalityAlphaMin"))

    def set_seasonality(self, val: float):
        assert val <= 2.0
        self.parameter_node.set("seasonalityAlphaMin", f"{val:.2f}")

    def get_airline_traffic(self) -> float:
        """ TrafficOccupancy scaled as 0-1 """
        return self.get_traffic_occupancy / 100.0

    def set_airline_traffic(self, val: float):
        """ TrafficOccupancy scaled as 0-1 """
        self.set_traffic_occupancy(round(val * 100))

    def get_traffic_occupancy(self) -> int:
        "Note: this an integer in percent"
        return int(self.parameter_node.get("occupancyRate"))

    def set_traffic_occupancy(self, val: int):
        "Note: this must be an integer in percent"
        assert isinstance(val, int)
        assert 0 <= val and val <= 100
        self.parameter_node.set("occupancyRate", str(int(val)))

    def get_commuting_rate(self, val: Union[float, int]):
        """ "time spent at commuting destination" in Gleam settings """
        self.parameter_node.set("occupancyRate", f"{val:.1f}")

    def set_commuting_rate(self, val: Union[float, int]):
        """ "time spent at commuting destination" in Gleam settings """
        self.parameter_node.set("occupancyRate", f"{val:.1f}")

    def get_compartment_variable(self, name: str) -> str:
        return self.variable_node(name).get("value")

    def set_compartment_variable(self, name: str, val: float):
        assert isinstance(name, str)
        assert isinstance(val, float)
        self.variable_node(name).set("value", f"{val:.2f}")

    ### Backwards Compatibility

    def get_variable(self, name: str) -> str:
        return self.get_compartment_variable(name)

    def set_variable(self, name: str, val: float):
        return self.set_compartment_variable(name, val)
