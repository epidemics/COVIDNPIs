import copy
from datetime import datetime, date
import logging
import xml.etree.ElementTree as ET
from typing import Iterable, Union, Callable

import pandas as pd

from ..regions import Level, RegionDataset, Region
from ..utils import utc_date

log = logging.getLogger(__name__)


class GleamDefinition:
    DEFAULT_XML_FILE = "data/default_gleam_definition.xml"
    GLOBAL_PARAMETERS = (
        "number of runs",
        "airline traffic",
        "seasonality",
        "commuting time",
    )
    COMPARTMENT_VARIABLES = (
        "beta",
        "epsilon",
        "mu",
        "imu",
    )

    def __init__(self, file=DEFAULT_XML_FILE):
        """
        Load gleam `definition.xml` from a file (path or a file-like object).
        """
        ET.register_namespace("", "http://www.gleamviz.org/xmlns/gleamviz_v4_0")
        self.ns = {"gv": "http://www.gleamviz.org/xmlns/gleamviz_v4_0"}
        self.tree = ET.parse(file)
        self.root = self.tree.getroot()

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
    def exceptions_node(self) -> ET.Element:
        return self.find_one("./gv:definition/gv:exceptions")

    @property
    def seeds_node(self) -> ET.Element:
        return self.find_one("./gv:definition/gv:seeds")

    def timestamp_node(self) -> ET.Element:
        return self.find_one("./gv:definition/gv:metadata/gv:creationDate")

    ### Exceptions

    def clear_exceptions(self):
        """Remove all exceptions from the XML."""
        self.exceptions_node.clear()

    def add_exception(
        self, regions: Iterable[Region], variables: dict, start=None, end=None
    ):
        """
        Add a single exception restricted to `regions` and given dates.

        `variables` is a dictionary `{variable_name: value}`.
        Default `start` is the simulation start, default `end` is the simulation end.
        NB: This is not changed if you change the simulation start/end later!
        """
        enode = self.exceptions_node
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
        ex = ET.SubElement(enode, "exception", attrs)
        for vn, vv in variables.items():
            ET.SubElement(ex, "variable", dict(name=str(vn), value=str(vv)))
        ex.tail = "\n"

    ### Seed compartments

    def clear_seeds(self):
        self.seeds_node.clear()

    def add_seeds(self, rds: RegionDataset, compartments: pd.DataFrame, top=None):
        """
        Add seed populations from `sizes` to `compartment`.

        Only considers Level.gleam_basin regions from `sizes`.
        `sizes` must be indexed by `Code`. `rds` must have the gleam
        regions loaded.
        """
        assert isinstance(rds, RegionDataset)
        assert isinstance(compartments, pd.DataFrame)
        seeds_node = self.seeds_node
        for c in compartments.columns:
            sizes = compartments[c].sort_values(ascending=False)
            sl = slice(None) if top is None else slice(0, top)
            for rc, s in list(sizes.items())[sl]:
                r = rds[rc]
                if r.Level != Level.gleam_basin:
                    continue
                assert not pd.isnull(r.GleamID)

                seed = ET.SubElement(
                    seeds_node,
                    "seed",
                    {"number": str(int(s)), "compartment": c, "city": str(r.GleamID)},
                )
                seed.tail = "\n"

    ### General attributes

    def get_name(self):
        return self.definition_node.attrib["name"]

    def set_name(self, val):
        assert isinstance(val, str)
        self.definition_node.attrib["name"] = val

    def set_default_name(self, comment=None):
        self.set_name(
            f"{self.timestamp_node.text} {comment} "
            f"Seas={self.get_seasonality()} TrOcc={self.get_traffic_occupancy()} "
            f"beta={self.get_variable('beta')}"
        )

    def get_id(self) -> str:
        return self.definition_node.get("id")

    def set_id(self, val: str):
        assert isinstance(val, str)
        return self.definition_node.set("id", val)

    def get_timestamp(self) -> datetime:
        return pd.Timestamp(self.timestamp_node.text)

    def set_timestamp(self, timestamp=None):
        timestamp = pd.Timestamp(timestamp or "now")
        self.timestamp_node.text = timestamp.strftime("%Y-%m-%dT%T")

    def get_start_date(self) -> datetime:
        return utc_date(self.parameter_node.get("startDate"))

    def set_start_date(self, date: Union[str, date, datetime]):
        self.parameter_node.set("startDate", utc_date(date).date().isoformat())

    def get_duration(self) -> int:
        """Return the number of days to simulate."""
        return int(self.parameter_node.get("duration"))

    def set_duration(self, duration: int):
        """Set duration in days."""
        assert isinstance(duration, int)
        self.parameter_node.set("duration", str(duration))

    def get_number_of_runs(self) -> int:
        """Set number of simulations to run."""
        return int(self.parameter_node.get("runCount"))

    def set_number_of_runs(self, run_count: int):
        """Set number of simulations to run."""
        assert isinstance(run_count, int)
        self.parameter_node.set("runCount", str(run_count))

    def get_end_date(self) -> datetime:
        return self.get_start_date() + pd.DateOffset(self.get_duration())

    def set_end_date(self, date: Union[str, date, datetime]):
        """ Note: this must be set *after* start_date
            or it may change unexpectedly """
        self.set_duration(utc_date(date) - self.get_start_date())

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

    def fa(self, query):
        return self.find_all(query)

    def f1(self, query):
        return self.find_one(query)

    def get_variable(self, name: str) -> str:
        return self.get_compartment_variable(name)

    def set_variable(self, name: str, val: float):
        return self.set_compartment_variable(name, val)
