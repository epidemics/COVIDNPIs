import copy
from datetime import datetime, date
import logging
import xml.etree.ElementTree as ET
from typing import Iterable, Union

import pandas as pd

from ..regions import Level, RegionDataset, Region
from ..utils import utc_date

log = logging.getLogger(__name__)


class GleamDefinition:
    def __init__(self, file):
        """
        Load gleam `definition.xml` from a file (path or a file-like object).
        """
        ET.register_namespace("", "http://www.gleamviz.org/xmlns/gleamviz_v4_0")
        self.ns = {"gv": "http://www.gleamviz.org/xmlns/gleamviz_v4_0"}
        self.tree = ET.parse(file)
        self.root = self.tree.getroot()

        self.updated = datetime.datetime.now()

    def copy(self):
        return copy.deepcopy(self)

    def fa(self, query):
        return self.root.findall(query, namespaces=self.ns)

    def f1(self, query):
        x = self.root.findall(query, namespaces=self.ns)
        if not len(x) == 1:
            raise Exception(
                "Expected one XML object at query {!r}, found {!r}".format(query, x)
            )
        return x[0]

    def save(self, file):
        self.tree.write(file)  # , default_namespace=self.ns['gv'])
        log.debug(f"Written Gleam definition to {file!r}")

    ### Exceptions

    def clear_exceptions(self):
        """Remove all exceptions from the XML."""
        self.f1("./gv:definition/gv:exceptions").clear()

    def add_exception(
        self, regions: Iterable[Region], variables: dict, start=None, end=None
    ):
        """
        Add a single exception restricted to `regions` and given dates.

        `variables` is a dictionary `{variable_name: value}`.
        Default `start` is the simulation start, default `end` is the simulation end.
        NB: This is not changed if you change the simulation start/end later!
        """
        enode = self.f1("./gv:definition/gv:exceptions")
        attrs = dict(basins="", continents="", countries="", hemispheres="", regions="")
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
        self.f1("./gv:definition/gv:seeds").clear()

    def add_seeds(self, rds: RegionDataset, compartments: pd.DataFrame, top=None):
        """
        Add seed populations from `sizes` to `compartment`.

        Only considers Level.gleam_basin regions from `sizes`.
        `sizes` must be indexed by `Code`. `rds` must have the gleam
        regions loaded.
        """
        assert isinstance(rds, RegionDataset)
        assert isinstance(compartments, pd.DataFrame)
        sroot = self.f1("./gv:definition/gv:seeds")
        for c in compartments.columns:
            sizes = compartments[c].sort_values(ascending=False)
            sl = slice(None) if top is None else slice(0, top)
            for rc, s in list(sizes.items())[sl]:
                r = rds[rc]
                if r.Level != Level.gleam_basin:
                    continue
                assert not pd.isnull(r.GleamID)

                seed = ET.SubElement(
                    sroot,
                    "seed",
                    {"number": str(int(s)), "compartment": c, "city": str(r.GleamID)},
                )
                seed.tail = "\n"

    ### General attributes

    def get_name(self):
        return self.f1("./gv:definition").attrib["name"]

    def set_name(self, val):
        assert isinstance(val, str)
        self.f1("./gv:definition").attrib["name"] = val

    def set_default_name(self, comment=None):
        self.set_name(
            f"{self.updated.strftime('%Y-%m-%d %H:%M:%S')} {comment} "
            f"Seas={self.get_seasonality()} TrOcc={self.get_traffic_occupancy()} "
            f"beta={self.get_variable('beta')}"
        )

    def get_id(self) -> str:
        return self.f1("gv:definition").get("id")

    def set_id(self, val: str):
        assert isinstance(val, str)
        return self.f1("gv:definition").set("id", val)

    def get_start_date(self) -> datetime:
        return utc_date(self.f1("./gv:definition/gv:parameters").get("startDate"))

    def set_start_date(self, date: Union[str, date, datetime]):
        self.f1("./gv:definition/gv:parameters").set(
            "startDate", utc_date(date).date().isoformat()
        )

    def get_duration(self) -> int:
        """Return the number of days to simulate."""
        return int(self.f1("./gv:definition/gv:parameters").get("duration"))

    def set_duration(self, duration: int):
        """Set duration in days."""
        assert isinstance(duration, int)
        self.f1("./gv:definition/gv:parameters").set("duration", str(duration))

    def get_end_date(self) -> datetime:
        return self.get_start_date() + pd.DateOffset(self.get_duration())

    ### Parameters

    def get_seasonality(self) -> float:
        return float(
            self.f1("./gv:definition/gv:parameters").get("seasonalityAlphaMin")
        )

    def set_seasonality(self, val: float):
        assert val <= 2.0
        self.f1("./gv:definition/gv:parameters").set(
            "seasonalityAlphaMin", f"{val:.2f}"
        )

    def get_variable(self, name: str) -> str:
        return self.f1(
            f'./gv:definition/gv:compartmentalModel/gv:variables/gv:variable[@name="{name}"]'
        ).get("value")

    def set_variable(self, name: str, val: float):
        assert isinstance(name, str)
        assert isinstance(val, float)
        self.f1(
            f'./gv:definition/gv:compartmentalModel/gv:variables/gv:variable[@name="{name}"]'
        ).set("value", f"{val:.2f}")

    def get_traffic_occupancy(self) -> int:
        "Note: this an integer in percent"
        return int(self.f1("./gv:definition/gv:parameters").get("occupancyRate"))

    def set_traffic_occupancy(self, val: int):
        "Note: this must be an integer in percent"
        assert isinstance(val, int)
        assert 0 <= val and val <= 100
        self.f1("./gv:definition/gv:parameters").set("occupancyRate", str(int(val)))
