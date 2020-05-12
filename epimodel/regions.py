import re
import enum
import logging
import weakref
from collections import OrderedDict

import pandas as pd
import yaml

from .utils import normalize_name

log = logging.getLogger(__name__)


class Level(enum.Enum):
    """
    Region levels in the dataset. The numbers are NOT canonical, only the names are.

    Ordered by "size" - world is the largest.
    """

    gleam_basin = 1
    subdivision = 2
    country = 3
    subregion = 4
    continent = 5
    world = 6

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Region:
    def __init__(self, rds, code):
        self._rds = weakref.ref(rds)
        self._code = code
        self._parent = None
        self._children = set()
        r = rds.data.loc[code]
        names = [r.Name, r.OfficialName]
        if not pd.isnull(r.OtherNames):
            names.extend(r.OtherNames.split(RegionDataset.SEP))
        names = [n for n in names if not pd.isnull(n) and n]
        rds.data.at[code, "AllNames"] = list(set(names))
        rds.data.at[code, "Region"] = self
        rds.data.at[code, "DisplayName"] = self.get_display_name()
        # Cache for safe debugging
        self._repr_str = self._gen_repr_str()

    def _gen_repr_str(self):
        return (
            f"<{self.__class__.__name__} {self._code} {self.Name} ({self.Level.name})>"
        )

    def get_display_name(self):
        if self.Level == Level.subdivision:
            return f"{self.Name}, {self.CountryCode}"
        if self.Level == Level.gleam_basin:
            if pd.notnull(self.SubdivisionCode) and self.SubdivisionCode != "":
                return f"{self.Name}, {self.SubdivisionCode}"
            else:
                return f"{self.Name}, {self.CountryCode}"
        return self.Name

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattr__(name)
        else:
            return self.__getitem__(name)

    def __getitem__(self, name):
        return self._rds().data.at[self._code, name]

    def __repr__(self):
        return self._repr_str

    def __setattr__(self, name, val):
        """Forbid direct writes to anything but _variables."""
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            raise AttributeError(
                f"Setting attribute {name} on {self!r} not allowed (use rds.data directly)."
            )

    @property
    def Code(self):
        return self._code

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    @property
    def agg_children(self):
        return self["agg_children"]

    def _region_prop(self, name):
        """Return the Region corresponding to code in `self[name]` (None if that is None)."""
        rds = self._rds()
        assert rds is not None
        cid = rds.data.at[self._code, name]
        if pd.isnull(cid) or cid == "":
            return None
        return rds[cid]

    @property
    def continent(self):
        return self._region_prop("ContinentCode")

    @property
    def subregion(self):
        return self._region_prop("SubregionCode")

    @property
    def country(self):
        return self._region_prop("CountryCode")

    @property
    def subdivision(self):
        return self._region_prop("SubdivisionCode")


class RegionDataset:
    """
    A set of regions and their attributes, with a hierarchy. A common index for most data files.

    The Id is:
    W     - The world, root node, Level="world"
    W-AS  - Prefixed ISO continent code, Level="continent"
    (TBD) - Subregion code, Level="subregion"
    US    - ISOa2 code, Level="country"
    US-CA - ISO 3166-2 state/province code, Level="subdivision"
    G-AAA - Prefixed IANA code, used for GLEAM basins, Level="gleam_basin"
    """

    # Separating names in name list and column name from date
    SEP = "|"

    LEVELS = pd.CategoricalDtype(pd.Index(list(Level), dtype="O"), ordered=True)

    COLUMN_TYPES = OrderedDict(
        #        Parent="string",
        # ASCII name (unidecoded)
        Name="U",
        # Official name (any charactersscript)
        OfficialName="U",
        # OtherNames, incl orig. name unicode if different
        # encoded as '|'-separated list
        OtherNames="U",
        # Administrative level
        Level=LEVELS,
        # Countries and above
        M49Code="U",
        # Location in hierarchy
        ContinentCode="U",
        SubregionCode="U",
        CountryCode="U",
        CountryCodeISOa3="U",
        SubdivisionCode="U",
        # Other data
        Lat="f4",
        Lon="f4",
        Population="f4",
        # Stored as string to allow undefined values
        GleamID="U",
    )

    def __init__(self):
        """
        Creates an empty region set. Use `RegionDataset.load` to create from CSV.
        """
        # Main DataFrame (empty)
        self.data = pd.DataFrame(
            index=pd.Index([], name="Code", dtype=pd.StringDtype())
        )
        for name, dtype in self.COLUMN_TYPES.items():
            self.data[name] = pd.Series(dtype=dtype, name=name)
        # name: [Region, Region, ..]
        self._name_index = {}
        # code: [Region, Region, ...]
        self._code_index = {}

    @classmethod
    def load(cls, *paths):
        """
        Create a RegionDataset and its Regions from the given CSV or YAML.

        Optionally also loads other CSVs with additional regions (e.g. GLEAM regions)
        """
        s = cls()
        cols = dict(cls.COLUMN_TYPES, Level="U")
        yaml_paths = []
        for path in paths:
            if re.match(r"\.ya?ml$", re.IGNORECASE):
                yaml_paths.append(path)
                continue
            log.debug(f"Loading regions from {path!r} ...")
            data = pd.read_csv(
                path,
                dtype=cols,
                index_col="Code",
                na_values=[""],
                keep_default_na=False,
            )
            # Convert Level to enum
            data["Level"] = data["Level"].map(lambda name: Level[name])
            s.data = s.data.append(data, verify_integrity=True)
        s._rebuild_index()
        for path in yaml_paths:
            s.add_aggregate_regions_yaml(path)
        return s

    def add_aggregate_regions_yaml(self, yaml_file):
        """
        Adds aggregate regions from yaml file. Uses the same input
        format as add_cusom_regions().

        Example input file:
        ---
        PK-GB:
          Name: 'Gilgit-Baltistan'
          AggregateFrom:
            G-CJL: 0.25
            G-GIL: 1
            G-KDU: 1
        PK-ICT:
          Name: 'Islamabad Capital Territory'
          AggregateFrom:
            G-ISB: 0.21
        """
        with open(yaml_file, "r") as fp:
            self.add_aggregate_regions(yaml.load(fp))

    def add_aggregate_regions(self, aggregate_regions=dict):
        """
        Adds custom regions composed of existing regions from a config
        dict. Dict keys are new region codes and values are other dicts
        whose keys correspond to regions.csv fields. Any field not
        specified will be aggregated from the children if possible or
        left blank otherwise. The default Level value is "subregion".

        The "AggregateFrom" key specifies other region codes and can be
        a list or a dict. If a list, all specified regions are assumed
        to be wholly contained in the custom region. If a dict, the
        values define what portion of each is contained in the custom
        region. These proportions are then used to weight aggregated
        info such as Gleam traces.

        See add_aggregate_regions_yaml() docstring for example input.
        """
        region_fields = (
            "M49Code",
            "ContinentCode",
            "SubregionCode",
            "CountryCode",
            "CountryCodeISOa3",
            "SubdivisionCode",
        )

        rows = []
        for code, data in aggregate_regions.items():
            agg_codes = data.get("AggregateFrom", [])
            if isinstance(agg_codes, dict):
                agg_weights = list(agg_codes.values())
                agg_codes = list(agg_codes.keys())
            else:
                agg_weights = [1 for _ in agg_codes]
            agg_children = [
                (self[code], weight) for code, weight in zip(agg_codes, agg_weights)
            ]

            row = {k: v for k, v in data.items() if k in self.COLUMN_TYPES}
            row["Code"] = code
            row["Level"] = row.get("Level", Level.subregion)

            # set superregion fields if not otherwise set
            # and value is same for all included regions
            for region_field in region_fields:
                if region_field not in row:
                    values = self.data.loc[agg_codes, region_field].unique()
                    if len(values) == 1 and pd.notnull(values[0]):
                        row[region_field] = values[0]

            # average lat/lng
            # this algo breaks down for regions that span 180º longitude
            row["Lat"] = row.get("Lat") or sum(
                child.Lat * weight for child, weight in agg_children
            ) / sum(agg_weights)
            row["Lon"] = row.get("Lon") or sum(
                child.Lon * weight for child, weight in agg_children
            ) / sum(agg_weights)

            # sum population
            row["Population"] = row.get("Population") or round(
                sum(child.Population * weight for child, weight in agg_children)
            )

            # add extra data
            row["agg_children"] = agg_children

            rows.append(row)

        data = pd.DataFrame(rows).set_index("Code")
        if "agg_children" not in self.data:
            self.data["agg_children"] = None
        self.data = self.data.append(data, verify_integrity=True)
        self._rebuild_index()

    @property
    def regions(self):
        """Iterator over all regions."""
        return self._code_index.values()

    def __getitem__(self, code):
        """
        Returns the Region corresponding to code, or raise KeyError.
        """
        return self._code_index[code.upper()]

    def __contains__(self, code):
        """
        Returns the Region corresponding to code, or raise KeyError.
        """
        if not isinstance(code, str):
            return False
        return code.upper() in self._code_index

    def get(self, code, default=None):
        """
        Returns the Region corresponding to code, or `default`.
        """
        try:
            return self[code]
        except KeyError:
            return default

    def find_all_by_name(self, s, levels=None):
        """
        Return all Regions with some matching names (filtering on levels).
        """
        if levels is not None:
            if isinstance(levels, (Level, str)):
                levels = [levels]
            for i in range(len(levels)):
                if isinstance(levels[i], str):
                    levels[i] = Level[levels[i]]
            assert all(isinstance(x, Level) for x in levels)
        rs = tuple(self._name_index.get(normalize_name(s), ()))
        if levels is not None:
            rs = tuple(r for r in rs if r.Level in levels)
        return rs

    def find_one_by_name(self, s, levels=None):
        """
        Find one region matching name (filter on levels).

        Raises KeyError if no or multiple regions found.
        """
        rs = self.find_all_by_name(s, levels=levels)
        if len(rs) == 1:
            return rs[0]
        lcmt = "" if levels is None else f" [levels={levels!r}]"
        if len(rs) < 1:
            raise KeyError(f"Found no regions matching {s!r}{lcmt}")
        raise KeyError(f"Found multiple regions matching {s!r}{lcmt}: {rs!r}")

    def write_csv(self, path):
        # Reconstruct the OtherNames column
        for r in self.regions:
            # don't include aggregate regions
            if r.agg_children:
                continue
            names = set(r.AllNames)
            if r.Name in names:
                names.remove(r.Name)
            if r.OfficialName in names:
                names.remove(r.OfficialName)
            self.data.loc[r.Code, "OtherNames"] = self.SEP.join(names)
        # Write only non-generated columns
        df = self.data[self.COLUMN_TYPES.keys()]
        # Convert Level to names
        df.Level = df.Level.map(lambda l: l.name)
        # Write
        df.to_csv(path, index_label="Code")

    def _rebuild_index(self):
        """Rebuilds the indexes and ALL Region objects!"""
        self._name_index = {}
        self._code_index = {}
        self.data = self.data.sort_index()
        self.data["AllNames"] = pd.Series(dtype=object)
        self.data["Region"] = pd.Series(dtype=object)
        self.data["DisplayName"] = pd.Series(dtype=object)
        conflicts = []

        # Create Regions
        for ri in self.data.index:
            reg = Region(self, ri)
            for n in set(normalize_name(name) for name in reg.AllNames):
                self._name_index.setdefault(n, list()).append(reg)
            assert ri not in self._code_index
            self._code_index[ri] = reg

        # Unify names in index and find conflicts
        for k in self._name_index:
            self._name_index[k] = list(set(self._name_index[k]))
            if len(self._name_index[k]) > 1:
                conflicts.append(k)
        if conflicts:
            log.debug(
                f"Name index has {len(conflicts)} potential conflicts: {conflicts!r}"
            )

        # Add parent/children relations
        for r in self.regions:
            parent = None
            if parent is None and r.Level <= Level.gleam_basin:
                parent = r.subdivision
            if parent is None and r.Level <= Level.subdivision:
                parent = r.country
            if parent is None and r.Level <= Level.country:
                parent = r.subregion
            if parent is None and r.Level <= Level.subregion:
                parent = r.continent
            if parent is None and r.Level < Level.world:
                parent = self.get("W", None)
            r._parent = parent
            if parent is not None:
                parent._children.add(r)
