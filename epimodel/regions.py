import enum
import logging
import weakref
from collections import OrderedDict

import pandas as pd

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

    def get_display_name(self):
        if self.Level == Level.subdivision:
            return f"{self.Name}, {self.CountryCode}"
        if self.Level == Level.gleam_basin:
            return f"{self.Name}, {self.SubdivisionCode}"
        return self.Name

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __getitem__(self, name):
        return self._rds().data.at[self._code, name]

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._code} {self.Name} ({self.Level})>"

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

    LEVELS = pd.CategoricalDtype(pd.Index(list(Level), dtype="O",), ordered=True,)

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
        Create a RegionDataset and its Regions from the given CSV.

        Optionally also loads other CSVs with additional regions (e.g. GLEAM regions)
        """
        s = cls()
        cols = dict(cls.COLUMN_TYPES, Level="U")
        for path in paths:
            log.debug("Loading regions from {path!r} ...")
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
        return s

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
            if isinstance(levels, Level):
                levels = [levels]
            assert all(isinstance(x, Level) for x in levels)
        rs = tuple(self._name_index.get(normalize_name(s), []))
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
            log.info(
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
