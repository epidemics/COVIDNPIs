import datetime
import logging
import re
import weakref
from collections import OrderedDict
from pathlib import Path

import dateutil
import numpy as np
import pandas as pd
import unidecode

from .utils import normalize_name

log = logging.getLogger(__name__)


class Region:
    def __init__(self, rds, code):
        self._rds = weakref.ref(rds)
        self._code = code
        r = rds.data.loc[code]
        names = [r.Name, r.OfficialName]
        if not pd.isnull(r.OtherNames):
            names.extend(r.OtherNames.split(RegionDataset.SEP))
        names = [n for n in names if not pd.isnull(n) and n]
        rds.data.at[code, "AllNames"] = list(set(names))

    @property
    def Code(self):
        return self._code

    @property
    def DisplayName(self):
        if self.Level == "subdivision":
            return f"{self.Name}, {self.CountryCode}"
        if self.Level == "gleam_basin" or self.Level == "city":
            return f"{self.Name}, {self.SubdivisionCode}"
        return self.Name

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __getitem__(self, name):
        return self._rds().data.at[self._code, name]

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._code} {self.Name} ({self.Level})>"

    def __setattr__(self, name, val):
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            raise AttributeError(
                f"Setting attribute {name} on {self.__class__.__name__} not allowed"
                " (use indexing)."
            )


class RegionDataset:
    """

    The Id is:
    # EARTH - root node
    # Continent?
    US - ISOa2 code, Level="country"
    US-CA - ISO 3166-2 state/province code, Level="subdivision"
    """

    # Separating names in name list and column name from date
    SEP = "|"

    LEVELS = pd.CategoricalDtype(
        pd.Index(
            [
                "world",
                "continent",
                "subregion",
                "country",
                "subdivision",
                "gleam_basin",
            ],
            dtype="U",
        ),
        ordered=True,
    )

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
        GleamID="int32",
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
    def load(cls, path):
        """
        Create a RegionDataset from a given CSV.
        """
        s = cls()
        data = pd.read_csv(
            path,
            dtype=cls.COLUMN_TYPES,
            index_col="Code",
            na_values=[""],
            keep_default_na=False,
        )
        s.data = s.data.append(data)
        s.data.sort_index()
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

    def find_all_by_name(self, s, levels=None):
        """
        Return all Regions with some matching names (filtering on levels).
        """
        if isinstance(levels, str):
            levels = [levels]
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
            r["OtherNames"] = self.SEP.join(names)
        # Write non-generated columns
        columns = self.COLUMN_TYPES.keys()
        self.data[columns].to_csv(path, index_label="Code")

    def _rebuild_index(self):
        self._name_index = {}
        self._code_index = {}
        self.data["AllNames"] = pd.Series(dtype=object)
        conflicts = []
        for ri in self.data.index:
            reg = Region(self, ri)
            for n in set(normalize_name(name) for name in reg.AllNames):
                self._name_index.setdefault(n, list()).append(reg)
            assert ri not in self._code_index
            self._code_index[ri] = reg
        for k in self._name_index:
            self._name_index[k] = list(set(self._name_index[k]))
            if len(self._name_index[k]) > 1:
                conflicts.append(k)
        if conflicts:
            log.info(
                f"Name index has {len(conflicts)} potential conflicts: {conflicts!r}"
            )
