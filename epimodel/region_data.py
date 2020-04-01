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

from .utils import flatten_multiindex, normalize_name, unflatten_multiindex

log = logging.getLogger(__name__)


class RegionException(Exception):
    pass


class Region:
    def __init__(self, rds, code):
        self._rds = weakref.ref(rds)
        self._code = code
        r = self.data()
        names = [r.Name, r.OfficialName]
        if not pd.isnull(r.OtherNames):
            names.extend(r.OtherNames.split(RegionDataset.SEP))
        names = [n for n in names if not pd.isnull(n) and n]
        rds.data.at[code, "AllNames"] = list(set(names))

    @property
    def Code(self):
        return self._code

    def data(self):
        return self._rds().data.loc[self._code]

    def series(self):
        return self._rds().series.loc[self._code]

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __setattr__(self, name, val):
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            raise AttributeError(
                f"Setting attribute {name} on {self.__class__.__name__} not allowed"
                " (use indexing)."
            )

    def __getitem__(self, name):
        return self._rds().data.at[self._code, name]

    # TODO(gavento): Needs a bit of thought re data/series assignment of new columns
    #                (or just forbid new columns?)
    # def __setitem__(self, name, value):
    #    self._rds().data.at[self._code, name] = value

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._code} {self.Name} ({self.Level})>"


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

    SKIP_SAVING_COLS = {"AllNames"}

    BASIC_COL_TYPES = OrderedDict(
        #        Parent="string",
        # ASCII name (unidecoded)
        Name="U",
        # Official name (any charactersscript)
        B_OfficialName="U",
        # OtherNames, incl orig. name unicode if different
        # encoded as '|'-separated list
        B_OtherNames="U",
        # Administrative level
        B_Level=LEVELS,
        # Countries and above
        B_M49Code="U",
        # Location in hierarchy
        B_ContinentCode="U",
        B_SubregionCode="U",
        B_CountryCode="U",
        B_CountryCodeISOa3="U",
        # Other data
        B_Lat="f4",
        B_Lon="f4",
        B_Population="f4",
        B_GleamID="int32",
    )

    def __init__(self):
        # Single-level DataFrame
        self.data = self._empty_basic_dataframe()
        # DataFrame with Multiindex:
        self.series = self._empty_series_dataframe()
        self.col_groups = {"basic": list(self.BASIC_COL_TYPES.keys())}
        # name: [code, code, ..]
        self.name_index = {}
        # code: [Region, Region, ...]
        self.code_index = {}

    @property
    def regions(self):
        return self.code_index.values()

    @property
    def columns(self):
        return list(self.data.columns) + list(self.series.columns.levels[0])

    @property
    def dates(self):
        return self.series.columns.levels[1]

    @classmethod
    def from_csv(cls, path):
        s = cls()
        data = pd.read_csv(
            path,
            dtype=cls.BASIC_COL_TYPES,
            index_col="Code",
            na_values=[""],
            keep_default_na=False,
        )
        s.data = s.data.append(data)
        s.rebuild_index()
        return s

    def rebuild_index(self):
        self.name_index = {}
        self.code_index = {}
        self.data["AllNames"] = pd.Series(dtype=object)
        conflicts = []
        for ri in self.data.index:
            reg = Region(self, ri)
            for n in reg.AllNames:
                self.name_index.setdefault(normalize_name(n), list()).append(ri)
            self.code_index[ri] = reg
        for k in self.name_index:
            self.name_index[k] = list(set(self.name_index[k]))
            if len(self.name_index[k]) > 1:
                conflicts.append(k)
        if conflicts:
            log.info(
                f"Name index has {len(conflicts)} potential conflicts: {conflicts!r}"
            )

    def __getitem__(self, code):
        """
        Returns the Region corresponding to code, or raise KeyError.
        """
        return self.code_index[code.upper()]

    def find_all_by_name(self, s, levels=None):
        """
        Return all Regions with some matching names (filtering on levels).
        """
        if isinstance(levels, str):
            levels = [levels]
        rs = self.name_index.get(normalize_name(s), [])
        if levels is not None:
            rs = [self[r] for r in rs if self[r].Level in levels]
        return rs

    def find_one_by_name(self, s, levels=None):
        """
        Find one region matching name (filter on levels).
        
        Raises RegionException if no or multiple regions found.
        """
        r = self.find_all_by_name(s, levels=levels)
        if len(r) == 1:
            return r[0]
        lcmt = "" if levels is None else f" [levels={levels!r}]"
        if len(r) < 1:
            raise RegionException(f"Found no regions matching {s!r}{lcmt}")
        raise RegionException(f"Found multiple regions matching {s!r}{lcmt}: {r!r}")

    def add_column(self, c, name=None, date=None, prefix=None):
        """
        Add given series or array as a new column.
        
        If date is given, adds to self.series.
        """
        if name is None:
            name = c.name
        if prefix is not None:
            name = f"{prefix.rstrip('_')}_{name}"
        if date is None:
            self.data[name] = c
        else:
            assert isinstance(date, datetime.date)
            self.series[(name, date)] = c

    def add_dataframe(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            assert all(isinstance(x, str) for x in df.columns.levels[0])
            assert all(isinstance(x, datetime.date) for x in df.columns.levels[1])
            self.series = pd.concat((self.series, df), axis=1)
        else:
            assert all(isinstance(x, str) for x in df.columns)
            self.data = pd.concat((self.data, df), axis=1)

    def read_csv(self, path, dtype=None, na_values=("",), prefix=None):
        """
        Read given CSV and add it to the dataframe.
        
        Detects columns with dates vs plain columns. Optionally adds
        the given prefix (with '_').
        """
        data = pd.read_csv(
            path,
            dtype=dtype,
            index_col="Code",
            na_values=na_values,
            keep_default_na=False,
        )
        for c in data.columns:
            if self.SEP in c:
                cs = c.split(self.SEP)
                assert len(cs) == 2
                date = dateutil.parser.parse(cs[1]).date()
                self.add_column(data[c], date=date, name=cs[0], prefix=prefix)
            else:
                self.add_column(data[c], name=c, prefix=prefix)

    def to_csv(self, path, *, columns=None, match=None, include_name=True):
        """
        Save some columns to CSV file.
        
        Columns can be given either by regexp `match`, prefix of column list.
        Both plain columns and series are saved.
        Any prefix is NOT removed from column names.
        """
        if columns is None:
            columns = []
        if match is not None:
            for c in self.columns:
                if re.match(match, c):
                    columns.append(c)
        columns = set(columns).difference(self.SKIP_SAVING_COLS)
        assert len(columns) > 0
        if "B_OtherNames" in columns:
            self._reconstruct_OtherNames()
        if include_name:
            columns.add("Name")

        series = []
        for c in self.data.columns:
            if c in columns:
                series.append(self.data[c])
        for c, date in self.series.columns:
            if c in columns:
                s = self.series[(c, date)]
                s.name = f"{c}{self.SEP}{date.isoformat()}"
                series.append(s)

        d2 = pd.concat(series, axis=1)
        d2.to_csv(path, index_label="Code")

    def _reconstruct_OtherNames(self):
        "Reconstruct the OtherNames column."
        for r in self.regions:
            names = set(r.AllNames)
            if r.Name in names:
                names.remove(r.Name)
            if r.OfficialName in names:
                names.remove(r.OfficialName)
            r["OtherNames"] = self.SEP.join(names)

    @classmethod
    def _empty_basic_dataframe(cls, index=()):
        df = cls._empty_dataframe(index=index)
        for name, dtype in cls.BASIC_COL_TYPES.items():
            df[name] = pd.Series(dtype=dtype, name=name)
        return df

    @classmethod
    def _empty_dataframe(cls, index=()):
        return pd.DataFrame(index=pd.Index(index, name="Code", dtype=pd.StringDtype()))

    @classmethod
    def _empty_series_dataframe(cls, index=()):
        return pd.DataFrame(
            index=pd.Index(index, name="Code", dtype=pd.StringDtype()),
            columns=pd.MultiIndex.from_tuples([], names=["Property", "Date"]),
        )
