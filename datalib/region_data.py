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

from .utils import flatten_multiindex, unflatten_multiindex, normalize_name

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
            raise AttributeError()

    def __getitem__(self, name):
        return self._rds().data.at[self._code, name]

    def __setitem__(self, name, value):
        self._rds().data.at[self._code, name] = value

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
        ["world", "continent", "subregion", "country", "subdivision"], ordered=True
    )
    BASIC_COL_TYPES = OrderedDict(
        Level=LEVELS,
        #        Parent="string",
        # ASCII name (unidecoded)
        Name="string",
        # Official name in unicode
        OfficialName="string",
        # OtherNames, incl orig. name unicode if different
        # encoded as '|'-separated list
        OtherNames="string",
        Continent="string",
        Subregion="string",
        Country="string",
        # Only for countries
        ISOa3="string",
        # Countries and above
        M49Code="string",
        Lat=np.float,
        Lon=np.float,
        Population=pd.Int64Dtype(),
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
        s.build_index()
        return s

    def build_index(self):
        self.name_index = {}
        self.code_index = {}
        self.data["AllNames"] = pd.Series(dtype=object)
        for ri in self.data.index:
            reg = Region(self, ri)
            for n in reg.AllNames:
                self.name_index.setdefault(normalize_name(n), list()).append(ri)
            self.code_index[ri] = reg
        for k in self.name_index:
            self.name_index[k] = list(set(self.name_index[k]))

    def __getitem__(self, code):
        "Returns the data row corresponding to code"
        return self.data.loc[code]

    def find_all(self, s, levels=None):
        "Return all codes with some matching names (filter on levels)"
        if isinstance(levels, str):
            levels = [levels]
        rs = self.name_index.get(normalize_name(s), [])
        if levels is not None:
            rs = [r for r in rs if self[r].Level in levels]
        return rs

    def add_column(self, c, group, date=None, name=None):
        if name is None:
            name = c.name
        if date is None:
            self.data[name] = c
        else:
            assert isinstance(date, datetime.date)
            self.series[(name, date)] = c
        self.col_groups.setdefault(group, []).append(name)

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

    def read_csv(self, path, dtype=None, group=None, na_values=()):
        if group is None:
            group = Path(path).stem
        data = pd.read_csv(
            path,
            dtype=dtype,
            index_col="Code",
            na_values=[""] + list(na_values),
            keep_default_na=False,
        )
        for c in data.columns:
            if self.SEP in c:
                cs = c.split(self.SEP)
                assert len(cs) == 2
                date = dateutil.parser.parse(cs[1]).date()
                self.add_column(data[c], group, date=date, name=c)
            else:
                self.add_column(data[c], group, name=c)

    def read_csv_groups(self, prefix):
        "Load all groups with given prefix."
        p = Path(prefix).parent
        n = Path(prefix).name
        for f in p.glob(f"{n}-*.csv"):
            gn = f.stem[len(n) + 1 :]
            log.info(f"Loading group {gn} from {f} ...")
            self.read_csv(f, group=gn)

    def _reconstruct_OtherNames(self):
        "Reconstruct the OtherNames column."
        for r in self.regions:
            names = set(r.AllNames)
            if r.Name in names:
                names.remove(r.Name)
            if r.OfficialName in names:
                names.remove(r.OfficialName)
            r["OtherNames"] = self.SEP.join(names)

    def to_csv(self, path, group=None, *, columns=None):
        "Save one group or list of columns (both series and plain cols)"
        if group == "basic":
            self._reconstruct_OtherNames()
        if group is not None:
            columns = self.col_groups[group]
        assert columns is not None
        columns = frozenset(columns)
        series = []
        for c, date in self.series.columns:
            if c in columns:
                s = self.series[(c, date)]
                s.name = f"{c}{self.SEP}{date.isoformat()}"
                series.append(s)
        for c in self.data.columns:
            if c in columns:
                series.append(self.data[c])
        d2 = pd.concat(series, axis=1)
        d2.to_csv(path, index_label="Code")

    def to_csv_all_groups(self, prefix, include_basic=False):
        "Save all groups (except 'basic') with given prefix."
        for g in self.col_groups:
            if g != "basic" or include_basic:
                self.to_csv(f"{prefix}-{g}.csv")
