import re
from collections import OrderedDict
from pathlib import Path

import dateutil
import numpy as np
import pandas as pd


def flatten_multiindex(df, sep="|"):
    "Flattens a multiindex of the form `(name, date_or_None)`"
    d2 = pd.DataFrame([], index=pd.Index([], name="Id", dtype=pd.StringDtype()))
    for c in df.columns:
        assert sep not in c[0]
        if c[1] is None or c[1] == "":
            cn = c[0]
        else:
            date = c[1].isoformat()
            assert sep not in date
            cn = f"{c[0]}{sep}{date}"
        d2[cn] = df[c]
    return d2


def unflatten_multiindex(df, sep="|"):
    d2 = pd.DataFrame(
        [],
        index=pd.Index([], name="Id", dtype=pd.StringDtype()),
        columns=pd.MultiIndex.from_tuples([], names=["Property", "Date"]),
    )
    for cn in df.columns:
        cns = cn.split(sep)
        assert len(cns) <= 2 and len(cns) >= 1
        col = df[cn]
        if len(cns) == 1:
            d2[(cns[0], "")] = col
        else:
            date =  dateutil.parser.parse(cns[1])
            if re.match("....-..-..$", cns[1].strip()):
                date = date.date()
            d2[(cns[0], date)] = col
    return d2


class Region:
    def __init__(self, id, name=None):
        self.id = id
        self.name = name


class RegionDataset:
    """

    The Id is:
    # EARTH - root node
    # Continent?
    US - ISOa2 code, Level="country"
    US-CA - ISO 3166-2 state/province code, Level="subdivision"
    """

    LEVELS = pd.CategoricalDtype(["world", "continent", "country", "subdivision"], ordered=True)
    BASIC_COL_TYPES = OrderedDict(
        Level=LEVELS,
        Name="string",
        Parent="string",
        # OtherNames encoded as '|'-separated list
        OtherNames="string",
        ISOa2="string",
        ISOa3="string",
        Lat=np.float,
        Lon=np.float,
        Population=pd.Int64Dtype(),
    )

    @classmethod
    def _empty_data(cls):
        df = pd.DataFrame(
            [],
            index=pd.Index([], name="Id", dtype=pd.StringDtype()),
            columns=pd.MultiIndex.from_tuples([], names=["Property", "Date"]),
        )
        for name, dtype in cls.BASIC_COL_TYPES.items():
            df[name] = pd.Series([], dtype=dtype)
        return df


    def __init__(self, data):
        self.data = data

    @classmethod
    def read_csv(cls, path="data/regions.csv"):
        data = unflatten_multiindex(pd.read_csv(path, dtype=cls.BASIC_COL_TYPES, index_col='Id'))
        return cls(data)

    def to_csv(self, path):
        flatten_multiindex(self.data).to_csv(path, index_label='Id')
