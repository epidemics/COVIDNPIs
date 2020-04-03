import logging

import dateutil
import numpy as np
import pandas as pd
import tqdm

from ..regions import RegionDataset

log = logging.getLogger(__name__)

SKIP_NAMES = {"US", "European Union", "Kosovo", "Palestine"}

SUBST_NAMES = {"Macau": "China:Macau", "Hong Kong": "China:Hong Kong"}


def _lookup(s, name):
    "Return the proper Code"
    if name in SKIP_NAMES:
        return None
    if name in SUBST_NAMES:
        name = SUBST_NAMES[name]
    if ":" in name:
        country, prov = (x.strip() for x in name.split(":"))
        if country in SKIP_NAMES or prov in SKIP_NAMES:
            return None
        c = s.find_one_by_name(country, levels="country")
        ps = [
            p
            for p in s.find_all_by_name(prov, levels="subdivision")
            if p.CountryCode == c.Code
        ]
        if len(ps) != 1:
            raise Exception(f"Unique region for {name} not found: {ps}")
        return ps[0].Code
    else:
        return s.find_one_by_name(name, levels="country").Code


def import_simplified_countermeasures(rds: RegionDataset, path):
    df = pd.read_csv(
        path,
        dtype={"Country": "string", "Date": "string"},
        index_col=0,
        na_values=[""],
        keep_default_na=False,
    )
    df = df.loc[pd.notnull(df.Country)]
    df["Date"] = [pd.to_datetime(x, utc=True) for x in df["Date"]]
    df["Code"] = [_lookup(rds, x) for x in df["Country"]]
    df = df.loc[pd.notnull(df.Code)]
    dti = pd.MultiIndex.from_arrays([df.Code, df.Date], names=["Code", "Date"])
    del df["Country"]
    del df["Code"]
    del df["Date"]
    df.index = dti
    df.fillna(0.0, inplace=True)
    return df
