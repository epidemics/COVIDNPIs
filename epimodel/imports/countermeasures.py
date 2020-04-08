import logging

import dateutil
import numpy as np
import pandas as pd
import tqdm

from ..regions import RegionDataset, Level

log = logging.getLogger(__name__)

SKIP_NAMES = {"US", "European Union", "Kosovo", "Palestine", "North Cyprus"}

SUBST_NAMES = {"Macau": "China:Macau", "Hong Kong": "China:Hong Kong"}


def import_countermeasures_csv(rds: RegionDataset, path):

    skipped = set()

    def _lookup(name):
        "Return the proper Code"
        if name in SKIP_NAMES:
            skipped.add(name)
            return None
        if name in SUBST_NAMES:
            name = SUBST_NAMES[name]
        if ":" in name:
            country, prov = (x.strip() for x in name.split(":"))
            if country in SKIP_NAMES or prov in SKIP_NAMES:
                skipped.add(name)
                return None
            c = rds.find_one_by_name(country, levels=Level.country)
            ps = [
                p
                for p in rds.find_all_by_name(prov, levels=Level.subdivision)
                if p.CountryCode == c.Code
            ]
            if len(ps) != 1:
                raise Exception(f"Unique region for {name} not found: {ps}")
            return ps[0].Code
        else:
            return rds.find_one_by_name(name, levels=Level.country).Code

    df = pd.read_csv(
        path,
        dtype={"Country": "string", "Date": "string"},
        index_col=0,
        na_values=[""],
        keep_default_na=False,
    )

    df = df.loc[pd.notnull(df.Country)]
    df["Date"] = [pd.to_datetime(x, utc=True) for x in df["Date"]]
    df["Code"] = [_lookup(x) for x in df["Country"]]
    df = df.loc[pd.notnull(df.Code)]
    dti = pd.MultiIndex.from_arrays([df.Code, df.Date], names=["Code", "Date"])
    del df["Country"]
    del df["Code"]
    del df["Date"]
    df.index = dti
    # df.fillna(0.0, inplace=True)

    if skipped:
        log.info(f"Skipped {len(skipped)} records: {skipped!r}")

    return df
