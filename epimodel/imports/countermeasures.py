import logging

import dateutil
import numpy as np
import pandas as pd
import tqdm

from ..region_data import RegionDataset, RegionException

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
        c = s.find_one(country, levels="country")
        ps = [p for p in s.find_all(prov, levels="subdivision") if p.Country == c.Code]
        if len(ps) != 1:
            raise RegionException(f"Unique region for {name} not found: {ps}")
        return ps[0].Code
    else:
        return s.find_one(name, levels="country").Code


def import_simplified_countermeasures(
    rds: RegionDataset, path, prefix="SCM", group="SCM"
):
    df = pd.read_csv(
        path,
        dtype={"Country": "string", "Date": "string"},
        index_col=0,
        na_values=[""],
        keep_default_na=False,
    )
    df = df.loc[pd.notnull(df.Country)]
    df["Date"] = [dateutil.parser.parse(x).date() for x in df["Date"]]
    # d=d.pivot(columns=["Date"])
    df["Code"] = [_lookup(rds, x) for x in df["Country"]]
    df = df.loc[pd.notnull(df.Code)]
    features = df.columns[:-3]
    df = df.pivot(index="Code", values=features, columns="Date")
    df.fillna(0.0, inplace=True)
    rds.add_dataframe(df[features], group)
    return

    # Propagate updates to per-day state, add tp rds
    for f in tqdm.tqdm(features):
        state = pd.Series(np.zeros(len(df)), index=df.index)
        for date in sorted(df.columns.levels[1]):
            non_nan = pd.notnull(df[(f, date)])
            state[non_nan] = df[(f, date)][non_nan]
            rds.add_column(state, group, date=date, name=f)
            # state_df[(f, date)] = state
