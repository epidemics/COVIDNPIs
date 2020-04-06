import logging

import dateutil
import pandas as pd
import numpy as np

from ..regions import RegionDataset

log = logging.getLogger(__name__)


SKIP_NAMES = {
    "Diamond Princess",
    "Grand Princess",
    "MS Zaandam",
    "Recovered",
}

SUBSTITUTE_COUNTRY = {
    "Taiwan*": "Taiwan",
    "US": "United States",
}

SUBSTITUTE_PROVINCE = {}

SUBDIVIDED_COUNTRIES = {"CA", "US", "CN", "AU"}

GITHUB_PREFIX = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"


def import_johns_hopkins(rds: RegionDataset, prefix=None):
    """
    Read a DataFrame of John Hopkins data from given directory or URL.
    
    By default loads data from CSSE github.
    """
    if prefix is None:
        prefix = GITHUB_PREFIX
    skipped = set()
    not_found = set()
    conflicts = set()
    ds = []
    for n in ["Recovered", "Confirmed", "Deaths"]:
        d = pd.read_csv(
            f"{prefix}/time_series_covid19_{n.lower()}_global.csv", dtype="U",
        )
        codes = np.full(len(d), "", dtype="U64")

        for i, r in d.iterrows():
            prov = r["Province/State"]
            prov = SUBSTITUTE_PROVINCE.get(prov, prov)
            country = r["Country/Region"]
            country = SUBSTITUTE_COUNTRY.get(country, country)

            if prov in SKIP_NAMES or country in SKIP_NAMES:
                skipped.add((country, prov))
                continue
            rs = rds.find_all_by_name(country, levels="country")
            if len(rs) > 1:
                conflicts.add((country, prov))
                continue
            if len(rs) < 1:
                not_found.add((country, prov))
                continue
            c = rs[0].Code

            if pd.isna(prov):
                # Add country
                codes[i] = c
            else:
                # Add province
                rs = rds.find_all_by_name(prov, levels="subdivision")
                rs = [r for r in rs if r.CountryCode == c]
                if len(rs) < 1:
                    not_found.add((country, prov))
                    continue
                if len(rs) > 1:
                    conflicts.add((country, prov))
                    continue
                codes[i] = rs[0].Code

        d.index = pd.Index(codes, name="Code")
        for col in ["Country/Region", "Province/State", "Lat", "Long"]:
            del d[col]
        d.columns = pd.DatetimeIndex(pd.to_datetime(d.columns, utc=True), name="Date")
        ds.append(
            pd.to_numeric(d.loc[d.index != ""].stack(), downcast="float").to_frame(n)
        )

    if skipped:
        log.info(f"Skipped {len(skipped)} records: {skipped!r}")
    if not_found:
        log.info(f"No matches for {len(not_found)} records: {not_found!r}")
    if conflicts:
        log.info(f"Multiple matches for {len(conflicts)} records: {conflicts!r}")

    df = pd.concat(ds, axis=1).sort_index()
    df["Active"] = df["Confirmed"] - df["Recovered"] - df["Deaths"]
    return df
