import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from ..regions import RegionDataset, Level

from urllib.error import HTTPError

log = logging.getLogger(__name__)


SKIP_NAMES = {"Diamond Princess", "Grand Princess", "MS Zaandam", "Recovered"}

SUBSTITUTE_COUNTRY = {"Taiwan*": "Taiwan", "US": "United States"}

SUBSTITUTE_PROVINCE = {}

SUBDIVIDED_COUNTRIES = {"CA", "US", "CN", "AU"}

SUBDIVIDED_DATASETS = ["US"]  # datasets with more granular data

DROP_COLUMNS = {
    "US": [
        "UID",
        "iso2",
        "iso3",
        "code3",
        "FIPS",
        "Admin2",
        "Combined_Key",
        "Population",
        "Lat",
        "Long",
        "Long_",
    ],
    "global": ["Lat", "Long"],
}

SUBSTITUTE_COLUMNS = {
    "US": {"Province_State": "Province/State", "Country_Region": "Country/Region"},
    "global": {},
}

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
            f"{prefix}/time_series_covid19_{n.lower()}_global.csv",
            dtype="U",
            usecols=lambda x: x not in DROP_COLUMNS["global"],
        )
        d = d.apply(pd.to_numeric, downcast="float", errors="ignore")

        for r in SUBDIVIDED_DATASETS:
            try:
                rd = pd.read_csv(
                    f"{prefix}/time_series_covid19_{n.lower()}_{r}.csv",
                    dtype="U",
                    usecols=lambda x: x not in DROP_COLUMNS[r],
                )
            except (HTTPError, FileNotFoundError) as e:
                log.info(f"Category '{n}' not found for regional record {r}")
                continue

            rd.rename(columns=SUBSTITUTE_COLUMNS[r], inplace=True)

            rd = rd.apply(pd.to_numeric, downcast="float", errors="ignore")

            rd = rd.groupby(by=["Country/Region", "Province/State"]).agg("sum")
            rd.reset_index(inplace=True)
            d = pd.concat([d, rd])
            # n.b. this yields unexpected results if the date columns in the global record do not match those in the US record

        d.reset_index(inplace=True)

        codes = np.full(len(d), "", dtype="U64")

        for i, r in d.iterrows():
            prov = r["Province/State"]
            prov = SUBSTITUTE_PROVINCE.get(prov, prov)
            country = r["Country/Region"]
            country = SUBSTITUTE_COUNTRY.get(country, country)

            if prov in SKIP_NAMES or country in SKIP_NAMES:
                skipped.add((country, prov))
                continue
            rs = rds.find_all_by_name(country, levels=Level.country)
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
                rs = rds.find_all_by_name(prov, levels=Level.subdivision)
                rs = [r for r in rs if r.CountryCode == c]
                if len(rs) < 1:
                    not_found.add((country, prov))
                    continue
                if len(rs) > 1:
                    conflicts.add((country, prov))
                    continue
                codes[i] = rs[0].Code

        d.index = pd.Index(codes, name="Code")

        for col in ["Country/Region", "Province/State", "index"]:
            del d[col]
        d.columns = pd.DatetimeIndex(pd.to_datetime(d.columns, utc=True), name="Date")

        ds.append(d.loc[d.index != ""].stack().to_frame(n))

    if skipped:
        log.info(f"Skipped {len(skipped)} records: {skipped!r}")
    if not_found:
        log.info(f"No matches for {len(not_found)} records: {not_found!r}")
    if conflicts:
        log.info(f"Multiple matches for {len(conflicts)} records: {conflicts!r}")

    df = pd.concat(ds, axis=1).sort_index()
    df["Active"] = df["Confirmed"] - df["Recovered"] - df["Deaths"]
    return df


def aggregate_countries(
        hopkins: pd.DataFrame, countries_with_provinces: Optional[List[str]], region_dataset,
) -> pd.DataFrame:
    if not countries_with_provinces:
        return hopkins

    to_append = []
    all_state_codes = []
    for country_code in countries_with_provinces:
        _state_codes = [x.Code for x in region_dataset.get(country_code).children]
        present_state_codes = list(
            set(_state_codes).intersection(hopkins.index.get_level_values("Code"))
        )
        log.info(
            "Aggregating hopkins data for %s into a single code %s",
            present_state_codes,
            country_code,
        )
        aggregated = (
            hopkins.loc[present_state_codes]
                .reset_index("Date")
                .groupby("Date")
                .sum()
                .assign(Code=country_code)
                .reset_index()
                .set_index(["Code", "Date"])
        )
        to_append.append(aggregated)
        all_state_codes.extend(present_state_codes)
    return hopkins.drop(index=all_state_codes).append(pd.concat(to_append))
