import logging

import dateutil
import pandas as pd

from ..region_data import RegionDataset

log = logging.getLogger(__name__)


SKIP_NAMES = {
    "Diamond Princess",
    "Grand Princess",
    "MS Zaandam",
    "Recovered",
}

SUBSTITUTES = {
    "Taiwan*": "Taiwan",
    "US": "United States",
}

SUBDIVIDED_CODES = {"CA", "US", "CN", "AU"}


def import_csse_covid(rds: RegionDataset, dir="data", prefix="JH", group="JH"):
    skipped = set()
    for n in ["Recovered", "Confirmed", "Deaths"]:
        d = pd.read_csv(
            f"{dir}/time_series_covid19_{n.lower()}_global.csv",
            dtype={"Country/Region": "string", "Province/State": "string"},
        )
        codes = []

        def skip(r):
            skipped.add(f"{r['Country/Region']}/{r['Province/State']}")
            codes.append("")

        for k, r in d.iterrows():
            prov = r["Province/State"]
            prov = SUBSTITUTES.get(prov, prov)
            country = r["Country/Region"]
            country = SUBSTITUTES.get(country, country)
            if prov in SKIP_NAMES or country in SKIP_NAMES:
                skip(r)
                continue
            cs = rds.find_all(country, levels="country")
            if len(cs) != 1:
                log.debug(f"Non-unique country match for {country}: {cs}")
                skip(r)
                continue
            c = cs[0]

            if pd.isna(prov):
                # Add country
                codes.append(c)
            else:
                # Add province
                if c not in SUBDIVIDED_CODES:
                    skip(r)
                    continue
                ps = rds.find_all(prov, levels="subdivision")
                ps = [p for p in ps if rds[p].Country == c]
                if len(ps) != 1:
                    log.debug(
                        f"Non-unique subdivision match for {country}/{prov}: {ps}"
                    )
                    skip(r)
                    continue
                codes.append(ps[0])

        d["Code"] = codes
        d = d[d.Code != ""]
        destname = f"{prefix}_{n}"
        for cname in d.columns:
            if not cname[0].isdigit():
                continue
            col = pd.Series(d[cname].values, index=d.Code)
            date = dateutil.parser.parse(cname).date()
            rds.add_column(col, group, date, name=destname)

    for date in set(rds.series.columns.get_level_values(1)):
        if (f"{prefix}_Recovered", date) in rds.series.columns:
            val = (
                rds.series[(f"{prefix}_Confirmed", date)]
                - rds.series[(f"{prefix}_Recovered", date)]
                - rds.series[(f"{prefix}_Deaths", date)]
            )
            rds.add_column(val, group, date=date, name=f"{prefix}_Active")

    log.info(f"Skipped {len(skipped)} records: {skipped!r}")
