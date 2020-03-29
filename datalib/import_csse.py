import logging

import dateutil
import pandas as pd

from .region_data import RegionDataset

log = logging.getLogger(__name__)


def import_csse_covid(rds: RegionDataset, dir="data", prefix="JH", group="JH"):
    for n in ["Recovered", "Confirmed", "Deaths"]:
        d = pd.read_csv(f"{dir}/time_series_covid19_{n.lower()}_global.csv")
        codes = []
        for k, r in d.iterrows():
            prov = r["Province/State"]
            country = r["Country/Region"]
            cs = rds.find_all(country, levels="country")
            if len(cs) != 1:
                log.warning(f"Non-unique country match for {country}: {cs}")
                codes.append("")
                continue
            c = cs[0]
            if pd.isna(prov):
                codes.append(c)
            else:
                ps = rds.find_all(country, levels="subdivision")
                ps = [p for p in ps if rds[p].Country == c]
                if len(ps) != 1:
                    log.warning(f"Non-unique subdivision match for {country}/{prov}: {ps}")
                    codes.append("")
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
