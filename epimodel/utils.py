import re

import dateutil
import pandas as pd
import unidecode


def normalize_name(name):
    return unidecode.unidecode(name).lower().replace("-", " ").replace("_", " ").strip()


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
            date = dateutil.parser.parse(cns[1])
            if re.match("....-..-..$", cns[1].strip()):
                date = date.date()
            d2[(cns[0], date)] = col
    return d2
