import re

import dateutil
import pandas as pd
import unidecode


def read_csv(path, regions=None, date_column="Date", drop_unknown=True, **kwargs):
    """
    Read given CSV and do some basic checks and create indexes.
    
    Checks that the CSV has 'Code' column and uses it as an index.
    If the CSV has 'Date' or `date_column` column, uses it as a secondary index.
    Dates are converted to datetime with UTC timezone.
    If `regions` are given, checks that the CSV codes are known,
    warns if not and drops the unknowns.
    Any other args are passed to `pd.read_csv`.
    """
    data = pd.read_csv(path, index_col="Code", **kwargs)
    if date_column in data.columns:
        dti = pd.DatetimeIndex(pd.to_datetime(data[date_column], utc=True))
        del data[date_column]
        data.index = pd.MultiIndex.from_arrays([data.index, dti])
    data.sort_index()
    # TODO check against regions
    return data


def normalize_name(name):
    """
    Return normalized version of the name for matching region names.

    Name is unidecoded, lowercased, '-' and '_' are replaced by spaces,
    whitespace is stripped.
    """
    return unidecode.unidecode(name).lower().replace("-", " ").replace("_", " ").strip()
