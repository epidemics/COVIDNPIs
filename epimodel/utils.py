import datetime
import logging
import re
from typing import Union, Set, Optional

import dateutil
import pandas as pd
import unidecode

import epimodel

log = logging.getLogger(__name__)


def read_csv(
    path,
    rds: "epimodel.RegionDataset",
    date_column: str = "Date",
    skip_unknown: bool = False,
    drop_underscored: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Read given CSV indexed by Code, create indexes and perform basic checks.

    Checks that the CSV has 'Code' column and uses it as an index.
    If the CSV has 'Date' or `date_column` column, uses it as a secondary index.
    Dates are converted to datetime with UTC timezone.

    By default drops any "_Undersored" columns (including the informative "_Name").
    If `rds` is not None, checks for region existence. By default skips unknown
    regions (issuing a warning), with `skip_unknwn=False` raises an exception.

    Any other keyword args are passed to `pd.read_csv`.
    """
    data = pd.read_csv(path, index_col="Code", **kwargs)
    return _process_loaded_table(
        data,
        rds,
        date_column=date_column,
        skip_unknown=skip_unknown,
        drop_underscored=drop_underscored,
    )


NAME_COLUMNS = ["Code", "code", "Name", "name"]

DATE_COLUMNS = ["Date", "date", "Day", "day"]


def read_csv_smart(
    path,
    rds: "epimodel.RegionDataset",
    date_column: str = None,
    name_column: str = None,
    skip_unknown: bool = False,
    levels=None,
    drop_underscored: bool = True,
    prefer_higher=False,
    **kwargs,
) -> pd.DataFrame:
    """
    Read given CSV indexed by name or code and optionally date, create indexes and
    perform basic checks.

    For every row, the named region is found in the region dataset by name or code.
    Without `prefer_higher`, name matches must be unique within selected levels (see
    `RegionDataset.find_one_by_name`). With `prefer_higher` the highest level is
    preferred (but still must be unique).

    If not given, the name/code column is auto-detected from "Code", "code", "Name",
    "name". The date column names tried are "Date", "date", "Day", "day".
    (All in that order). If `date_coumn` name is given, it must be present in the file.

    If the CSV has 'Date' or `date_column` column, uses it as a secondary index.
    Dates are converted to datetime with UTC timezone.

    By default drops any "_Undersored" columns (including e.g. the informative "_Name").

    Any other keyword args are passed to `pd.read_csv`.
    """

    def find(n):
        if not isinstance(n, str):
            n = str(n)
        rs = set(rds.find_all_by_name(n, levels=levels))
        if n in rds:
            rs.add(rds[n])
        if prefer_higher:
            max_level = max(r.Level for r in rs)
            rs = set(r for r in rs if r.Level == max_level)
        if len(rs) > 1:
            raise Exception(f"Found multiple matches for {n!r}: {rs!r}")
        elif len(rs) == 1:
            return rs.pop().Code
        elif skip_unknown:
            unknown.add(n)
            return ""
        else:
            raise Exception(f"No region found for {n!r}")

    unknown: Set[str] = set()
    data = pd.read_csv(path, **kwargs)

    if name_column is None:
        for n in NAME_COLUMNS:
            if n in data.columns:
                name_column = n
                break
    if name_column is None:
        raise ValueError(f"CSV file has no column in {NAME_COLUMNS}")
    if name_column not in data.columns:
        raise ValueError(f"CSV file does not have column {name_column}")
    data["Code"] = data[name_column].map(find)
    data = data[data.Code != ""]
    if name_column != "Code":
        del data[name_column]

    if date_column is None:
        for n in DATE_COLUMNS:
            if n in data.columns:
                date_column = n
                break
    if date_column is not None and date_column not in data.columns:
        raise ValueError(f"CSV file does not have column {name_column}")

    if unknown:
        log.warning(f"Skipped unknown regions {unknown!r}")
    data = data.set_index("Code")
    return _process_loaded_table(
        data, rds, date_column=date_column, drop_underscored=drop_underscored
    )


def _process_loaded_table(
    data: pd.DataFrame,
    rds: "epimodel.RegionDataset",
    date_column: Optional[str] = "Date",
    drop_underscored: bool = True,
    skip_unknown: bool = True,
):
    """Internal helper for `read_csv{_names}`."""
    if date_column in data.columns:
        dti = pd.DatetimeIndex(pd.to_datetime(data[date_column], utc=True))
        del data[date_column]
        data.index = pd.MultiIndex.from_arrays(
            [data.index, dti], names=["Code", "Date"]
        )
    if drop_underscored:
        for n in list(data.columns):
            if n.startswith("_"):
                del data[n]

    # TODO check against regions

    return data.sort_index()


def write_csv(df, path, regions=None, with_name=False):
    """
    Write given CSV normally, adding purely informative "_Name" column by default.
    """

    if with_name and regions is None:
        raise ValueError("Provide `regions` with `with_name=True`")
    if with_name:
        ns = pd.Series(regions.data.DisplayName, name="_Name")
        df = df.join(ns, how="inner")
    df.write_csv(path)


def normalize_name(name):
    """
    Return normalized version of the name for matching region names.

    Name is unidecoded, lowercased, '-' and '_' are replaced by spaces,
    whitespace is stripped.
    """
    return unidecode.unidecode(name).lower().replace("-", " ").replace("_", " ").strip()


def utc_date(d: Union[str, datetime.date, datetime.datetime]) -> datetime.datetime:
    """
    Agressively convert any date spec (str, date or datetime) into UTC datetime with time 00:00:00.

    Discards any old time and timezone info!
    Note that dates as UTC 00:00:00 is used as the day identifier throughout epimodel.
    """
    if isinstance(d, str):
        d = dateutil.parser.parse(d)
    if isinstance(d, datetime.date):
        d = datetime.datetime.combine(d, datetime.time())
    if not isinstance(d, datetime.datetime):
        raise TypeError(f"Only str, datetime or date objects accepted, got {d!r}")
    return datetime.datetime.combine(d, datetime.time(tzinfo=datetime.timezone.utc))
