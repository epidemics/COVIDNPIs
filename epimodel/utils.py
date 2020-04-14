import re

import dateutil
import pandas as pd
import unidecode


def read_csv(
    path,
    rds: "epimodel.RegionDataset",
    date_column: str = "Date",
    skip_unknown: bool = True,
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


def read_csv_names(
    path,
    rds: "epimodel.RegionDataset",
    date_column: str = "Date",
    name_column: str = "Name",
    levels=None,
    drop_underscored: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Read given CSV indexed by name, create indexes and perform basic checks.
    
    Checks that the CSV has name column, finds every name in region dataset
    and and uses it as an index. Name matches must be unique within selected levels
    (see `RegionDataset.find_one_by_name`).

    If the CSV has 'Date' or `date_column` column, uses it as a secondary index.
    Dates are converted to datetime with UTC timezone.

    By default drops any "_Undersored" columns (including the informative "_Name").

    Any other keyword args are passed to `pd.read_csv`.
    """
    data = pd.read_csv(path, **kwargs)
    if name_column not in data.columns:
        raise ValueError(f"CSV file does not have column {name_column}")
    data["Code"] = data[name_column].map(
        lambda n: rds.find_one_by_name(n, levels=levels)
    )
    del data[name_column]
    return _process_loaded_table(
        data, rds, date_column=date_column, drop_underscored=drop_underscored
    )


def _process_loaded_table(
    data: pd.DataFrame,
    rds: "epimodel.RegionDataset",
    date_column: str = "Date",
    drop_underscored: bool = True,
    skip_unknown: bool = True,
):
    """Internal helper for `read_csv{_names}`."""
    if date_column in data.columns:
        dti = pd.DatetimeIndex(pd.to_datetime(data[date_column], utc=True))
        del data[date_column]
        data.index = pd.MultiIndex.from_arrays([data.index, dti])
    if drop_underscored:
        for n in list(data.columns):
            if n.startswith("_"):
                del data[n]

    # TODO check against regions

    return data.sort_index()


def write_csv(df, path, regions=None, with_name=True):
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
