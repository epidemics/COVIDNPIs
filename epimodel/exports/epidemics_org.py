import datetime
import getpass
import json
import os
import logging
import shutil
import socket
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..gleam import Batch
from ..regions import Region, RegionDataset, Level
import epimodel

log = logging.getLogger(__name__)


class WebExport:
    """
    Document holding one data export to web. Contains a subset of Regions.
    """

    def __init__(self, date_resample: str, comment=None):
        self.created = datetime.datetime.now().astimezone(datetime.timezone.utc)
        self.created_by = f"{getpass.getuser()}@{socket.gethostname()}"
        self.comment = comment
        self.date_resample = date_resample
        self.export_regions: Dict[str, WebExportRegion] = {}

    def to_json(self):
        return {
            "created": self.created,
            "created_by": self.created_by,
            "comment": self.comment,
            "date_resample": self.date_resample,
            "regions": {k: a.to_json() for k, a in self.export_regions.items()},
        }

    def new_region(
        self,
        region,
        current_estimate: Optional[pd.DataFrame],
        groups,
        models: pd.DataFrame,
        simulation_spec: pd.DataFrame,
        rates: Optional[pd.DataFrame],
        hopkins: Optional[pd.DataFrame],
        foretold: Optional[pd.DataFrame],
        timezones: Optional[pd.DataFrame],
        un_age_dist: Optional[pd.DataFrame],
    ):
        er = WebExportRegion(
            region,
            current_estimate,
            groups,
            models,
            simulation_spec,
            rates,
            hopkins,
            foretold,
            timezones,
            un_age_dist,
        )
        self.export_regions[region.Code] = er
        return er

    def write(
        self,
        export_directory: Path,
        main_data_filename: Path,
        latest=None,
        pretty_print=False,
    ):
        indent = None
        if pretty_print:
            indent = 4

        os.makedirs(export_directory, exist_ok=False)

        log.info(f"Writing WebExport to {export_directory} ...")
        for rc, er in tqdm(list(self.export_regions.items()), desc="Writing regions"):
            fname = f"extdata-{rc}.json"
            er.data_url = f"{fname}"
            with open(export_directory / fname, "wt") as f:
                json.dump(
                    er.data_ext,
                    f,
                    default=types_to_json,
                    allow_nan=False,
                    separators=(",", ":"),
                    indent=indent,
                )
        with open(export_directory / main_data_filename, "wt") as f:
            json.dump(
                self.to_json(),
                f,
                default=types_to_json,
                allow_nan=False,
                separators=(",", ":"),
                indent=indent,
            )
        log.info(f"Exported {len(self.export_regions)} regions to {export_directory}")
        if latest is not None:
            latestdir = Path(os.path.dirname(export_directory)) / latest
            if latestdir.exists():
                shutil.rmtree(latestdir)
            shutil.copytree(export_directory, latestdir)
            log.info(f"Copied export to {latestdir}")


class WebExportRegion:
    def __init__(
        self,
        region: Region,
        current_estimate: Optional[pd.DataFrame],
        groups,
        models: pd.DataFrame,
        simulations_spec: pd.DataFrame,
        rates: Optional[pd.DataFrame],
        hopkins: Optional[pd.DataFrame],
        foretold: Optional[pd.DataFrame],
        timezones: Optional[pd.DataFrame],
        un_age_dist: Optional[pd.DataFrame],
    ):
        log.debug(f"Prepare WebExport: {region.Code}, {region.Name}")

        assert isinstance(region, Region)
        self.region = region
        self.current_estimate = current_estimate
        # Any per-region data. Large ones should go to data_ext.
        self.data = self.extract_smallish_data(
            rates, hopkins, foretold, timezones, un_age_dist
        )
        # Extended data to be written in a separate per-region file
        self.data_ext = self.extract_external_data(models, simulations_spec, groups)
        # Relative URL of the extended data file, set on write
        self.data_url = None

    def extract_smallish_data(
        self,
        rates: Optional[pd.DataFrame],
        hopkins: Optional[pd.DataFrame],
        foretold: Optional[pd.DataFrame],
        timezones: pd.DataFrame,
        un_age_dist: Optional[pd.DataFrame],
    ) -> Dict[str, Dict[str, Any]]:
        d = {}

        if rates is not None:
            d["Rates"] = rates.replace({np.nan: None}).to_dict()

        if hopkins is not None:
            nulls = hopkins.isna().sum()
            if (nulls != 0).any():
                # this happens e.g. for countries with provinces and not in config
                log.warning(
                    "Some hopkins data for %s contains empty values: %s.",
                    self.region.Code,
                    nulls.to_dict(),
                )
            d["JohnsHopkins"] = {
                "Date": [x.date().isoformat() for x in hopkins.index],
                **hopkins.astype("Int64").replace({pd.NA: None}).to_dict(orient="list"),
            }

        if foretold is not None:
            d["Foretold"] = {
                "Date": [x.isoformat() for x in foretold.index],
                **foretold.replace({np.nan: None})
                .loc[:, ["Mean", "Variance", "0.05", "0.50", "0.95"]]
                .to_dict(orient="list"),
            }

        d["Timezones"] = timezones["Timezone"].tolist()

        if un_age_dist is not None:
            d["AgeDist"] = un_age_dist.to_dict()

        return d

    @staticmethod
    def get_stats(
        cummulative_active_df: pd.DataFrame, simulation_specs: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        stats = {}
        for group in simulation_specs.Group.unique():
            sim_ids = list(simulation_specs[simulation_specs.Group == group].index)
            group_stats = Batch.generate_sim_stats(cummulative_active_df, sim_ids)
            stats[group] = group_stats
        return stats

    @staticmethod
    def get_date_index(models: pd.DataFrame) -> Iterable[datetime.datetime]:
        date_indexes = models.groupby(level=0).apply(
            lambda x: x.index.get_level_values("Date")
        )
        first = date_indexes[0]
        for dix in date_indexes:
            is_equal = (dix == first).all()
            if not is_equal:
                raise KeyError("Date indexes of two simulations differ!")
        return first

    @staticmethod
    def extract_external_data(
        models: pd.DataFrame, simulation_spec: pd.DataFrame, groups,
    ) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "date_index": [
                x.date().isoformat() for x in WebExportRegion.get_date_index(models)
            ]
        }
        traces = []
        for simulation_id, simulation_def in simulation_spec.iterrows():
            trace_data = models.loc[simulation_id]
            trace = {
                "group": simulation_def["Group"],
                "key": simulation_def["Key"],
                "name": simulation_def["Name"],
                # note that all of these are from the cummulative DF
                "infected": trace_data.loc[:, "Infected"].tolist(),
                "recovered": trace_data.loc[:, "Recovered"].tolist(),
                "active": trace_data.loc[:, "Active"].tolist(),
            }

            traces.append(trace)

        d["traces"] = traces

        stats = WebExportRegion.get_stats(models, simulation_spec)
        d["statistics"] = stats

        return {"scenarios": groups, "models": d}

    def to_json(self):
        d = {
            "data": self.data,
            "data_url": self.data_url,
            "Name": self.region.DisplayName,
            "CurrentEstimate": self.current_estimate,
        }
        for n in [
            "Lat",
            "Lon",
            "OfficialName",
            "Level",
            "M49Code",
            "ContinentCode",
            "SubregionCode",
            "CountryCode",
            "CountryCodeISOa3",
            "SubdivisionCode",
        ]:
            if not pd.isnull(self.region[n]):
                d[n] = self.region[n]

        if not pd.isnull(self.region["Population"]):
            d["Population"] = int(self.region["Population"])

        if self.current_estimate is not None:
            d["CurrentEstimate"] = self.current_estimate.to_dict()

        return d


def raise_(msg):
    raise Exception(msg)


def assert_valid_json(file, minify=False):
    with open(file, "r") as blob:
        data = json.load(
            blob,
            parse_constant=(lambda x: raise_("Not valid JSON: detected `" + x + "'")),
        )
    if minify:
        with open(file, "wt") as f:
            json.dump(
                data, f, default=types_to_json, allow_nan=False, separators=(",", ":"),
            )


def upload_export(dir_to_export: Path, gs_prefix: str, channel: str):
    CMD = [
        "gsutil",
        "-m",
        "-h",
        "Cache-Control:public,max-age=30",
        "cp",
        "-a",
        "public-read",
    ]
    assert dir_to_export.is_dir()

    for json_file in dir_to_export.iterdir():
        if json_file.suffix != ".json":
            continue
        try:
            assert_valid_json(json_file, minify=True)
        except Exception:
            log.error(f"Error in JSON file {json_file}")
            raise

    release_name = dir_to_export.parts[-1]
    gcs_path = os.path.join(gs_prefix, channel, release_name)
    log.info(f"Uploading data folder {dir_to_export} to {gcs_path} ...")
    cmd = CMD + ["-Z", "-R", dir_to_export.as_posix(), gcs_path]
    log.debug(f"Running {cmd!r}")
    subprocess.run(cmd, check=True)

    if channel != "main":
        log.info(f"Custom web URL: http://epidemicforecasting.org/?channel={channel}")


def types_to_json(obj):
    if isinstance(obj, (np.float16, np.float32, np.float64, np.float128)):
        return float(obj)
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.name
    else:
        raise TypeError(f"Unserializable object {obj} of type {type(obj)}")


def get_cmi(df: pd.DataFrame):
    return df.index.get_level_values("Code").unique()


def analyze_data_consistency(
    debug: Optional[None],
    export_regions: List[str],
    models,
    rates_df,
    hopkins,
    foretold,
) -> None:
    codes = {
        "models": get_cmi(models),
        "hopkins": get_cmi(hopkins),
        "foretold": get_cmi(foretold),
        "rates": rates_df.index.unique(),
    }

    fatal = list()
    to_export = set(export_regions)

    union_codes = set()
    any_nan = False
    for source_name, ixs in codes.items():
        if ixs.isna().sum() > 0:
            log.error("Dataset %s contains NaN in index!", source_name)
            fatal.append("Some datasets indexed by NaNs.  Fix the source data.")
            any_nan = True
        union_codes.update(ixs)

    df = pd.DataFrame(index=sorted(union_codes))
    for source_name, ixs in codes.items():
        df[source_name] = pd.Series(True, index=ixs)
    df = df.fillna(False)

    log.info("Modelled %s", set(codes["models"]).difference(export_regions))

    log.info("Total data availability, number of locations: %s", df.sum().to_dict())
    log.info("Export requested for %s regions: %s", len(export_regions), export_regions)

    if debug:
        _df = df.loc[export_regions, ["hopkins", "rates"]]
        res = _df.loc[~_df.all(axis=1)].replace({False: "Missing", True: "OK"})
        log.debug(
            "Data presence for hopkins or rates in the following countries: \n%s", res
        )

    diff_export_and_models = to_export.difference(codes["models"])
    if diff_export_and_models:
        log.error(
            "You requested to export %s but that's not modelled yet.",
            diff_export_and_models,
        )
        fatal.append(f"Regions {diff_export_and_models} not present in modelled data.")

    if fatal:
        raise ValueError(
            "Cannot procede for the following reasons:\n - " + ("\n - ".join(fatal))
        )

    log.info(
        "From exported regions (N=%s): %s",
        len(export_regions),
        df.loc[export_regions].sum().to_dict(),
    )


def get_df_else_none(df: pd.DataFrame, code) -> Optional[pd.DataFrame]:
    if code in df.index:
        return df.loc[code].sort_index()
    else:
        return None


def get_df_list(df: pd.DataFrame, code) -> pd.DataFrame:
    if code not in df.index:
        return df.loc[[]]
    return df.loc[[code]].sort_index()


def get_extra_path(config, name: str) -> Path:
    return Path(config["data_dir"]) / config["web_export"][name]


def aggregate_countries(
    hopkins: pd.DataFrame, countries_with_provinces: List[str], region_dataset,
) -> pd.DataFrame:
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


def add_aggregate_traces(aggregate_regions, cummulative_active_df):
    additions = []

    for reg in aggregate_regions:
        # compute aggregate weights
        agg_codes = [child.Code for child, _ in reg.agg_children]
        weights = pd.Series(
            [child.Population * weight for child, weight in reg.agg_children],
            index=agg_codes,
        )
        weights /= weights.sum()

        log.info(
            "Aggregating model traces for region %s using weights %r",
            reg.Code,
            weights.to_dict(),
        )

        # weight totals for each aggregated region
        reg_cad = cummulative_active_df.loc[pd.IndexSlice[:, agg_codes], :].copy()
        for child, weight in weights.items():
            reg_cad.loc[pd.IndexSlice[:, child], :] *= weight

        # combine weighted values and add to output
        additions.append(reg_cad.groupby(level=["SimulationID", "Date"]).sum())

    if len(additions) > 0:
        # re-add Code index & combine results
        additions_df = pd.concat(
            additions, keys=[reg.Code for reg in aggregate_regions], names=["Code"]
        ).reorder_levels(["SimulationID", "Code", "Date"])

        return cummulative_active_df.append(additions_df)
    else:
        return cummulative_active_df


def process_export(
    config: dict,
    inputs: dict,
    rds: RegionDataset,
    debug,
    comment,
    batch_file,
    estimates,
) -> WebExport:
    ex = WebExport(config["gleam_resample"], comment=comment)

    hopkins = inputs["hopkins"].path
    foretold = inputs["foretold"].path
    rates = inputs["rates"].path
    timezone = inputs["timezones"].path
    un_age_dist = inputs["age_distributions"].path

    export_regions = sorted(config["export_regions"])

    batch = Batch.open(batch_file)
    simulation_specs: pd.DataFrame = batch.hdf["simulations"]
    cummulative_active_df = batch.get_cummulative_active_df()

    estimates_df = epimodel.read_csv_smart(estimates, rds, prefer_higher=True)

    rates_df: pd.DataFrame = pd.read_csv(
        rates, index_col="Code", keep_default_na=False, na_values=[""]
    )
    timezone_df: pd.DataFrame = pd.read_csv(
        timezone, index_col="Code", keep_default_na=False, na_values=[""],
    )

    un_age_dist_df: pd.DataFrame = pd.read_csv(un_age_dist, index_col="Code M49").drop(
        columns=["Type", "Region Name", "Parent Code M49"]
    )

    foretold_df: pd.DataFrame = pd.read_csv(
        foretold,
        index_col=["Code", "Date"],
        parse_dates=["Date"],
        keep_default_na=False,
        na_values=[""],
    )

    hopkins_df: pd.DataFrame = pd.read_csv(
        hopkins,
        index_col=["Code", "Date"],
        parse_dates=["Date"],
        keep_default_na=False,
        na_values=[""],
    ).pipe(aggregate_countries, config["state_to_country"], rds)

    cummulative_active_df = add_aggregate_traces(
        [reg for reg in rds.aggregate_regions if reg.Code in export_regions],
        cummulative_active_df,
    )

    analyze_data_consistency(
        debug, export_regions, cummulative_active_df, rates_df, hopkins_df, foretold_df,
    )

    for code in export_regions:
        reg: Region = rds[code]
        m49 = int(reg["M49Code"]) if pd.notnull(reg["M49Code"]) else -1

        ex.new_region(
            reg,
            get_df_else_none(estimates_df, code),
            config["groups"],
            cummulative_active_df.xs(key=code, level="Code").sort_index(level="Date"),
            simulation_specs,
            get_df_else_none(rates_df, code),
            get_df_else_none(hopkins_df, code),
            get_df_else_none(foretold_df, code),
            get_df_list(timezone_df, code),
            get_df_else_none(un_age_dist_df, m49),
        )
    return ex
