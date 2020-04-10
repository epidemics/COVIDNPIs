import datetime
import getpass
import json
import logging
import socket
import subprocess
from pathlib import Path
from enum import Enum
from typing import Optional, Iterable, Dict, Any

import pandas as pd
import numpy as np
from tqdm import tqdm

from ..regions import Region

log = logging.getLogger(__name__)

MAIN_DATA_FILENAME = "data-CHANNEL-v4.json"


class WebExport:
    """
    Document holding one data export to web. Contains a subset of Regions.
    """

    def __init__(self, date_resample: str, comment=None):
        self.created = datetime.datetime.now().astimezone(datetime.timezone.utc)
        self.created_by = f"{getpass.getuser()}@{socket.gethostname()}"
        self.comment = comment
        self.date_resample = date_resample
        self.export_regions = {}

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
        models: pd.DataFrame,
        simulation_spec: pd.DataFrame,
        rates: Optional[pd.DataFrame],
        hopkins: Optional[pd.DataFrame],
        foretold: Optional[pd.DataFrame],
    ):
        er = WebExportRegion(region, models, simulation_spec, rates, hopkins, foretold)
        self.export_regions[region.Code] = er
        return er

    def write(self, path, name=None):
        if name is None:
            name = f"export-{self.created.isoformat()}"
            if self.comment:
                name += self.comment
        name = name.replace(" ", "_").replace(":", "-")
        outdir = Path(path)
        assert (not outdir.exists()) or outdir.is_dir()
        exdir = Path(path) / name
        log.info(f"Writing WebExport to {exdir} ...")
        exdir.mkdir(exist_ok=False, parents=True)
        for rc, er in tqdm(list(self.export_regions.items()), desc="Writing regions"):
            fname = f"extdata-{rc}.json"
            er.data_url = f"{name}/{fname}"
            with open(exdir / fname, "wt") as f:
                json.dump(er.data_ext, f)
        with open(exdir / MAIN_DATA_FILENAME, "wt") as f:
            json.dump(self.to_json(), f, indent=2, default=types_to_json)
        log.info(f"Exported {len(self.export_regions)} regions to {exdir}")


class WebExportRegion:
    def __init__(
        self,
        region,
        models: pd.DataFrame,
        simulations_spec: pd.DataFrame,
        rates: Optional[pd.DataFrame],
        hopkins: Optional[pd.DataFrame],
        foretold: Optional[pd.DataFrame],
    ):
        assert isinstance(region, Region)
        self.region = region
        # Any per-region data. Large ones should go to data_ext.
        self.data = self.extract_smallish_data(rates, hopkins, foretold)
        # Extended data to be written in a separate per-region file
        # TODO: this is where we should have extra data
        self.data_ext = self.extract_models_data(models, simulations_spec)
        # Relative URL of the extended data file, set on write
        self.data_url = None

    @staticmethod
    def extract_smallish_data(
        rates: Optional[pd.DataFrame],
        hopkins: Optional[pd.DataFrame],
        foretold: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        d = {
            "critical_rates": rates.to_dict() if rates is not None else None,
            "hopkins": {
                "date_index": [x.isoformat() for x in hopkins.index],
                **hopkins.to_dict(orient="list"),
            }
            if hopkins is not None
            else None,
            "foretold": {
                "date_index": [x.isoformat() for x in foretold.index],
                **foretold.loc[:, ["Mean", "Variance"]].to_dict(orient="list"),
            }
            if foretold is not None
            else None,
        }
        return d

    @staticmethod
    def extract_models_data(
        models: pd.DataFrame, simulation_spec: pd.DataFrame
    ) -> Dict[str, Any]:
        d = {
            "date_index": [x.isoformat() for x in models.index.levels[0]],
        }
        traces = []
        for simulation_id, simulation_def in simulation_spec.iterrows():
            trace_data = models.xs(simulation_id, level="SimulationID")
            trace = {
                "group": simulation_def["Group"],
                "key": simulation_def["Key"],
                "name": simulation_def["Name"],
                "infected": trace_data.loc[:, "Infected"].tolist(),
                "recovered": trace_data.loc[:, "Recovered"].tolist(),
            }
            traces.append(trace)
        d["traces"] = traces
        return {"models": d}

    def to_json(self):
        d = {
            "data": self.data,
            "data_url": self.data_url,
            "Name": self.region.DisplayName,
        }
        for n in [
            "Population",
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
            d[n] = None if pd.isnull(self.region[n]) else self.region[n]
        return d


def upload_export(dir_to_export, gs_prefix, gs_url, channel="test"):
    """The 'upload' subcommand"""
    CMD = [
        "gsutil",
        "-m",
        "-h",
        "Cache-Control:public,max-age=30",
        "cp",
        "-a",
        "public-read",
    ]
    gs_prefix = gs_prefix.rstrip("/")
    gs_url = gs_url.rstrip("/")
    exdir = Path(dir_to_export)
    assert exdir.is_dir()

    log.info(f"Uploading data folder {exdir} to {gs_prefix}/{exdir.parts[-1]} ...")
    cmd = CMD + ["-Z", "-R", exdir, gs_prefix]
    log.debug(f"Running {cmd!r}")
    subprocess.run(cmd, check=True)

    datafile = MAIN_DATA_FILENAME.replace("CHANNEL", channel)
    gs_tgt = f"{gs_prefix}/{datafile}"
    log.info(f"Uploading main data file to {gs_tgt} ...")
    cmd = CMD + ["-Z", exdir / MAIN_DATA_FILENAME, gs_tgt]
    log.debug(f"Running {cmd!r}")
    subprocess.run(cmd, check=True)
    log.info(f"File URL: {gs_url}/{datafile}")

    if channel != "main":
        log.info(f"Custom web URL: http://epidemicforecasting.org/?channel={channel}")


def types_to_json(obj):
    if isinstance(obj, (np.float16, np.float32, np.float64, np.float128)):
        return float(obj)
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.name
    else:
        raise TypeError(f"Unserializable object {obj} of type {type(obj)}")
