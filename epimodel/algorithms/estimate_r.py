from typing import List
import tempfile
import os
from subprocess import Popen
import pandas as pd

from epimodel.imports.johns_hopkins import aggregate_countries
from epimodel.regions import RegionDataset

project_dir = os.path.join(os.path.dirname(__file__), "../..")
script_dir = os.path.join(project_dir, "scripts/r_estimate")


def preprocess_hopkins(
    hopkins_file: str,
    rds: RegionDataset,
    state_to_country: List[str],
) -> pd.DataFrame:
    return pd.read_csv(
        hopkins_file,
        index_col=["Code", "Date"],
        parse_dates=["Date"],
        keep_default_na=False,
        na_values=[""],
    ).pipe(aggregate_countries, state_to_country, rds)


def estimate_r(
    r_script_executable: str,
    output_file: str,
    john_hopkins_csv: str,
    serial_interval_sample: str,
    rds: RegionDataset,
    state_to_country: List[str]
):
    hopkins_df = preprocess_hopkins(
        john_hopkins_csv,
        rds,
        state_to_country
    )
    hopkins_file = tempfile.mktemp()
    hopkins_df.to_csv(hopkins_file)

    process = Popen([
        r_script_executable,
        os.path.join(script_dir, "estimate_R.R"),
        serial_interval_sample,
        hopkins_file,
        output_file
    ])
    _ = process.communicate()
    rc = process.returncode
    if rc != 0:
        raise RuntimeError("Could not estimate R")
