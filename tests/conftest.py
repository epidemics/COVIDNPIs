from pathlib import Path

import pytest

import epimodel


@pytest.fixture
def datadir(request):
    return Path(request.module.__file__).parent / "data"


@pytest.fixture
def regions(datadir):
    return epimodel.RegionDataset.load(datadir / "cli-dir/data/regions.csv")


@pytest.fixture
def regions_gleam(datadir):
    return epimodel.RegionDataset.load(
        datadir / "cli-dir/data/regions.csv", datadir / "cli-dir/data/regions-gleam.csv"
    )
