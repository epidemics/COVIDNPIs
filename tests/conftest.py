from pathlib import Path

import pytest

import epimodel


@pytest.fixture
def datadir(request):
    return Path(request.module.__file__).parent / "data"


@pytest.fixture
def regions(datadir):
    return epimodel.RegionDataset.load(datadir / "regions.csv")


@pytest.fixture
def regions_gleam(datadir):
    return epimodel.RegionDataset.load(
        datadir / "regions.csv", datadir / "regions-gleam.csv"
    )
