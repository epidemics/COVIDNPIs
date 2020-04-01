from pathlib import Path

import pytest

import epimodel


@pytest.fixture
def datadir(request):
    return Path(request.module.__file__).parent / "data"


@pytest.fixture
def regions(datadir):
    return epimodel.RegionDataset.load(datadir / "regions.csv")
