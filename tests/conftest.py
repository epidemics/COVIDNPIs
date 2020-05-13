from pathlib import Path

import pytest

import epimodel


@pytest.fixture
def datadir(request):
    return _datadir(request)


@pytest.fixture
def regions(datadir):
    return _regions(datadir)


@pytest.fixture
def regions_gleam(datadir):
    return _regions_gleam(datadir)


# unittest


@pytest.fixture(scope="class")
def ut_datadir(request):
    request.cls.datadir = _datadir(request)


@pytest.fixture(scope="class")
def ut_rds(request, ut_datadir):
    cls = request.cls
    cls.rds = _regions_gleam(cls.datadir)


# shared logic


def _datadir(request):
    return Path(request.module.__file__).parent / "data"


def _regions(datadir):
    return epimodel.RegionDataset.load(datadir / "regions.csv")


def _regions_gleam(datadir):
    return epimodel.RegionDataset.load(
        datadir / "regions.csv", datadir / "regions-gleam.csv"
    )
