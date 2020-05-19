from pathlib import Path
import pytest
from _pytest.tmpdir import _mk_tmp
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

@pytest.fixture(scope="class")
def ut_tmp_path(request, tmp_path_factory):
    """
    pytest provides no class-scoped tmp_path. This function copies the
    implementation, but it must use private modules to do so.
    """
    request.cls.tmp_path = _mk_tmp(request, tmp_path_factory)


# shared logic


def _datadir(request):
    return Path(request.module.__file__).parent / "data"


def _regions(datadir):
    return epimodel.RegionDataset.load(datadir / "regions.csv")


def _regions_gleam(datadir):
    return epimodel.RegionDataset.load(
        datadir / "regions.csv", datadir / "regions-gleam.csv"
    )
