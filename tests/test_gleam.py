import numpy as np
import pandas as pd
import pytest

import epimodel
from epimodel.utils import utc_date


def test_batch_new_open(tmp_path):
    b = epimodel.gleam.Batch.new(dir=tmp_path)
    path = b.path
    b.close()
    b2 = epimodel.gleam.Batch.open(path)
    b2.close()


def test_gleam_def(datadir):
    d = epimodel.gleam.GleamDefinition(datadir / "files/test_definition.xml")
    assert d.get_id() == "1585188102568.574"
    assert float(d.get_variable("beta")) == pytest.approx(1.01)
    assert d.get_traffic_occupancy() == 20
    assert d.get_start_date().isoformat() == "2020-03-25T00:00:00+00:00"
    assert d.get_seasonality() == pytest.approx(0.85)


def test_add_seeds_add_export_sims(regions_gleam, datadir, tmp_path):
    d = epimodel.gleam.GleamDefinition(datadir / "files/test_definition.xml")
    b = epimodel.gleam.Batch.new(dir=tmp_path)

    i_df = pd.DataFrame(
        {"Infectious": [1, 2, 3], "Exposed": [4, 5, np.nan]},
        index=["G-AAA", "G-AAB", "CZ"],
    )
    d.clear_seeds()
    d.add_seeds(regions_gleam, i_df)

    b.set_simulations([(d, "Name1", "MEDIUM", "WEAK_WEAK")])
    b.set_initial_compartments(i_df)
    path = b.path
    b.close()

    b2 = epimodel.gleam.Batch.open(path)
    assert len(b2.hdf["simulations"]) == 1
    assert len(b2.hdf["initial_compartments"]) == 3
    b2.export_definitions_to_gleam(tmp_path)
    b2.close()


def test_gleam_def_dates(datadir):
    d = epimodel.gleam.GleamDefinition(datadir / "files/test_definition.xml")
    d.set_start_date("2020-10-11")
    d.set_duration(11)
    assert d.get_duration() == 11
    assert d.get_end_date() == utc_date("2020-10-22")


def test_gleam_def_exceptions(datadir, regions_gleam):
    d = epimodel.gleam.GleamDefinition(datadir / "files/test_definition.xml")
    d.clear_exceptions()
    d.add_exception([regions_gleam["CZ"], regions_gleam["G-AAA"]], {})
    d.add_exception([regions_gleam["W-EU"]], {"beta": 1e-10})
