import pytest

import epimodel


def test_import_john_hopkins(datadir, regions):
    d = epimodel.imports.import_johns_hopkins(regions, prefix=datadir)
    for c in ["Confirmed", "Recovered", "Deaths", "Active"]:
        assert c in d.columns
    assert all(d.loc[("CZ", "2020-03-20"), "Confirmed"] > 10)
