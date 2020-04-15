import epimodel


def test_load_index_region(datadir):
    rds = epimodel.RegionDataset.load(datadir / "regions.csv")
    assert rds["CZ"].CountryCodeISOa3 == "CZE"
    r = rds.find_one_by_name("Czech Republic")
    assert r.Code == "CZ"
    assert r.CountryCode == "CZ"


def test_hierarchy(regions):
    ca = regions["US-CA"]
    assert ca.parent.Code == "US"
    # NOTE: Here could be sub-region!
    assert ca.parent.parent.Code == "W-NA"
    assert ca.parent.parent.parent.Code == "W"
    assert ca.parent.parent.parent.parent is None
    assert ca.country.Code == "US"
    assert ca.country.country.Code == "US"
    assert ca.continent.Code == "W-NA"
    assert ca.subdivision is ca
