import epimodel


def test_load_index_region(datadir):
    rds = epimodel.RegionDataset.load(datadir / "regions.csv")
    assert rds["CZ"].CountryCodeISOa3 == "CZE"
    r = rds.find_one_by_name("Czech Republic")
    assert r.Code == "CZ"
    assert r.CountryCode == "CZ"
