import sys

from epimodel import RegionDataset, Level


def main(files):
    rds = RegionDataset.load(*files)
    for r in rds.regions:
        print(r)
        if r.Level <= Level.gleam_basin:
            assert r.subdivision is r.parent.subdivision
        if r.Level <= Level.subdivision:
            assert r.country.Code
            assert r.country == r.parent.country
            assert r.CountryCodeISOa3 == r.parent.CountryCodeISOa3
        if r.Level == Level.subdivision:
            assert r.Code == r.SubdivisionCode
        if r.Level <= Level.country:
            assert r.continent.Code
            assert r.continent == r.parent.continent
        if r.Level == Level.country:
            assert r.Code == r.CountryCode
        if r.Level == Level.continent:
            assert r.Code == r.ContinentCode
            assert r.parent.Code == "W"


if __name__ == "__main__":
    main(sys.argv[1:])
