import pandas as pd
import pycountry
import unidecode

from .region_data import RegionDataset


def import_pycountry_coutries(rds: RegionDataset):
    for c in pycountry.countries:
        ofn = c.name if not hasattr(c, "official_name") else c.official_name
        name = unidecode.unidecode(c.name)
        nms = []
        if name != unidecode.unidecode(c.name):
            nms.append(c.name)

        d = dict(
            Level="country",
            Name=name,
            OfficialName=ofn,
            OtherNames="|".join(nms),
            Country=c.alpha_2,
            ISOa3=c.alpha_3,
            M49Code=c.numeric,
        )
        a = [d.get(k, pd.NA) for k in rds.BASIC_COL_TYPES.keys()]
        rds.data.loc[c.alpha_2] = a


def import_pycountry_subdivisions(rds: RegionDataset, countries):
    if isinstance(countries, str):
        countries = [countries]
    countries = frozenset(countries)

    for sd in pycountry.subdivisions:
        if sd.country_code not in countries:
            continue
        if sd.parent_code is not None:
            continue
        name = unidecode.unidecode(sd.name)
        d = dict(Level="subdivision", Name=name, OfficialName=sd.name, Country=sd.country_code,)
        a = [d.get(k, pd.NA) for k in rds.BASIC_COL_TYPES.keys()]
        rds.data.loc[sd.code] = a
