import datetime
import io

import countryinfo
import numpy as np
import pandas as pd
import pycountry

from datalib.region_data import RegionDataset
from datalib.import_utils import import_pycountry_coutries, import_pycountry_subdivisions

s = RegionDataset()
import_pycountry_coutries(s)
import_pycountry_subdivisions(s, ["US", "CA", "UK", "AU", "CN"])
s = RegionDataset.from_csv("data/regions.csv")
for r in s.regions:
    if r.Name.endswith(' Sheng'):
        r.AllNames.append(r.Name)
        r['Name'] = r.Name[:-len(' Sheng')]
s.to_csv("data/regions.csv")
