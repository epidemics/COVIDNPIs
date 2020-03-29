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
s.to_csv("data/regions.csv")
