### Initial imports
import logging
import numpy as np
import pymc3 as pm
import seaborn as sns
import pandas as pd

sns.set_style("ticks")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from epimodel.pymc3_models import cm_effect
from epimodel.pymc3_models.cm_effect.datapreprocessor import ICLDataPreprocessor, DataPreprocessor
import argparse
import theano

argparser = argparse.ArgumentParser()
argparser.add_argument("--m", dest="model", type=int)
args = argparser.parse_args()

if __name__ == "__main__":

    region_info = [
        ("Andorra", "AD", "AND"),
        ("Austria", "AT", "AUT"),
        ("Albania", "AL", "ALB"),
        ("Bosnia and Herzegovina", "BA", "BIH"),
        ("Belgium", "BE", "BEL"),
        ("Bulgaria", "BG", "BGR"),
        ("Switzerland", "CH", "CHE"),
        ("Czech Republic", "CZ", "CZE"),
        ("Germany", "DE", "DEU"),
        ("Denmark", "DK", "DNK"),
        ("Estonia", "EE", "EST"),
        ("Spain", "ES", "ESP"),
        ("Finland", "FI", "FIN"),
        ("France", "FR", "FRA"),
        ("United Kingdom", "GB", "GBR"),
        ("Georgia", "GE", "GEO"),
        ("Greece", "GR", "GRC"),
        ("Croatia", "HR", "HRV"),
        ("Hungary", "HU", "HUN"),
        ("Ireland", "IE", "IRL"),
        ("Israel", "IL", "ISR"),
        ("Iceland", "IS", "ISL"),
        ("Italy", "IT", "ITA"),
        ("Lithuania", "LT", "LTU"),
        ("Latvia", "LV", "LVA"),
        ("Malta", "MT", "MLT"),
        ("Morocco", "MA", "MAR"),
        ("Mexico", "MX", "MEX"),
        ("Malaysia", "MY", "MYS"),
        ("Netherlands", "NL", "NLD"),
        ("Norway", "NO", "NOR"),
        ("New Zealand", "NZ", "NZL"),
        ("Poland", "PL", "POL"),
        ("Portugal", "PT", "PRT"),
        ("Romania", "RO", "ROU"),
        ("Serbia", "RS", "SRB"),
        ("Sweden", "SE", "SWE"),
        ("Singapore", "SG", "SGP"),
        ("Slovenia", "SI", "SVN"),
        ("Slovakia", "SK", "SVK"),
        ("South Africa", "ZA", "ZAF"),
    ]

    region_info.sort(key=lambda x: x[0])
    region_names = list([x for x, _, _ in region_info])
    regions_epi = list([x for _, x, _ in region_info])
    regions_threecode = list([x for _, _, x in region_info])


    def eur_to_epi_code(x):
        if x in regions_threecode:
            return regions_epi[regions_threecode.index(x)]
        else:
            return "not found"


    dp = ICLDataPreprocessor(drop_HS=True)
    dp.N_smooth = 1
    data = dp.preprocess_data("notebooks/final_data/data_final.csv", "notebooks/final_data/ICL.csv")

    eur_df = pd.read_csv("notebooks/final_data/eur_data.csv", parse_dates=["dateRep"], infer_datetime_format=True)
    eur_df['dateRep'] = pd.to_datetime(eur_df['dateRep'], utc=True)
    epi_codes = [eur_to_epi_code(cc) for cc in eur_df["countryterritoryCode"]]
    dti = pd.to_datetime(eur_df['dateRep'], utc=True)

    eur_df.index = pd.MultiIndex.from_arrays([epi_codes, dti])

    columns_to_drop = ["day", "month", "year", "countriesAndTerritories", "geoId", "popData2018", "continentExp",
                       "dateRep", "countryterritoryCode"]

    for col in columns_to_drop:
        del eur_df[col]

    eur_df = eur_df.loc[regions_epi]

    AltNewCases = np.zeros((len(data.Rs), len(data.Ds)))
    AltNewDeaths = np.zeros((len(data.Rs), len(data.Ds)))

    for r_i, r in enumerate(data.Rs):
        for d_i, d in enumerate(data.Ds):
            c_vals = eur_df.loc[r]
            if d in c_vals.index:
                AltNewCases[r_i, d_i] = c_vals["cases"].loc[d]
                AltNewDeaths[r_i, d_i] = c_vals["deaths"].loc[d]

    AltNewDeaths[AltNewDeaths < 0] = np.nan
    AltNewCases[AltNewCases < 0] = np.nan

    data.NewCases = np.ma.masked_invalid(AltNewCases.astype(theano.config.floatX))
    data.NewDeaths = np.ma.masked_invalid(AltNewDeaths.astype(theano.config.floatX))

    if args.model == 0:
        with cm_effect.models.CMCombined_ICL_NoNoise_2(data, None) as model:
            model.build_model()

    elif args.model == 1:
        with cm_effect.models.CMCombined_ICL_NoNoise_3(data, None) as model:
            model.build_model()

    elif args.model == 2:
        with cm_effect.models.CMCombined_ICL_NoNoise_4(data, None) as model:
            model.build_model()

    elif args.model == 3:
        with cm_effect.models.CMCombined_ICL_NoNoise_5(data, None) as model:
            model.build_model()

    elif args.model == 4:
        with cm_effect.models.CMCombined_ICL_NoNoise_6(data, None) as model:
            model.build_model()

    elif args.model == 5:
        with cm_effect.models.CMCombined_ICL_NoNoise_7(data, None) as model:
            model.build_model()

    elif args.model == 6:
        dp = DataPreprocessor(drop_HS=True)
        dp.N_smooth = 1
        data = dp.preprocess_data("notebooks/final_data/data_final.csv")
        with cm_effect.models.CMCombined_ICL_NoNoise_7(data, None) as model:
            model.build_model()

    with model.model:
        model.trace = pm.sample(1500, chains=6, target_accept=0.9)

    np.savetxt(f"icl_bugs/model_{args.model+2}.csv", model.trace.CMReduction)
