"""
:code:`scaling.py`

Test results when scaling the number of new cases
"""

import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.data_preprocessor import preprocess_data

import argparse
import pickle

import pandas as pd

from scripts.sensitivity_analysis.utils import *

from datetime import date

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

argparser = argparse.ArgumentParser()
argparser.add_argument('--scaling_type', dest='scaling_type', type=str, help='Scaling type.'
                                                                             'Options are `simple` (time-constant), or'
                                                                             '`variable` (time-varying)')
argparser.add_argument('--rg', dest='rg', type=str, help='Region to leave out - alpha 2 code')
add_argparse_arguments(argparser)

if __name__ == '__main__':
    args, extras = argparser.parse_known_args()

    data = preprocess_data(get_data_path(), last_day='2020-05-30')
    data.mask_reopenings(print_out=False)

    ep = EpidemiologicalParameters()
    model_class = get_model_class_from_str(args.model_type)

    if args.scaling_type == 'simple':
        adjustment = np.array([np.random.choice([0.25, .5, 2, 3, 4]) for i in range(len(data.Rs))])
        data.NewCases = data.NewCases * adjustment.reshape((41, 1))
    elif args.scaling_type == 'variable':
        country_codes = {
            "AFG": "AF",
            "ALA": "AX",
            "ALB": "AL",
            "DZA": "DZ",
            "ASM": "AS",
            "AND": "AD",
            "AGO": "AO",
            "AIA": "AI",
            "ATA": "AQ",
            "ATG": "AG",
            "ARG": "AR",
            "ARM": "AM",
            "ABW": "AW",
            "AUS": "AU",
            "AUT": "AT",
            "AZE": "AZ",
            "BHS": "BS",
            "BHR": "BH",
            "BGD": "BD",
            "BRB": "BB",
            "BLR": "BY",
            "BEL": "BE",
            "BLZ": "BZ",
            "BEN": "BJ",
            "BMU": "BM",
            "BTN": "BT",
            "BOL": "BO",
            "BES": "BQ",
            "BIH": "BA",
            "BWA": "BW",
            "BVT": "BV",
            "BRA": "BR",
            "IOT": "IO",
            "BRN": "BN",
            "BGR": "BG",
            "BFA": "BF",
            "BDI": "BI",
            "CPV": "CV",
            "KHM": "KH",
            "CMR": "CM",
            "CAN": "CA",
            "CYM": "KY",
            "CAF": "CF",
            "TCD": "TD",
            "CHL": "CL",
            "CHN": "CN",
            "CXR": "CX",
            "CCK": "CC",
            "COL": "CO",
            "COM": "KM",
            "COD": "CD",
            "COG": "CG",
            "COK": "CK",
            "CRI": "CR",
            "CIV": "CI",
            "HRV": "HR",
            "CUB": "CU",
            "CUW": "CW",
            "CYP": "CY",
            "CZE": "CZ",
            "DNK": "DK",
            "DJI": "DJ",
            "DMA": "DM",
            "DOM": "DO",
            "ECU": "EC",
            "EGY": "EG",
            "SLV": "SV",
            "GNQ": "GQ",
            "ERI": "ER",
            "EST": "EE",
            "SWZ": "SZ",
            "ETH": "ET",
            "FLK": "FK",
            "FRO": "FO",
            "FJI": "FJ",
            "FIN": "FI",
            "FRA": "FR",
            "GUF": "GF",
            "PYF": "PF",
            "ATF": "TF",
            "GAB": "GA",
            "GMB": "GM",
            "GEO": "GE",
            "DEU": "DE",
            "GHA": "GH",
            "GIB": "GI",
            "GRC": "GR",
            "GRL": "GL",
            "GRD": "GD",
            "GLP": "GP",
            "GUM": "GU",
            "GTM": "GT",
            "GGY": "GG",
            "GIN": "GN",
            "GNB": "GW",
            "GUY": "GY",
            "HTI": "HT",
            "HMD": "HM",
            "VAT": "VA",
            "HND": "HN",
            "HKG": "HK",
            "HUN": "HU",
            "ISL": "IS",
            "IND": "IN",
            "IDN": "ID",
            "IRN": "IR",
            "IRQ": "IQ",
            "IRL": "IE",
            "IMN": "IM",
            "ISR": "IL",
            "ITA": "IT",
            "JAM": "JM",
            "JPN": "JP",
            "JEY": "JE",
            "JOR": "JO",
            "KAZ": "KZ",
            "KEN": "KE",
            "KIR": "KI",
            "PRK": "KP",
            "KOR": "KR",
            "KWT": "KW",
            "KGZ": "KG",
            "LAO": "LA",
            "LVA": "LV",
            "LBN": "LB",
            "LSO": "LS",
            "LBR": "LR",
            "LBY": "LY",
            "LIE": "LI",
            "LTU": "LT",
            "LUX": "LU",
            "MAC": "MO",
            "MKD": "MK",
            "MDG": "MG",
            "MWI": "MW",
            "MYS": "MY",
            "MDV": "MV",
            "MLI": "ML",
            "MLT": "MT",
            "MHL": "MH",
            "MTQ": "MQ",
            "MRT": "MR",
            "MUS": "MU",
            "MYT": "YT",
            "MEX": "MX",
            "FSM": "FM",
            "MDA": "MD",
            "MCO": "MC",
            "MNG": "MN",
            "MNE": "ME",
            "MSR": "MS",
            "MAR": "MA",
            "MOZ": "MZ",
            "MMR": "MM",
            "NAM": "NA",
            "NRU": "NR",
            "NPL": "NP",
            "NLD": "NL",
            "NCL": "NC",
            "NZL": "NZ",
            "NIC": "NI",
            "NER": "NE",
            "NGA": "NG",
            "NIU": "NU",
            "NFK": "NF",
            "MNP": "MP",
            "NOR": "NO",
            "OMN": "OM",
            "PAK": "PK",
            "PLW": "PW",
            "PSE": "PS",
            "PAN": "PA",
            "PNG": "PG",
            "PRY": "PY",
            "PER": "PE",
            "PHL": "PH",
            "PCN": "PN",
            "POL": "PL",
            "PRT": "PT",
            "PRI": "PR",
            "QAT": "QA",
            "REU": "RE",
            "ROU": "RO",
            "RUS": "RU",
            "RWA": "RW",
            "BLM": "BL",
            "SHN": "SH",
            "KNA": "KN",
            "LCA": "LC",
            "MAF": "MF",
            "SPM": "PM",
            "VCT": "VC",
            "WSM": "WS",
            "SMR": "SM",
            "STP": "ST",
            "SAU": "SA",
            "SEN": "SN",
            "SRB": "RS",
            "SYC": "SC",
            "SLE": "SL",
            "SGP": "SG",
            "SXM": "SX",
            "SVK": "SK",
            "SVN": "SI",
            "SLB": "SB",
            "SOM": "SO",
            "ZAF": "ZA",
            "SGS": "GS",
            "SSD": "SS",
            "ESP": "ES",
            "LKA": "LK",
            "SDN": "SD",
            "SUR": "SR",
            "SJM": "SJ",
            "SWE": "SE",
            "CHE": "CH",
            "SYR": "SY",
            "TWN": "TW",
            "TJK": "TJ",
            "TZA": "TZ",
            "THA": "TH",
            "TLS": "TL",
            "TGO": "TG",
            "TKL": "TK",
            "TON": "TO",
            "TTO": "TT",
            "TUN": "TN",
            "TUR": "TR",
            "TKM": "TM",
            "TCA": "TC",
            "TUV": "TV",
            "UGA": "UG",
            "UKR": "UA",
            "ARE": "AE",
            "GBR": "GB",
            "UMI": "UM",
            "USA": "US",
            "URY": "UY",
            "UZB": "UZ",
            "VUT": "VU",
            "VEN": "VE",
            "VNM": "VN",
            "VGB": "VG",
            "VIR": "VI",
            "WLF": "WF",
            "ESH": "EH",
            "YEM": "YE",
            "ZMB": "ZM",
            "ZWE": "ZW"
        }

        trdf = pd.read_csv("scripts/sensitivity_analysis/under_ascertainment_estimates.txt",
                           parse_dates=["date"],
                           infer_datetime_format=True)

        trdf['region'] = trdf.iso_code.map(country_codes)
        trdf = trdf.set_index(["date"])

        trdf = trdf.append(pd.DataFrame(
            {
                'region': ['MT'],
                'median': [np.nan]
            }, index=[pd.to_datetime('2020-08-01')]))

        empty = pd.Series([np.nan], index=[pd.to_datetime('2020-01-01')])

        a = trdf.groupby('region')['median'].apply(lambda g:
                                                   (empty.append(g)
                                                    .asfreq('D')
                                                    ))

        test_rates = np.zeros((len(data.Rs), len(data.Ds)))
        for r_i, r in enumerate(data.Rs):
            for d_i, d in enumerate(data.Ds):
                test_rates[r_i, d_i] = a[(r, pd.to_datetime(d))]

        for r_i, r in enumerate(data.Rs):
            nz_inds = np.nonzero(np.logical_not(np.isnan(test_rates[r_i, :])))[0]

            if len(nz_inds) > 0:
                first_ind = nz_inds[0]
                first_val = test_rates[r_i, first_ind]
                test_rates[r_i, :first_ind] = first_val
            else:
                test_rates[r_i, :] = 1

        data.NewCases = data.NewCases / test_rates
        data.NewCases = np.ma.masked_array(np.around(data.NewCases).astype(int))
        data.Confirmed = np.cumsum(data.NewCases, axis=-1)
        data.NewCases.mask[data.NewCases < 0] = True
        data.NewCases.mask = data.Confirmed < 100
        data.mask_reopenings(print_out=False)
        data.mask_region(args.rg)

    bd = {**ep.get_model_build_dict(), **parse_extra_model_args(extras)}

    with model_class(data) as model:
        model.build_model(**bd)

    ta = get_target_accept_from_model_str(args.model_type)

    with model.model:
        model.trace = pm.sample(750, tune=500, chains=2, cores=2, max_treedepth=16,
                                target_accept=0.925, init='adapt_diag')

    import pickle
    pickle.dump(model.trace, open(f'{args.rg}.pkl', 'wb'))
