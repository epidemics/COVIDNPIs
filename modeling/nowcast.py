import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from epimodel import RegionDataset, read_csv
from modeling.nowcasting.nowcasting import exp_model

rds = RegionDataset.load("../data/regions.csv")
csse_ds = read_csv("../data/CSSE.csv")


def exp_csse(data, actual_pred=None, date=None):
    case_to_pred = "Confirmed"
    days_prev = 7
    days_to_pred = 7

    # Getting all the regions
    regions = data.index.get_level_values(0).unique().to_list()
    data_x = np.linspace(1, days_prev, days_prev)

    for region in regions:
        reg_data = data.loc[region]
        reg_data_prev = reg_data[-1 * days_prev :]
        data_y = reg_data_prev[case_to_pred].to_list()

        if actual_pred is None:
            actual_pred = data_y[-1]
        if date is None:
            date = reg_data_prev.index[-1]

        print("-----------------")
        print(region)
        print(data_x, data_y, days_prev, days_to_pred, actual_pred)

        pred_next_week = None

        try:
            pred_next_week = exp_model(
                data_x, data_y, date, days_prev, days_to_pred, actual_pred
            )
        except ValueError:
            print(region, "data has NaNs")
        except Exception as inst:
            print("There was an error doing the forecasting")
            print(type(inst), inst)

        print(region, pred_next_week)
