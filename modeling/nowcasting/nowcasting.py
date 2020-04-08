import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from epimodel import RegionDataset, read_csv


def exp_model(data_x, data_y, date, days_prev, days_to_pred, actual_pred=None):
    days_prev = int(days_prev)
    days_to_pred = int(days_to_pred)

    # Define exponential fit function
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    if actual_pred is None:
        actual_pred = data_y[-1]

    popt, pcov = curve_fit(func, data_x, data_y, maxfev=2000)

    datelist = pd.date_range(start=date, periods=days_to_pred).tolist()

    data_xp = np.linspace(days_prev + 1, days_prev + days_to_pred, days_to_pred)

    pred_next_week = [
        [datelist[i], int(func(data_xp[i], *popt))] for i in range(days_to_pred)
    ]

    return pred_next_week
