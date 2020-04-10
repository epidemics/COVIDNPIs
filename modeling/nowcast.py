import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from epimodel import RegionDataset, read_csv
from modeling.nowcasting.nowcasting import exp_model


def exp_csse(data, actual_pred=None, date=None):
    days_prev = 7
    days_to_pred = 14

    nowcasting = []
    # Getting all the regions
    regions = data.index.get_level_values(0).unique().to_list()
    data_x = np.linspace(1, days_prev, days_prev)

    for region in regions:
        reg_data = data.loc[region]
        reg_data_prev = reg_data[-1 * days_prev :]
        data_y_Confirmed = reg_data_prev["Confirmed"].to_list()
        data_y_Deaths = reg_data_prev["Deaths"].to_list()
        data_y_Recovered = reg_data_prev["Recovered"].to_list()
        data_y_Active = reg_data_prev["Active"].to_list()

        if actual_pred is None:
            actual_pred = data_y_Confirmed[-1]
        if date is None:
            date = reg_data_prev.index[-1]

        print("-----------------")
        print(region)
        print(data_x, data_y_Confirmed, days_prev, days_to_pred, actual_pred)

        datelist = pd.date_range(start=date, periods=days_to_pred).tolist()

        try:
            pred_next_week_confirmed = exp_model(
                data_x, data_y_Confirmed, date, days_prev, days_to_pred, actual_pred
            )
        except ValueError:
            pred_next_week_confirmed = [
                [datelist[i], "NaN"] for i in range(days_to_pred)
            ]
            print(region, "data has NaNs")
        except Exception as inst:
            pred_next_week_confirmed = [
                [datelist[i], "NaN"] for i in range(days_to_pred)
            ]
            print("There was an error doing the forecasting")
            print(type(inst))
            print(inst)

        try:
            pred_next_week_deaths = exp_model(
                data_x, data_y_Deaths, date, days_prev, days_to_pred, actual_pred
            )
        except ValueError:
            pred_next_week_deaths = [[datelist[i], "NaN"] for i in range(days_to_pred)]
            print(region, "data has NaNs")
        except Exception as inst:
            pred_next_week_deaths = [[datelist[i], "NaN"] for i in range(days_to_pred)]
            print("There was an error doing the forecasting")
            print(type(inst))
            print(inst)

        try:
            pred_next_week_recovered = exp_model(
                data_x, data_y_Recovered, date, days_prev, days_to_pred, actual_pred
            )
        except ValueError:
            pred_next_week_recovered = [
                [datelist[i], "NaN"] for i in range(days_to_pred)
            ]
            print(region, "data has NaNs")
        except Exception as inst:
            pred_next_week_recovered = [
                [datelist[i], "NaN"] for i in range(days_to_pred)
            ]
            print("There was an error doing the forecasting")
            print(type(inst))
            print(inst)

        try:
            pred_next_week_active = exp_model(
                data_x, data_y_Active, date, days_prev, days_to_pred, actual_pred
            )
        except ValueError:
            pred_next_week_active = [[datelist[i], "NaN"] for i in range(days_to_pred)]
            print(region, "data has NaNs")
        except Exception as inst:
            pred_next_week_active = [[datelist[i], "NaN"] for i in range(days_to_pred)]
            print("There was an error doing the forecasting")
            print(type(inst))
            print(inst)

        pred_next_week = [
            [
                pred_next_week_recovered[i][1],
                pred_next_week_confirmed[i][1],
                pred_next_week_deaths[i][1],
                pred_next_week_active[i][1],
            ]
            for i in range(len(pred_next_week_confirmed))
        ]

        code = [region]
        index = pd.MultiIndex.from_product((code, datelist), names=("Code", "Date"))

        df3 = pd.DataFrame(
            data=pred_next_week,
            index=index,
            columns=["Recovered", "Confirmed", "Deaths", "Active"],
        )

        index2 = pd.MultiIndex.from_product(
            (code, reg_data.index), names=("Code", "Date")
        )
        df4 = pd.DataFrame(
            data=reg_data.values,
            index=index2,
            columns=["Recovered", "Confirmed", "Deaths", "Active"],
        )
        df4 = df4.append(df3)

        nowcasting.append(df4)

    nc_df = pd.concat(nowcasting)
    return nc_df


if __name__ == "__main__":
    rds = RegionDataset.load("../data/regions.csv")
    csse_ds = read_csv("../data/johns-hopkins.csv")

    nowcasting_df = exp_csse(csse_ds)
    nowcasting_df.to_csv("nowcasting.csv")
