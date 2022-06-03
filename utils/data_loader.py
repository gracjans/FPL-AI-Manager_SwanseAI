import os
import pandas


def load_data(season):
    try:
        data_path = os.getcwd() + "\\data\\Fantasy-Premier-League"
        merged_gw_2021_path = data_path + f"\\{season}\\gws\\merged_gw.csv"
        return pandas.read_csv(merged_gw_2021_path, encoding='latin-1')
    except FileNotFoundError:
        raise FileNotFoundError("ERROR: Please enter valid season! Available seasons range from '2016-17' to '2021-22'")
