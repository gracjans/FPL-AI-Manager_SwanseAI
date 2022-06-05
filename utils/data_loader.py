import os
import pandas


def load_data(season):
    root_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = root_dir + f"\\data\\Fantasy-Premier-League\\{season}\\gws\\merged_gw.csv"
    try:
        return pandas.read_csv(data_path, encoding='latin-1')
    except FileNotFoundError:
        raise FileNotFoundError("ERROR: Please enter valid season! Available seasons range from '2016-17' to '2021-22'")
