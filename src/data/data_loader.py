import os
import pandas as pd


def load_merged_gw(season: str):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = root_dir + f"\\data\\raw\\Fantasy-Premier-League\\{season}\\gws\\merged_gw.csv"
    return __read_file(data_path)


def load_players_raw(season: str):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = root_dir + f"\\data\\raw\\Fantasy-Premier-League\\{season}\\players_raw.csv"
    return __read_file(data_path)


def __read_file(data_path: str):
    try:
        return pd.read_csv(data_path, encoding='latin-1')
    except FileNotFoundError:
        raise FileNotFoundError("ERROR: Please enter valid season! Available seasons range from '2016-17' to '2021-22'")


def load_average_pts():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = root_dir + f"\\data\\external\\average_pts.csv"
    return pd.read_csv(data_path, encoding='latin-1')
