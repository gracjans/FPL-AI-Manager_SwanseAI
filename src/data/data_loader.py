import os
import aiohttp
import pandas as pd
from understat import Understat

import nest_asyncio
nest_asyncio.apply()


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


async def get_league_table(season, start_date=None, end_date=None):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        table = await understat.get_league_table("EPL", season, start_date=start_date, end_date=end_date)
        return pd.DataFrame(table[1:], columns=table[0])
