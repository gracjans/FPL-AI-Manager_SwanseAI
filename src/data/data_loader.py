import asyncio
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
    await asyncio.sleep(0.5)
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        table = await understat.get_league_table("EPL", season, start_date=start_date, end_date=end_date)
        df_table = pd.DataFrame(table[1:], columns=table[0])
        df_table.insert(0, 'Position', df_table.index + 1)
        return df_table


def load_master_team_list():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = root_dir + f"\\data\\raw\\Fantasy-Premier-League\\master_team_list.csv"
    master_team_list_df = pd.read_csv(data_path, encoding='latin-1')

    # Rename the teams in the master team list, that they are compatible with the teams names from Understat data.
    master_team_list_df.replace('Man City', 'Manchester City', inplace=True)
    master_team_list_df.replace('Man Utd', 'Manchester United', inplace=True)
    master_team_list_df.replace('Spurs', 'Tottenham', inplace=True)
    master_team_list_df.replace('West Brom', 'West Bromwich Albion', inplace=True)
    master_team_list_df.replace('Newcastle', 'Newcastle United', inplace=True)
    master_team_list_df.replace('Wolves', 'Wolverhampton Wanderers', inplace=True)
    master_team_list_df.replace('Sheffield Utd', 'Sheffield United', inplace=True)

    return master_team_list_df
