import os
import fsspec
from pathlib import Path
import joblib

import tensorflow as tf
import pandas as pd

from src.features.data_engineering import preprocess_prediction_data
from src.squad_selection.select_team import select_team, print_selected_team


def download_newest_fpl_data(season: str):
    print('Downloading newest FPL data')

    destination = Path(os.path.dirname(os.getcwd()) + f'\\data\\raw\\Fantasy-Premier-League\\{season}')
    destination.mkdir(exist_ok=True, parents=True)

    fs = fsspec.filesystem("github", org="vaastav", repo="Fantasy-Premier-League", username='gracjans', token='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')
    listed_files = fs.ls(f"data/{season}/")

    # do not download folders with many data that is not used
    listed_files.remove(f'data/{season}/players')
    listed_files.remove(f'data/{season}/understat')

    fs.get(listed_files, destination.as_posix(), recursive=True)

    print('Download complete')


def predict_and_select_team(season: str, gameweek: int, model_path_from_root: str, scaler_path_from_root: str, download_newest_data: bool = True):
    if download_newest_data:
        download_newest_fpl_data(season)

    rolling_columns = ['assists', 'bonus', 'bps', 'clean_sheets',
                       'creativity', 'goals_conceded', 'goals_scored',
                       'ict_index', 'influence', 'minutes',
                       'saves', 'selected', 'player_team_score', 'opponent_team_score', 'threat',
                       'total_points', 'transfers_in', 'transfers_out',
                       'value', 'yellow_cards']

    times = ['all', 6, 3]

    x_prediction, x_target = preprocess_prediction_data(season, gameweek, rolling_columns=rolling_columns, rolling_times=times, opponent_team_stats=True)

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model = tf.keras.models.load_model(root_dir + model_path_from_root)
    x_scaler = joblib.load(root_dir + scaler_path_from_root)

    y_pred = pd.Series(model.predict(x_scaler.transform(x_prediction)).reshape(-1, ), index=x_prediction.index, name='predicted_total_points_next_gameweek')
    prediction_df = pd.concat([y_pred, x_target], axis=1)

    prediction_df_sum = prediction_df.groupby(['name', 'element', 'position', 'value', 'team', 'GW', 'season']).sum().reset_index()
    list_of_opponents = prediction_df.groupby(['name', 'element', 'team', 'GW', 'season'])['opponent_next_gameweek'].apply(list).reset_index()['opponent_next_gameweek']

    prediction_df_sum['next_4_GWs_matches'] = list_of_opponents.apply(lambda x: len(x))
    prediction_df_sum['opponent_next_gameweek'] = list_of_opponents

    prediction_df_sum['value'] = prediction_df_sum['value'] / 10

    prediction_df_sum['position_id'] = prediction_df_sum['position'].replace({'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})

    decisions, captain_decisions, sub_decisions = select_team(prediction_df_sum.predicted_total_points_next_gameweek.values,
                                                              prediction_df_sum.value.values, prediction_df_sum.position_id.values,
                                                              prediction_df_sum.team.values, total_budget=100, sub_factor=0.15)

    print_selected_team(prediction_df_sum, decisions, captain_decisions, sub_decisions)


def predict_and_select_transfer(season: str, gameweek: int, model_path_from_root: str, scaler_path_from_root: str):
    download_newest_fpl_data(season)

    rolling_columns = ['assists', 'bonus', 'bps', 'clean_sheets',
                       'creativity', 'goals_conceded', 'goals_scored',
                       'ict_index', 'influence', 'minutes',
                       'saves', 'selected', 'player_team_score', 'opponent_team_score', 'threat']