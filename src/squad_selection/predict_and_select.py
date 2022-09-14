import os
import fsspec
from pathlib import Path
import joblib
import requests

import tensorflow as tf
import pandas as pd

from src.features.data_engineering import preprocess_prediction_data
from src.squad_selection.select_team import select_team, print_selected_team

from src.squad_selection.select_transfer import TransferOptimiser, print_selected_transfer


def download_newest_fpl_data(season: str):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    secret = open(root_dir + "/secret.txt", "r")

    print('Downloading newest FPL data')

    destination = Path(os.path.dirname(os.getcwd()) + f'\\data\\raw\\Fantasy-Premier-League\\{season}')
    destination.mkdir(exist_ok=True, parents=True)

    fs = fsspec.filesystem("github", org="vaastav", repo="Fantasy-Premier-League", username='gracjans', token=secret.read())
    listed_files = fs.ls(f"data/{season}/")

    # do not download folders with many data that is not used
    listed_files.remove(f'data/{season}/players')
    listed_files.remove(f'data/{season}/understat')

    fs.get(listed_files, destination.as_posix(), recursive=True)

    print('Download complete')


def get_manager_squad(manager_id: int, gameweek: int):

    url = f'https://fantasy.premierleague.com/api/entry/{manager_id}/event/{gameweek}/picks/'
    r = requests.get(url)
    picks = r.json()

    squad = []
    for player in picks['picks']:
        squad.append(player['element'])

    return squad


def get_actual_chance_playing_next_round():

    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url)
    bootstrap = r.json()

    chance_playing = {}
    for player in bootstrap['elements']:
        chance_playing[player['id']] = player['chance_of_playing_next_round']

    return chance_playing


def predict_and_select_team(season: str, gameweek: int, model_path_from_root: str, scaler_path_from_root: str,
                            download_newest_data: bool = True, get_actual_chance_playing: bool = True, position_separated: bool = False,
                            budget_now: float = 100.0):
    if download_newest_data:
        download_newest_fpl_data(season)

    rolling_columns = ['assists', 'bonus', 'bps', 'clean_sheets',
                       'creativity', 'goals_conceded', 'goals_scored',
                       'ict_index', 'influence', 'minutes',
                       'saves', 'selected', 'player_team_score', 'opponent_team_score', 'threat',
                       'total_points', 'transfers_in', 'transfers_out',
                       'value', 'yellow_cards']

    rolling_columns_gk = ['bonus', 'bps', 'clean_sheets', 'goals_conceded', 'influence', 'minutes',
                          'penalties_saved', 'saves', 'selected', 'player_team_score', 'opponent_team_score',
                          'total_points', 'transfers_in', 'transfers_out','value']

    rolling_columns_field = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded', 'goals_scored',
                             'ict_index', 'influence', 'minutes', 'selected', 'player_team_score', 'opponent_team_score',
                             'threat', 'total_points', 'transfers_in', 'transfers_out', 'value', 'yellow_cards']

    times = ['all', 6, 3]
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if position_separated:
        x_prediction_field, x_target_field = preprocess_prediction_data(season, gameweek, rolling_columns=rolling_columns_field,
                                                                        rolling_times=times, opponent_team_stats=True, position='field')
        model_field = tf.keras.models.load_model(root_dir + model_path_from_root[1])
        x_scaler_field = joblib.load(root_dir + scaler_path_from_root[1])

        y_pred_field = pd.Series(model_field.predict(x_scaler_field.transform(x_prediction_field)).reshape(-1, ), index=x_prediction_field.index, name='predicted_total_points_next_gameweek')
        prediction_df_field = pd.concat([y_pred_field, x_target_field], axis=1)

        x_prediction_gk, x_target_gk = preprocess_prediction_data(season, gameweek, rolling_columns=rolling_columns_gk,
                                                                  rolling_times=times, opponent_team_stats=True, position='gk')
        model_gk = tf.keras.models.load_model(root_dir + model_path_from_root[0])
        x_scaler_gk = joblib.load(root_dir + scaler_path_from_root[0])

        y_pred_gk = pd.Series(model_gk.predict(x_scaler_gk.transform(x_prediction_gk)).reshape(-1, ), index=x_prediction_gk.index, name='predicted_total_points_next_gameweek')
        prediction_df_gk = pd.concat([y_pred_gk, x_target_gk], axis=1)

        # reset index
        prediction_df_field = prediction_df_field.reset_index(drop=True)
        prediction_df_gk = prediction_df_gk.reset_index(drop=True)

        prediction_df = pd.concat([prediction_df_gk, prediction_df_field], axis=0)

    else:
        x_prediction, x_target = preprocess_prediction_data(season, gameweek, rolling_columns=rolling_columns,
                                                            rolling_times=times, opponent_team_stats=True, position='all')

        model = tf.keras.models.load_model(root_dir + model_path_from_root[0])
        x_scaler = joblib.load(root_dir + scaler_path_from_root[0])

        y_pred = pd.Series(model.predict(x_scaler.transform(x_prediction)).reshape(-1, ), index=x_prediction.index, name='predicted_total_points_next_gameweek')
        prediction_df = pd.concat([y_pred, x_target], axis=1)

    prediction_df_sum = prediction_df.groupby(['name', 'element', 'position', 'value', 'team', 'GW', 'season']).sum().reset_index()
    list_of_opponents = prediction_df.groupby(['name', 'element', 'team', 'GW', 'season'])['opponent_next_gameweek'].apply(list).reset_index()['opponent_next_gameweek']

    prediction_df_sum['next_4_GWs_matches'] = list_of_opponents.apply(lambda x: len(x))
    prediction_df_sum['opponent_next_gameweek'] = list_of_opponents

    prediction_df_sum['value'] = prediction_df_sum['value'] / 10

    prediction_df_sum['position_id'] = prediction_df_sum['position'].replace({'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})

    # sort by element
    prediction_df_sum = prediction_df_sum.sort_values(by=['element'])

    # reset index
    prediction_df_sum = prediction_df_sum.reset_index(drop=True)

    if get_actual_chance_playing:
        actual_chance_playing = get_actual_chance_playing_next_round()
        for id in actual_chance_playing:
            if actual_chance_playing[id] is not None and actual_chance_playing[id] != 100:
                prediction_df_sum.loc[prediction_df_sum['element'] == id, 'predicted_total_points_next_gameweek'] = \
                    prediction_df_sum.loc[prediction_df_sum['element'] == id, 'predicted_total_points_next_gameweek'] * (int(actual_chance_playing[id]) / 100)

    decisions, captain_decisions, sub_decisions = select_team(prediction_df_sum.predicted_total_points_next_gameweek.values,
                                                              prediction_df_sum.value.values, prediction_df_sum.position_id.values,
                                                              prediction_df_sum.team.values, total_budget=budget_now, sub_factor=0.15)

    print_selected_team(prediction_df_sum, decisions, captain_decisions, sub_decisions)


def predict_and_select_transfer(season: str, gameweek: int, model_path_from_root: list, scaler_path_from_root: list,
                                download_newest_data: bool = True, get_actual_chance_playing: bool = True, position_separated: bool = False,
                                budget_now: int = 0):
    """Model path from root = ['gk model path', 'field model path'] if position separated is True"""
    # TODO: merge this function with predict_and_select_team, it's almos the same!
    if download_newest_data:
        download_newest_fpl_data(season)

    rolling_columns = ['assists', 'bonus', 'bps', 'clean_sheets',
                       'creativity', 'goals_conceded', 'goals_scored',
                       'ict_index', 'influence', 'minutes',
                       'saves', 'selected', 'player_team_score', 'opponent_team_score', 'threat',
                       'total_points', 'transfers_in', 'transfers_out',
                       'value', 'yellow_cards']

    rolling_columns_gk = ['bonus', 'bps', 'clean_sheets', 'goals_conceded', 'influence', 'minutes',
                          'penalties_saved', 'saves', 'selected', 'player_team_score', 'opponent_team_score',
                          'total_points', 'transfers_in', 'transfers_out','value']

    rolling_columns_field = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded', 'goals_scored',
                             'ict_index', 'influence', 'minutes', 'selected', 'player_team_score', 'opponent_team_score',
                             'threat', 'total_points', 'transfers_in', 'transfers_out', 'value', 'yellow_cards']

    times = ['all', 6, 3]
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if position_separated:
        x_prediction_field, x_target_field = preprocess_prediction_data(season, gameweek, rolling_columns=rolling_columns_field,
                                                                        rolling_times=times, opponent_team_stats=True, position='field')
        model_field = tf.keras.models.load_model(root_dir + model_path_from_root[1])
        x_scaler_field = joblib.load(root_dir + scaler_path_from_root[1])

        y_pred_field = pd.Series(model_field.predict(x_scaler_field.transform(x_prediction_field)).reshape(-1, ), index=x_prediction_field.index, name='predicted_total_points_next_gameweek')
        prediction_df_field = pd.concat([y_pred_field, x_target_field], axis=1)

        x_prediction_gk, x_target_gk = preprocess_prediction_data(season, gameweek, rolling_columns=rolling_columns_gk,
                                                                  rolling_times=times, opponent_team_stats=True, position='gk')
        model_gk = tf.keras.models.load_model(root_dir + model_path_from_root[0])
        x_scaler_gk = joblib.load(root_dir + scaler_path_from_root[0])

        y_pred_gk = pd.Series(model_gk.predict(x_scaler_gk.transform(x_prediction_gk)).reshape(-1, ), index=x_prediction_gk.index, name='predicted_total_points_next_gameweek')
        prediction_df_gk = pd.concat([y_pred_gk, x_target_gk], axis=1)

        # reset index
        prediction_df_field = prediction_df_field.reset_index(drop=True)
        prediction_df_gk = prediction_df_gk.reset_index(drop=True)

        prediction_df = pd.concat([prediction_df_gk, prediction_df_field], axis=0)

    else:
        x_prediction, x_target = preprocess_prediction_data(season, gameweek, rolling_columns=rolling_columns,
                                                            rolling_times=times, opponent_team_stats=True, position='all')

        model = tf.keras.models.load_model(root_dir + model_path_from_root[0])
        x_scaler = joblib.load(root_dir + scaler_path_from_root[0])

        y_pred = pd.Series(model.predict(x_scaler.transform(x_prediction)).reshape(-1, ), index=x_prediction.index, name='predicted_total_points_next_gameweek')
        prediction_df = pd.concat([y_pred, x_target], axis=1)

    prediction_df_sum = prediction_df.groupby(['name', 'element', 'position', 'value', 'team', 'GW', 'season']).sum().reset_index()
    list_of_opponents = prediction_df.groupby(['name', 'element', 'team', 'GW', 'season'])['opponent_next_gameweek'].apply(list).reset_index()['opponent_next_gameweek']

    prediction_df_sum['next_3_GWs_matches'] = list_of_opponents.apply(lambda x: len(x))
    prediction_df_sum['opponent_next_gameweek'] = list_of_opponents

    prediction_df_sum['value'] = prediction_df_sum['value'] / 10

    prediction_df_sum['position_id'] = prediction_df_sum['position'].replace({'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})

    # sort by element
    prediction_df_sum = prediction_df_sum.sort_values(by=['element'])

    # reset index
    prediction_df_sum = prediction_df_sum.reset_index(drop=True)

    if get_actual_chance_playing:
        actual_chance_playing = get_actual_chance_playing_next_round()
        for id in actual_chance_playing:
            if actual_chance_playing[id] is not None and actual_chance_playing[id] != 100:
                prediction_df_sum.loc[prediction_df_sum['element'] == id, 'predicted_total_points_next_gameweek'] = \
                    prediction_df_sum.loc[prediction_df_sum['element'] == id, 'predicted_total_points_next_gameweek'] * (int(actual_chance_playing[id]) / 100)

    opt = TransferOptimiser(prediction_df_sum.predicted_total_points_next_gameweek.values,
                            prediction_df_sum.value.values, prediction_df_sum.value.values,
                            prediction_df_sum.position_id.values, prediction_df_sum.team.values)

    squad = get_manager_squad(6703146, gameweek)

    # get indices of players in squad (index of row with matching 'element' value)
    indices = [prediction_df_sum.index[prediction_df_sum['element'] == player_id].tolist()[0] for player_id in squad]

    transfer_in_decisions, transfer_out_decisions, decisions, sub_decisions, captain_decisions = opt.solve(indices, budget_now=budget_now, sub_factor=0.15)

    print('\nSelected Transfers:')
    print_selected_transfer(prediction_df_sum, transfer_in_decisions, transfer_out_decisions)
    print('\nSelected Squad:')
    print_selected_team(prediction_df_sum, decisions, captain_decisions, sub_decisions)

    return prediction_df_sum.sort_values(by=['predicted_total_points_next_gameweek'], ascending=False).head(15)
