import os

import pandas as pd
import mlflow

from src.data.data_loader import load_average_pts


def get_average_pts(season: str, gameweek: int):
    """
    Returns the average squad points of FPL player in the given season and gameweek.
    """

    average_pts = load_average_pts()
    return average_pts.loc[average_pts['GW'] == gameweek, [f'AVG_PTS_{season.replace("-", "/")}']].values[0][0]


def squad_selection_without_constraints(predictions_merged: pd.DataFrame, season: str, gameweek: int):
    """
    Selects the best predicted squad for the given season and gameweek without squad value constraints.
    """

    # get data from predictions_merged only for selected season and gameweek
    predictions_merged = predictions_merged[(predictions_merged.season == season) & (predictions_merged.GW == gameweek)]
    # sort predictions_merged by predicted points in descending order
    predictions_merged = predictions_merged.sort_values(by='predicted_total_points_next_gameweek', ascending=False)

    # get first row from predictions_merged and double 'total_points_next_gameweek' value, because this player would be chosen as a capitan
    predictions_merged.iloc[0, predictions_merged.columns.get_loc('total_points_next_gameweek')] *= 2

    # separate players by position and get dataframe with best squad in given formation
    df_top_11 = pd.DataFrame()
    positions = ['position_GK', 'position_DEF', 'position_MID', 'position_FWD']
    formation = [1, 3, 5, 2]
    for i, position in enumerate(positions):
        # get players with the given position
        players = predictions_merged[predictions_merged[position] == 1].head(formation[i])
        # add players to df_top_11 dataframe
        df_top_11 = pd.concat((df_top_11, players), ignore_index=True)

    # get 'name', 'total_points_next_gameweek', 'transfers_balance', 'value' columns from df_top_11
    df_squad = df_top_11[['name', 'total_points_next_gameweek', 'value']]

    df_total_points = df_top_11.total_points_next_gameweek.sum()

    return df_squad, df_total_points


def evaluate_selected_squad_without_constraints(predictions_merged: pd.DataFrame, test_subset: tuple, model_name: str):
    """
    Evaluates selected squad without value constraints compared to the average points gained by real FPL players
    Log results (comparison of AI squad and average FPL player squad) to MLFlow.
    """
    mlruns_path = os.path.dirname('file:\\' + os.path.dirname(os.path.dirname(__file__))) + '\\mlruns'
    mlflow.set_tracking_uri(mlruns_path)

    with mlflow.start_run(experiment_id='0'):
        # get best squad and total points for each gameweek from test_subset
        results = []
        season_gameweeks = []
        for season in test_subset:
            for gameweek in season[1]:
                results.append(squad_selection_without_constraints(predictions_merged, season[0], gameweek))
                season_gameweeks.append((season[0], gameweek))

        # get total points gained by selected squad
        selected_squad_points = []
        for result in results:
            selected_squad_points.append(result[1])

        # get average points gained by real FPL players
        real_player_average_points = []
        for gameweek in season_gameweeks:
            real_player_average_points.append(get_average_pts(gameweek[0], gameweek[1]))

        mlflow.log_param('test subset', test_subset)
        mlflow.log_metric("selected squad", sum(selected_squad_points))
        mlflow.log_metric("real player average", sum(real_player_average_points))
        mlflow.log_metric("difference", sum(selected_squad_points) - sum(real_player_average_points))
        mlflow.log_param("model name", model_name)

        return results, selected_squad_points, real_player_average_points, season_gameweeks
