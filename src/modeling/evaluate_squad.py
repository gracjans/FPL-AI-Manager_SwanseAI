import pandas as pd

from src.data.data_loader import load_average_pts


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
        players = predictions_merged[predictions_merged[position] == 1]
        # add players to df_top_11 dataframe
        df_top_11 = df_top_11.append(players.head(formation[i]), ignore_index=True)

    # get 'name', 'total_points_next_gameweek', 'transfers_balance', 'value' columns from df_top_11
    df_squad = df_top_11[['name', 'total_points_next_gameweek', 'transfers_balance', 'value']]

    df_total_points = df_top_11.total_points_next_gameweek.sum()

    return df_squad, df_total_points


def get_average_pts(season: str, gameweek: int):
    """
    Returns the average squad points of FPL player in the given season and gameweek.
    """

    average_pts = load_average_pts()
    return average_pts.loc[average_pts['GW'] == gameweek, [f'AVG_PTS_{season.replace("-", "/")}']].values[0][0]
