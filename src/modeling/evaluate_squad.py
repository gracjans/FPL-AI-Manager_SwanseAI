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

    # get goalkeepers from predictions_merged_27 (with 1 in 'position_GK' column)
    df_gk = predictions_merged[predictions_merged.position_GK == 1]
    df_def = predictions_merged[predictions_merged.position_DEF == 1]
    df_mid = predictions_merged[predictions_merged.position_MID == 1]
    df_fwd = predictions_merged[predictions_merged.position_FWD == 1]

    # get one top row from df_gk, three top from df_def, five top from df_mid, two top from df_fwd and concatenate them into one dataframe
    df_top_11 = pd.concat([df_gk.head(1), df_def.head(3), df_mid.head(5), df_fwd.head(2)])

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
