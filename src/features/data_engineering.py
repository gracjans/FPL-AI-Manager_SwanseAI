import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.data.data_loader import load_merged_gw
from src.data.data_loader import load_players_raw


def reverse_processing(x_data: np.array, x_data_scaler: MinMaxScaler, extracted_target: pd.DataFrame = None):
    """
    Reverses the preprocessing of the data and optionally concatenates it with the extracted target data
    """

    # reverse scaling of x data
    x = pd.DataFrame(x_data_scaler.inverse_transform(x_data), index=x_data.index, columns=x_data.columns)

    if extracted_target is not None:
        return pd.concat([extracted_target, x], axis=1)
    else:
        return x


def get_merged_seasons_data():
    """
    Load and merge together data from all seasons, deleting incompatible columns and adding position.
    """

    seasons = ['2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22']

    # load merged gw data for all seasons
    data = {}
    for season in seasons:
        data[season] = load_merged_gw(season)

    # get common features for every season
    common_features = data['2018-19'].columns.intersection(data['2019-20'].columns).intersection(data['2020-21'].columns)

    # select common features columns for all seasons
    for season in data:
        data[season] = data[season][common_features]

    # load 'players_raw' for every season
    players_raw = {}
    for season in seasons:
        players_raw[season] = load_players_raw(season)

    # leave only 'id' and 'element_type' columns
    player_position = {}
    for season in players_raw:
        player_position[season] = players_raw[season][['id', 'element_type']].rename(columns={'id': 'element', 'element_type': 'position'})

    # change values from element type to 1: GK, 2: DEF, 3: MID, 4: FWD
    for season in player_position:
        player_position[season]['position'].replace({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}, inplace=True)

    # add position column to every season dataset
    for season in player_position:
        data[season] = pd.merge(data[season], player_position[season], on='element', how='left')

    # add 'season' column to every season dataset
    for season in data:
        data[season]['season'] = season

    # make data one single dataframe
    data_merged = pd.concat(data.values(), ignore_index=True)

    return data_merged


def preprocess_seasons_data(random_split: bool = True, test_subset: tuple = None, season: str = None):
    """
    Preprocesses the merged seasons data.

    Args:
        random_split: If True, the data is split into train and test sets randomly,
            if False, the data is split into train and test sets according to the test_subset parameter.
        test_subset: If random_split is False, this parameter specifies subset of the data used for the test set
            test_subset example: (['2016-17', [35,36,37,38,39]], ['2021-22', [27,28,29,30]], ['season', [gws]])
        season: If specified, only this season is preprocessed.
    """
    target_features = ['name', 'GW', 'element', 'total_points_next_gameweek', 'season']
    data = get_merged_seasons_data()

    # if season parameter is specified, only select data from that season
    if season is not None:
        data = data[data['season'] == season]

    # add column where total_points_next_gameweek = total_points from next 'GW' for each player (element)
    data['total_points_next_gameweek'] = data.sort_values('kickoff_time').groupby(['season', 'element'])['total_points'].shift(-1)

    # Drop the columns that are not needed for the baseline model
    data_processed = data.drop(['fixture', 'kickoff_time', 'opponent_team', 'round', 'team_h_score', 'team_a_score'], axis=1)

    # one-hot encode 'position' column
    data_processed = pd.get_dummies(data_processed, columns=['position'])

    # change 'was_home' column to binary
    data_processed['was_home'] = data_processed['was_home'].map({True: 1, False: 0})

    # drop rows with NaN values
    data_processed.dropna(inplace=True)

    # extract 'name', 'GW', 'element' and 'total_points_next_gameweek' from data_processed
    data_extract_target = data_processed[target_features]

    # prepare x, y data
    x = data_processed.drop(target_features, axis=1)
    y = data_processed['total_points_next_gameweek']

    # scale x data
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    x_data_scaled = pd.DataFrame(x_scaler.fit_transform(x), index=x.index, columns=x.columns)

    # concatenate data_extract_target and x_data_scaled
    x_data_scaled = pd.concat([data_extract_target, x_data_scaled], axis=1)

    if random_split:
        # split data into train and test sets randomly
        x_train, x_test, y_train, y_test = train_test_split(x_data_scaled, y, test_size=0.2, random_state=42)
    else:
        # split data into train and test sets according to each list in test_subset    # TODO: make it separate function
        x_train, x_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for subset in test_subset:
            x_test = pd.concat([x_test, x_data_scaled.loc[x_data_scaled['season'].isin([subset[0]]) & x_data_scaled['GW'].isin(subset[1])]])
            y_test = pd.concat([y_test, y.loc[x_data_scaled['season'].isin([subset[0]]) & x_data_scaled['GW'].isin(subset[1])]])

        x_train = pd.concat([x_train, x_data_scaled.drop(x_test.index)])
        y_train = pd.concat([y_train, y.drop(y_test.index)])

    # extract target and drop it from x data
    x_train_data_extract_target = x_train[target_features]
    x_train.drop(target_features, axis=1, inplace=True)

    x_test_data_extract_target = x_test[target_features]
    x_test.drop(target_features, axis=1, inplace=True)

    return (x_train, y_train), (x_test, y_test), (x_train_data_extract_target, x_test_data_extract_target), x_scaler
