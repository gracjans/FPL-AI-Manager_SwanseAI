import asyncio

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.data.data_loader import load_merged_gw, get_league_table, load_master_team_list, load_understat_team_stats
from src.data.data_loader import load_players_raw
from src.features.utils import idx_to_team_name, str_date_months_back, str_date_days_forward


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

    # sort data by kickoff_time
    data_merged.sort_values('kickoff_time', inplace=True, ignore_index=True)

    return data_merged


def update_team_score_feature(df):
    """
    Create feature 'player_team_score' - team_h_score if was_home, team_a_score otherwise and 'opponent_team_score' likewise
    """
    player_team_score = df.apply(lambda row: row['team_h_score'] if row['was_home'] else row['team_a_score'], axis=1)
    opponent_team_score = df.apply(lambda row: row['team_a_score'] if row['was_home'] else row['team_h_score'], axis=1)

    df.insert(list(df.columns).index('team_a_score'), 'player_team_score', player_team_score)
    df.insert(list(df.columns).index('team_h_score'), 'opponent_team_score', opponent_team_score)
    df.drop(['team_h_score', 'team_a_score'], axis=1, inplace=True)

    return df


def create_rolling_features(df, rolling_columns, times):
    for t in times:
        t_str = '-all' if t == 'all' else '-' + str(t)
        t = df.groupby(['season', 'element'], as_index=False).size()['size'][0] if t == 'all' else t
        for col in rolling_columns:
            insert_loc = list(df.columns).index(col) + 1
            df.insert(insert_loc, col + t_str, df.groupby(['season', 'element'], as_index=False)[col].rolling(t, min_periods=1).mean()[col])
    return df


def scrape_team_stats(row, master_team_list, table_dict):
    columns_to_get = ['Position', 'PPDA', 'OPPDA', 'G', 'GA', 'xG', 'NPxG', 'xGA', 'NPxGA', 'NPxGD', 'DC', 'ODC', 'xPTS']
    opponent_team = idx_to_team_name(master_team_list, row['opponent_next_gameweek'], row['season'])

    season_year = row['season'].split('-')[0]
    date = str_date_days_forward(row['kickoff_time'].split('T')[0], 2)

    key = date + '_' + opponent_team

    # if key is in the dict, pass
    if key in table_dict:
        return

    date_back = str_date_months_back(date, 2)

    table = asyncio.run(get_league_table(season_year, date_back, date))

    # get row from table where Team == opponent_team
    table_opponent = table.loc[table['Team'] == opponent_team]

    cols_normalize = table_opponent.filter(items=columns_to_get[3:]).columns
    table_opponent[cols_normalize] = table_opponent[cols_normalize].divide(table_opponent['M'], axis=0)

    value = table_opponent[columns_to_get].add_prefix('opponent_next_')

    table_dict[key] = value.to_dict()
    len_dict = len(table_dict)
    if len_dict % 50 == 0:
        print(len_dict)

    return True


def get_oponent_team_stats(row, master_team_list, team_stats):
    opponent_team = idx_to_team_name(master_team_list, row['opponent_next_gameweek'], row['season'])
    date = str_date_days_forward(row['kickoff_time'].split('T')[0], 2)
    key = date + '_' + opponent_team
    return pd.DataFrame(team_stats[row['season']][key])


def preprocess_seasons_data(data: pd.DataFrame = None, random_split: bool = True, test_subset: tuple = None, season: str = None,
                            rolling_features: bool = False, rolling_columns: list = None, rolling_times: list = None,
                            opponent_team_stats: bool = False, encode_position: bool = True):
    """
    Preprocesses the merged seasons data.

    Args:
        data: data to preprocess. If None, preprocesses the merged seasons data.
        random_split: If True, the data is split into train and test sets randomly,
            if False, the data is split into train and test sets according to the test_subset parameter.
        test_subset: If random_split is False, this parameter specifies subset of the data used for the test set
            test_subset example: (['2016-17', [35,36,37,38,39]], ['2021-22', [27,28,29,30]], ['season', [gws]])
        season: If specified, only this season is preprocessed.
        rolling_features: If True, the rolling features are created.
        rolling_times: If rolling_features is True, this parameter specifies the rolling times.
        rolling_columns: If rolling_features is True, this parameter specifies the features to be rolled.
        opponent_team_stats: If True, there are features about next gameweek opponent team stats added.
        encode_position: If true, the position feature is one hot encoded, else drop this feature.
    """
    target_features = ['name', 'GW', 'element', 'total_points_next_gameweek', 'season']

    if data is None:
        data = get_merged_seasons_data()
    else:
        data = data.copy()

    # if season parameter is specified, only select data from that season
    if season is not None:
        data = data[data['season'] == season]

    # Change 'team_h_score' and 'team_a_score' to 'player_team_score' and 'opponent_team_score'
    data_processed = update_team_score_feature(data)

    # add column where total_points_next_gameweek = total_points from next 'GW' for each player (element)
    data_processed['total_points_next_gameweek'] = data_processed.sort_values('kickoff_time').groupby(['season', 'element'])['total_points'].shift(-1)

    if opponent_team_stats:
        seasons = ['2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22']
        team_stats = {}
        for season in seasons:
            team_stats[season] = load_understat_team_stats(season)

        # add column where opponent_next_gameweek = opponent_team from next 'GW' for each player (element)
        data_processed['opponent_next_gameweek'] = data_processed.sort_values('kickoff_time').groupby(['season', 'element'])['opponent_team'].shift(-1)
        data_processed = data_processed.dropna(subset=['opponent_next_gameweek']).astype({'opponent_next_gameweek': int})

        master_team_list = load_master_team_list()

        # for every row, get mean stats from last two months for each next gameweek opponent team
        opponent_data = data_processed.apply(lambda row: get_oponent_team_stats(row, master_team_list, team_stats), axis=1)

        df_opponent_data = pd.concat([r for r in opponent_data], ignore_index=True)
        data_processed = pd.concat([data_processed, df_opponent_data.set_index(data_processed.index)], axis=1)

        data_processed = data_processed.drop(['opponent_next_gameweek'], axis=1)

    if rolling_features:
        data_processed = create_rolling_features(data_processed, rolling_columns, rolling_times)

    # Drop the columns that are not needed for now
    data_processed = data_processed.drop(['fixture', 'kickoff_time', 'opponent_team', 'round',
                                          'transfers_balance', 'was_home'], axis=1)

    if encode_position:
        # one-hot encode 'position' column
        data_processed = pd.get_dummies(data_processed, columns=['position'])
    else:
        data_processed = data_processed.drop(['position'], axis=1)

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
        x_train, x_test, y_train, y_test = __subset_train_test_split(x_data_scaled, y, test_subset)

    # extract target and drop it from x data
    x_train_data_extract_target = x_train[target_features]
    x_train.drop(target_features, axis=1, inplace=True)

    x_test_data_extract_target = x_test[target_features]
    x_test.drop(target_features, axis=1, inplace=True)

    return (x_train, y_train), (x_test, y_test), (x_train_data_extract_target, x_test_data_extract_target), x_scaler


def __subset_train_test_split(x: pd.DataFrame, y: pd.DataFrame, test_subset: tuple):
    """
    Split data into train and test sets according to each list in test_subset
    """
    x_train, x_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for subset in test_subset:
        x_test = pd.concat([x_test, x.loc[x['season'].isin([subset[0]]) & x['GW'].isin(subset[1])]])
        y_test = pd.concat([y_test, y.loc[x['season'].isin([subset[0]]) & x['GW'].isin(subset[1])]])

    x_train = pd.concat([x_train, x.drop(x_test.index)])
    y_train = pd.concat([y_train, y.drop(y_test.index)])

    return x_train, x_test, y_train, y_test
