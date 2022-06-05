import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.data_loader import load_data


def preprocess_baseline_data(season):
    """
    Preprocesses the data for the baseline model.
    """

    # Drop the columns that are not needed for the baseline model
    data = load_data(season)

    # add 'total_points_next_gameweek' column where total_points_next_gameweek = total_points from next 'GW' for each 'element'
    data['total_points_next_gameweek'] = data.groupby('element')['total_points'].shift(-1)

    # create data_2021_processed dataframe without 'name', 'team', 'fixture', 'kickoff_time', 'opponent_team', 'round', 'team_h_score', 'team_a_score'
    data_processed = data.drop(
        ['name', 'team', 'fixture', 'kickoff_time', 'opponent_team', 'round', 'team_h_score', 'team_a_score'], axis=1)

    # extract 'GW', 'element' and 'total_points_next_gameweek' from data_2021_processed
    data_extract_target = data_processed[['GW', 'element', 'total_points_next_gameweek']]

    # drop 'GW' and 'element' from data_2021_processed
    data_processed = data_processed.drop(['GW', 'element'], axis=1)

    # one-hot encode 'position' column
    data_processed = pd.get_dummies(data_processed, columns=['position'])

    # change 'was_home' column to binary
    data_processed['was_home'] = data_processed['was_home'].map({True: 1, False: 0})

    # drop rows with NaN values
    data_2021_processed = data_processed.dropna()

    X = data_2021_processed.drop(['total_points_next_gameweek'], axis=1)
    y = data_2021_processed['total_points_next_gameweek']

    # scale X data
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    X_2021_scaled = pd.DataFrame(X_scaler.fit_transform(X), columns=X.columns)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_2021_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
