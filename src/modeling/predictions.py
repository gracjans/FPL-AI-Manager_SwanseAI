import numpy as np
import pandas as pd


def merge_reversed_data_with_predictions(model: object, x_test: np.array, y_test: np.array, x_test_reversed: pd.DataFrame):
    """
    Merges the reversed data with the predictions.
    """

    y_pred = pd.Series(model.predict(x_test).reshape(-1, ), index=y_test.index, name='predicted_total_points_next_gameweek')
    return pd.concat([y_pred, x_test_reversed], axis=1)