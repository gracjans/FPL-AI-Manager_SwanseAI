import pandas as pd


def merge_reversed_data_with_predictions(model, x_test, y_test, x_test_reversed):
    """
    Merges the reversed data with the predictions.
    """

    y_pred = pd.Series(model.predict(x_test).reshape(-1, ), index=y_test.index, name='predicted_total_points_next_gameweek')
    return pd.concat([y_pred, x_test_reversed], axis=1)