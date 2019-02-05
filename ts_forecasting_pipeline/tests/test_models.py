import pytest
import pandas as pd


@pytest.mark.skip(reason="Not implemented yet.")
def test_uneventful_forecasts_identified(tol=0.01):
    """
    Test if forecast model identifies uneventful forecasts

    If observation and regressor are both zero, then the regressor gave a perfect forecast.
    The forecast model should be good enough to pick up on that.
    We'll test for non-perfect forecasts for at most x percent of the times at which observation == regressor == 0.

    :param tol:         Tolerance in the percentage of forecasts where the tests fails
    :return:

    """

    # TODO: Create model
    # TODO: Make forecasts

    # TODO: Take only the validation set
    df = pd.DataFrame(
        data={"x": [1, 0, 0, 3, 0], "y": [45, 0, 0, 32, 91], "yhat": [46, 0, 1, 31, 90]}
    )  # temp

    #  TODO: Take only the part of the dataframe for which both x and y are zero
    df_zero = pd.DataFrame(data={"x": [0, 0], "y": [0, 0], "yhat": [0, 1]})  # temp

    # TODO: Count the number of times yhat is not zero
    num_non_zero_errors = 1  # temp

    assert num_non_zero_errors / len(df_zero) <= tol


@pytest.mark.skip(reason="Not implemented yet.")
def test_forecasts_within_limits(tol=0.01):
    """

    Test if forecasts are within limits

    If observations clearly show limits on the data (e.g. between 0 and 1),
    new forecasts should not exceed those limits more than x percent of the time.
    If this fails, either there are no clear limits on the data
    (e.g. due to a trend or due to seasonalities with a longer period than the duration of the training set),
    or we should reject the forecast model.

    :param tol:         Tolerance in the percentage of forecasts where the tests fails
    :return:

    """

    # TODO: Create model
    # TODO: Make forecasts

    # TODO: Take only the validation set
    running_limits = [0, 45]  # Running minimum and maximum from training set
    df = pd.DataFrame(
        data={
            "x": [1, 0, 0, 3, 0, 4],
            "y": [45, 0, 0, 32, 91, 89],
            "yhat": [46, 0, 1, 31, 90, 91],
        }
    )  # temp

    # TODO: Count the number of times yhat is lower then the running minimum (of y)
    num_min_breaches = 0  # temp

    # TODO: Count the number of times yhat is higher then the running maximum (of y)
    num_max_breaches = (
        2
    )  # temp (constraint violated when yhat was 46 and again when yhat was 90)

    assert num_min_breaches + num_max_breaches / len(df) <= tol
