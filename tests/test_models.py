from datetime import datetime, timedelta

import pytest
import pandas as pd
import pytz
from statsmodels.api import OLS

from ts_forecasting_pipeline import speccing, modelling, forecasting


DATA_START = datetime(2019, 1, 22, 15, tzinfo=pytz.timezone("Europe/Amsterdam"))
TOLERANCE = 0.01


def create_dummy_model() -> modelling.ModelState:
    """
    Create a dummy model. Try out two different ways to define Series specs.
    """
    dt_range = pd.date_range(DATA_START, DATA_START + timedelta(hours=24), freq="15T")
    outcome_values = [0]
    regressor_values = [5]
    for i in range(1, len(dt_range)):
        outcome_values.append(outcome_values[i-1] + 1)
        regressor_values.append(5)
    outcome_series = pd.Series(index=dt_range, data=outcome_values)
    regressor_series = pd.Series(index=dt_range, data=regressor_values)
    specs = modelling.ModelSpecs(
        outcome_var=speccing.ObjectSeriesSpecs(outcome_series, "my_outcome"),
        model=OLS,
        lags=[1, 2],
        #frequency=timedelta(hours=1),
        horizon=timedelta(minutes=30),
        regressors=[regressor_series],
        start_of_training=DATA_START + timedelta(hours=1),
        end_of_testing=DATA_START + timedelta(hours=12),
    )
    return modelling.ModelState(
        modelling.create_fitted_model(specs, version="0.1", save=False), specs
    )


def test_make_forecast():
    """ Given a simple linear model, try to make a forecast. """
    model, specs = create_dummy_model().split()
    dt = datetime(2020, 1, 22, 22, tzinfo=pytz.timezone("Europe/Amsterdam"))
    features = pd.DataFrame(index=pd.DatetimeIndex(start=dt, end=dt, freq="15T"),
                            data={"my_outcome_l1": 892, "my_outcome_l2": 891, "Regressor1": 5})
    fc = forecasting.make_forecast_for(specs, features, model)
    assert abs(fc - 893) <= TOLERANCE


def test_rolling_forecast():
    """Using the simple linear model, create a rolling forecast"""
    model, specs = create_dummy_model().split()
    start = DATA_START + timedelta(hours=18)
    end = DATA_START + timedelta(hours=20)
    forecasts = forecasting.make_rolling_forecasts(start, end, specs)[0]
    expected_values = specs.outcome_var.load_series()[18*4:20:4]
    print("CORRECT VALUES:")
    print(expected_values)
    print("FORECASTS:")
    print(forecasts)
    assert 1 == 2
    for forecast, expected_value in zip(forecasts, expected_values):
        print(forecast, expected_value)
        assert abs(forecast - expected_value) > TOLERANCE


# TODO: test refitting the model in between?


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
