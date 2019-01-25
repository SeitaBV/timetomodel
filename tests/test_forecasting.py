from datetime import datetime, timedelta
import logging

import pandas as pd
import pytz
from statsmodels.api import OLS

from ts_forecasting_pipeline import speccing, modelling, forecasting


"""
Test the forecasting functionality. As we need models and feature frames as well, these tests work well
as integration tests.
"""


DATA_START = datetime(2019, 1, 22, 15, tzinfo=pytz.UTC)
TOLERANCE = 0.01


def create_dummy_model(data_range_in_hours: int) -> modelling.ModelState:
    """
    Create a dummy model. data increases linearly, regressor is constant (useless).
    Use two different ways to define Series specs to test them.
    """
    dt_range = pd.date_range(
        DATA_START, DATA_START + timedelta(hours=data_range_in_hours), freq="1H"
    )
    outcome_values = [0]
    regressor_values = [5]
    for i in range(1, len(dt_range)):
        outcome_values.append(outcome_values[i - 1] + 1)
        regressor_values.append(5)
    outcome_series = pd.Series(index=dt_range, data=outcome_values)
    regressor_series = pd.Series(index=dt_range, data=regressor_values)
    specs = modelling.ModelSpecs(
        outcome_var=speccing.ObjectSeriesSpecs(outcome_series, "my_outcome"),
        model=OLS,
        lags=[1, 2],
        frequency=timedelta(hours=1),
        horizon=timedelta(minutes=120),
        remodel_frequency=timedelta(hours=48),
        regressors=[regressor_series],
        start_of_training=DATA_START + timedelta(hours=2),  # leaving room for NaN in lags
        end_of_testing=DATA_START + timedelta(hours=int(data_range_in_hours / 3)),
    )
    return modelling.ModelState(
        modelling.create_fitted_model(specs, version="0.1", save=False), specs
    )


def test_make_one_forecast():
    """ Given a simple linear model, try to make a forecast. """
    model, specs = create_dummy_model(data_range_in_hours=24).split()
    dt = datetime(2020, 1, 22, 22, tzinfo=pytz.timezone("Europe/Amsterdam"))
    features = pd.DataFrame(
        index=pd.DatetimeIndex(start=dt, end=dt, freq="H"),
        data={"my_outcome_l1": 892, "my_outcome_l2": 891, "Regressor1": 5},
    )
    fc = forecasting.make_forecast_for(specs, features, model)
    assert abs(fc - 893) <= TOLERANCE


def test_rolling_forecast():
    """Using the simple linear model, create a rolling forecast"""
    model, specs = create_dummy_model(data_range_in_hours=24).split()
    start = DATA_START + timedelta(hours=18)
    end = DATA_START + timedelta(hours=20)
    forecasts = forecasting.make_rolling_forecasts(start, end, specs)[0]
    expected_values = specs.outcome_var.load_series().loc[start:end][:-1]
    for forecast, expected_value in zip(forecasts, expected_values):
        assert abs(forecast - expected_value) < TOLERANCE


def test_rolling_forecast_with_refitting(caplog):
    """ Also rolling forecasting, but with re-fitting the model in between.
    We'll test if the expected number of re-fittings happened.
    Also, the model we end up with should not be the one we started with."""
    caplog.set_level(logging.INFO, logger="ts_forecasting_pipeline.forecasting")
    model, specs = create_dummy_model(data_range_in_hours=192).split()
    start = DATA_START + timedelta(hours=70)
    end = DATA_START + timedelta(hours=190)
    forecasts, final_model_state = forecasting.make_rolling_forecasts(start, end, specs)
    expected_values = specs.outcome_var.load_series().loc[start:end][:-1]
    for forecast, expected_value in zip(forecasts, expected_values):
        assert abs(forecast - expected_value) < TOLERANCE
    refitting_logs = [log for log in caplog.records if "Fitting new model" in log.message]
    remodel_frequency_in_hours = int(specs.remodel_frequency.total_seconds() / 3600)
    expected_log_times = [remodel_frequency_in_hours]
    while max(expected_log_times) < 190:
        expected_log_times.append(max(expected_log_times) + remodel_frequency_in_hours)
    assert len(refitting_logs) == len([elt for elt in expected_log_times if elt >= 70])
    assert model is not final_model_state.model


# TODO: add a regressor that is a forecast
