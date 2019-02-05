from datetime import datetime, timedelta
import logging

import pandas as pd
import pytz

from ts_forecasting_pipeline import forecasting
from ts_forecasting_pipeline.tests import utils as test_utils


"""
Test the forecasting functionality. As we need models and feature frames as well, these tests work well
as integration tests.
"""


DATA_START = datetime(2019, 1, 22, 15, tzinfo=pytz.UTC)
TOLERANCE = 0.01


def test_make_one_forecast():
    """ Given a simple linear model, try to make a forecast. """
    model, specs = test_utils.create_dummy_model_state(
        DATA_START, data_range_in_hours=24
    ).split()
    dt = datetime(2020, 1, 22, 22, tzinfo=pytz.timezone("Europe/Amsterdam"))
    features = pd.DataFrame(
        index=pd.DatetimeIndex(start=dt, end=dt, freq="H"),
        data={"my_outcome_l1": 892, "my_outcome_l2": 891, "Regressor1": 5},
    )
    fc = forecasting.make_forecast_for(specs, features, model)
    assert abs(fc - 893) <= TOLERANCE


def test_make_one_forecast_with_transformation():
    """ Given a simple linear model with a transformation, try to make a forecast. """
    model, specs = test_utils.create_dummy_model_state(
        DATA_START,
        data_range_in_hours=24,
        outcome_feature_transformation=test_utils.MyAdditionTransformation(addition=7),
    ).split()
    dt = datetime(2020, 1, 22, 22, tzinfo=pytz.timezone("Europe/Amsterdam"))
    feature_data = specs.outcome_var.feature_transformation.transform_series(
        pd.Series([891, 892])
    )
    features = pd.DataFrame(
        index=pd.DatetimeIndex(start=dt, end=dt, freq="H"),
        data={
            "my_outcome_l1": feature_data[1],
            "my_outcome_l2": feature_data[0],
            "Regressor1": 5,
        },
    )
    fc = forecasting.make_forecast_for(specs, features, model)
    assert abs(fc - 893) <= TOLERANCE


def test_rolling_forecast():
    """Using the simple linear model, create a rolling forecast"""
    model, specs = test_utils.create_dummy_model_state(
        DATA_START, data_range_in_hours=24
    ).split()
    start = DATA_START + timedelta(hours=18)
    end = DATA_START + timedelta(hours=20)
    forecasts = forecasting.make_rolling_forecasts(start, end, specs)[0]
    expected_values = specs.outcome_var.load_series(
        expected_frequency=timedelta(hours=1)
    ).loc[start:end][:-1]
    for forecast, expected_value in zip(forecasts, expected_values):
        assert abs(forecast - expected_value) < TOLERANCE


def test_rolling_forecast_with_refitting(caplog):
    """ Also rolling forecasting, but with re-fitting the model in between.
    We'll test if the expected number of re-fittings happened.
    Also, the model we end up with should not be the one we started with."""
    caplog.set_level(logging.INFO, logger="ts_forecasting_pipeline.forecasting")
    model, specs = test_utils.create_dummy_model_state(
        DATA_START, data_range_in_hours=192
    ).split()
    start = DATA_START + timedelta(hours=70)
    end = DATA_START + timedelta(hours=190)
    forecasts, final_model_state = forecasting.make_rolling_forecasts(start, end, specs)
    expected_values = specs.outcome_var.load_series(
        expected_frequency=timedelta(hours=1)
    ).loc[start:end][:-1]
    for forecast, expected_value in zip(forecasts, expected_values):
        assert abs(forecast - expected_value) < TOLERANCE
    refitting_logs = [
        log for log in caplog.records if "Fitting new model" in log.message
    ]
    remodel_frequency_in_hours = int(specs.remodel_frequency.total_seconds() / 3600)
    expected_log_times = [remodel_frequency_in_hours]
    while max(expected_log_times) < 190:
        expected_log_times.append(max(expected_log_times) + remodel_frequency_in_hours)
    assert len(refitting_logs) == len([elt for elt in expected_log_times if elt >= 70])
    assert model is not final_model_state.model
