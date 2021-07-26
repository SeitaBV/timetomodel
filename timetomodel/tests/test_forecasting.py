import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz
from statsmodels.tools.sm_exceptions import MissingDataError

from timetomodel import forecasting
from timetomodel.tests import utils as test_utils

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
    # instead of using the 2019 data in our dummy model, we explicitly provide features ourselves
    # the model was trained on a linearly increasing range of integers, so
    # with lags 2 and 3 having values 892 and 891, we expect an outcome of 894
    # the model was trained on an external regressor that was always a constant 5, so
    # yet another 5 will provide no information
    features = pd.DataFrame(
        index=pd.date_range(start=dt, end=dt, freq="H"),
        data={"my_outcome_l2": 892, "my_outcome_l3": 891, "Regressor1": 5},
    )
    fc = forecasting.make_forecast_for(specs, features, model)
    assert abs(fc - 894) <= TOLERANCE


def test_make_one_forecast_with_transformation():
    """ Given a simple linear model with a transformation, try to make a forecast. """
    model, specs = test_utils.create_dummy_model_state(
        DATA_START,
        data_range_in_hours=24,
        outcome_feature_transformation=test_utils.MyAdditionTransformation(addition=7),
    ).split()
    dt = datetime(2020, 1, 22, 22, tzinfo=pytz.timezone("Europe/Amsterdam"))
    # instead of using the 2019 data in our dummy model, we explicitly provide features ourselves
    # the model was trained on a linearly increasing range of integers, so
    # with lags 2 and 3 having values 892 and 891, we expect an outcome of 894 (after the back-transformation)
    # the model was trained on an external regressor that was always a constant 5,
    # so yet another 5 will provide no information
    feature_data = specs.outcome_var.feature_transformation.transform_series(
        pd.Series([891, 892])
    )
    features = pd.DataFrame(
        index=pd.date_range(start=dt, end=dt, freq="H"),
        data={
            "my_outcome_l2": feature_data[1],
            "my_outcome_l3": feature_data[0],
            "Regressor1": 5,
        },
    )
    fc = forecasting.make_forecast_for(specs, features, model)
    assert abs(fc - 894) <= TOLERANCE


def test_rolling_forecast():
    """Using the simple linear model, create a rolling forecast"""
    model, specs = test_utils.create_dummy_model_state(
        DATA_START, data_range_in_hours=24
    ).split()
    h0 = 3  # first 3 hours can't be predicted,lacking the lagged outcome variable
    hn = 26  # only 2 additional forecast can be made, because the lowest lag is 2 hours
    start = DATA_START + timedelta(hours=h0)
    end = DATA_START + timedelta(hours=hn)
    forecasts = forecasting.make_rolling_forecasts(start, end, specs)[0]
    expected_values = range(h0, hn)
    for forecast, expected_value in zip(forecasts, expected_values):
        assert abs(forecast - expected_value) < TOLERANCE


def test_rolling_forecast_with_refitting(caplog):
    """Also rolling forecasting, but with re-fitting the model in between.
    We'll test if the expected number of re-fittings happened.
    Also, the model we end up with should not be the one we started with."""
    caplog.set_level(logging.DEBUG, logger="timetomodel.forecasting")
    model, specs = test_utils.create_dummy_model_state(
        DATA_START, data_range_in_hours=192
    ).split()
    start = DATA_START + timedelta(hours=70)
    end = DATA_START + timedelta(hours=190)
    forecasts, final_model_state = forecasting.make_rolling_forecasts(start, end, specs)
    expected_values = specs.outcome_var.load_series(
        time_window=(start, end),
        expected_frequency=timedelta(hours=1),
        check_time_window=True,
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


def test_missing_data_warning(caplog):

    caplog.set_level(logging.WARNING, logger="timetomodel.forecasting")
    with pytest.raises(MissingDataError):
        test_utils.create_dummy_model_state(
            DATA_START,
            data_range_in_hours=24,
            regressor_feature_transformation=test_utils.MyNanIntroducingTransformation(),
        )

    missing_data_logs = [log for log in caplog.records if "missing data" in log.message]
    assert len(missing_data_logs) == 1
