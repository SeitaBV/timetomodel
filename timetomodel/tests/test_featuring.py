from datetime import datetime, timedelta
import logging
import pytz

from timetomodel.featuring import get_time_steps
from timetomodel.speccing import DEFAULT_RATIO_TRAINING_TESTING_DATA
from timetomodel.tests import utils as test_utils


DATA_START = datetime(2019, 1, 22, 15, tzinfo=pytz.UTC)


def test_get_time_steps(caplog):
    """Specs and debug logs should show the training and testing windows we expect to see."""
    caplog.set_level(logging.DEBUG)
    lags = [2, 3]
    training_and_testing_period = timedelta(hours=6)
    model, specs = test_utils.create_dummy_model_state(
        DATA_START, data_range_in_hours=24, lags=lags, training_and_testing_period=training_and_testing_period
    ).split()
    assert specs.start_of_training == DATA_START + timedelta(hours=max(lags))
    assert specs.end_of_training == specs.start_of_training + DEFAULT_RATIO_TRAINING_TESTING_DATA * training_and_testing_period
    assert specs.start_of_testing == specs.end_of_training
    assert specs.end_of_testing == specs.start_of_training + training_and_testing_period

    dt_indices = get_time_steps("train", specs)
    assert f"Start of training: {specs.start_of_training.strftime('%Y-%m-%d %H:%M')}" in caplog.text  # our training start leaves room for lags
    assert f"End of training: {specs.end_of_training.strftime('%Y-%m-%d %H:%M')}" in caplog.text  # 4 out of 6 hours are training, the other 2 testing
    assert dt_indices[0] == specs.start_of_training
    assert dt_indices[-1] == specs.end_of_training - specs.frequency
    dt_indices = get_time_steps("test", specs)
    assert f"Start of testing: {specs.start_of_testing.strftime('%Y-%m-%d %H:%M')}" in caplog.text  # our training start leaves room for lags
    assert f"End of testing: {specs.end_of_testing.strftime('%Y-%m-%d %H:%M')}" in caplog.text  # 4 out of 6 hours are training, the other 2 testing
    assert dt_indices[0] == specs.start_of_testing
    assert dt_indices[-1] == specs.end_of_testing - specs.frequency
